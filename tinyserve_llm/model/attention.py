"""
The model hands q, k, v to an injected backend and never sees how KV is stored, so the cache can
change (paged, offloaded, ablated) without touching model code.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch


class AttentionBackend(Protocol):
    """
    Called once per layer with the projected, rotated q/k/v of every token in the flat batch.
    The backend owns caching, masking and which tokens belong to which sequence.
    """

    def __call__(
            self,
            layer_idx: int,
            q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
            k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
            v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    ) -> torch.Tensor:  # [num_tokens, num_heads, head_dim]
        ...


class DenseCausalAttention:
    """
    Per-sequence causal attention over packed sequences, with no cache. Every call sees the whole
    of each sequence, so it only serves full prefills; it exists as a test oracle.
    """

    def __init__(self, cu_seqlens: Sequence[int] | torch.Tensor, scale: float | None = None) -> None:
        # cu_seqlens follows flash-attn's varlen layout: [0, len0, len0 + len1, ...]
        self.cu_seqlens = [int(x) for x in cu_seqlens]
        self.scale = scale

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert self.cu_seqlens[0] == 0 and self.cu_seqlens[-1] == q.shape[0], "cu_seqlens must cover the batch"
        num_heads, head_dim = q.shape[1], q.shape[2]
        group = num_heads // k.shape[1]
        scale = self.scale if self.scale is not None else head_dim ** -0.5
        out = torch.empty_like(q)
        for start, end in zip(self.cu_seqlens[:-1], self.cu_seqlens[1:]):
            qs = q[start:end].transpose(0, 1)  # [heads, len, head_dim]
            # GQA: each kv head serves `group` consecutive query heads
            ks = k[start:end].repeat_interleave(group, dim=1).transpose(0, 1)
            vs = v[start:end].repeat_interleave(group, dim=1).transpose(0, 1)
            scores = (qs @ ks.transpose(1, 2)) * scale
            n = end - start
            future = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out[start:end] = (weights @ vs).transpose(0, 1)
        return out
