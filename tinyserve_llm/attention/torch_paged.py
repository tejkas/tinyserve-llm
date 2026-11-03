"""
Paged attention in plain PyTorch: every query attends over its sequence's keys in the paged cache.

It covers prefill, prefill on top of a cached prefix, and decode with one rule, because the step's
own K/V are written to the cache before attention runs. It is slow (a Python loop over sequences)
and exists so the whole engine runs and is tested on CPU; flash-attn's varlen call with a block
table computes the same thing on the GPU.
"""

from __future__ import annotations

import torch


def paged_attention_torch(
        q: torch.Tensor,             # [num_tokens, num_heads, head_dim], sequences packed back to back
        k_cache: torch.Tensor,       # [num_blocks, block_size, num_kv_heads, head_dim], may be a strided view
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,  # [batch, max_blocks_per_seq]
        cu_seqlens_q: list[int],     # [batch + 1] query offsets into q
        context_lens: list[int],     # [batch] keys per sequence, this step's tokens included
        scale: float | None = None,
) -> torch.Tensor:                   # [num_tokens, num_heads, head_dim]
    num_heads, head_dim = q.shape[1], q.shape[2]
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = num_heads // num_kv_heads
    scale = scale if scale is not None else head_dim ** -0.5
    out = torch.empty_like(q)
    for i, context_len in enumerate(context_lens):
        start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        num_queries = end - start
        blocks = block_tables[i, : -(-context_len // block_size)]
        # Gather this sequence's blocks in order, then trim the unused tail of the last one.
        k = k_cache[blocks].flatten(0, 1)[:context_len]  # [context_len, num_kv_heads, head_dim]
        v = v_cache[blocks].flatten(0, 1)[:context_len]
        # GQA: each kv head serves `group` consecutive query heads.
        k = k.repeat_interleave(group, dim=1).transpose(0, 1)  # [heads, context_len, head_dim]
        v = v.repeat_interleave(group, dim=1).transpose(0, 1)
        scores = (q[start:end].transpose(0, 1) @ k.transpose(1, 2)).float() * scale  # [heads, queries, keys]
        # The queries are the last num_queries positions, so query j sits at position
        # context_len - num_queries + j and sees keys up to and including that position.
        future = torch.ones(num_queries, context_len, dtype=torch.bool, device=q.device).triu(
            context_len - num_queries + 1)
        weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1).to(q.dtype)
        out[start:end] = (weights @ v).transpose(0, 1)
    return out
