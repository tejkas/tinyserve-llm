# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), MIT License,
# Copyright (c) 2025 Xingkai Yu. Based on nanovllm/models/qwen3.py and nanovllm/layers/*.

"""
Llama 3 forward pass over flat packed batches: input_ids and positions are [num_tokens] with
several sequences concatenated and no padding. Attention goes through an injected backend, so this
module never knows how KV is stored.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from tinyserve_llm.model.attention import AttentionBackend

# torch.compile on CPU takes seconds per kernel; tests and debugging can switch it off.
# Read at call time, so flipping it after import works.
COMPILE = os.environ.get("TINYSERVE_COMPILE", "1") != "0"


def maybe_compile(fn: Callable) -> Callable:
    """torch.compile on the same kernels nano-vllm compiles, unless COMPILE is off."""
    compiled = torch.compile(fn)  # lazy: nothing compiles until the first call

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return compiled(*args, **kwargs) if COMPILE else fn(*args, **kwargs)

    return wrapper


class RMSNorm(nn.Module):
    """
    RMSNorm with an optional fused residual add. The sum is normalised in fp32 before rounding,
    and the rounded sum is returned as the next residual.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # Out-of-place where nano works in place: x.float() is a no-op on fp32 input, so in-place ops
    # would clobber the caller's tensor (the residual). Inductor emits the same fused kernel.
    @maybe_compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight

    @maybe_compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight, residual

    def forward(
            self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)


class SiluAndMul(nn.Module):
    """SwiGLU gate: the fused gate_up output is [gate | up]."""

    @maybe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


def rope_settings(config: Any) -> tuple[float, dict | None]:
    """
    (rope_theta, scaling dict or None). transformers 5 folds rope_theta into rope_parameters;
    older versions and raw Hub config.json files keep rope_theta and rope_scaling at the top level.
    """
    params = getattr(config, "rope_parameters", None)
    if params:
        theta = params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        scaling = params
    else:
        theta = getattr(config, "rope_theta", 10000.0)
        scaling = getattr(config, "rope_scaling", None)
    if scaling is not None:
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type == "default":
            scaling = None
        elif rope_type != "llama3":
            raise NotImplementedError(f"rope_type {rope_type!r} is not supported")
    return float(theta), scaling


def llama3_inv_freq(
        inv_freq: torch.Tensor,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
) -> torch.Tensor:
    """
    Llama 3 context extension: wavelengths longer than the pretraining window are slowed by
    `factor`, short ones are left alone, and the band between is interpolated smoothly.
    Mirrors transformers' _compute_llama3_parameters.
    """
    old_context_len = original_max_position_embeddings
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Llama's "rotate half" layout: dim i pairs with dim i + head_dim/2, not with its neighbour.
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    Precomputed cos/sin table indexed by each token's position, so packed sequences at different
    offsets rotate in one gather. The table is fp32; build the model under a default dtype rather
    than calling .to(dtype), which would round it.
    """

    def __init__(self, head_dim: int, max_position: int, base: float, scaling: dict | None = None) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        if scaling is not None:
            inv_freq = llama3_inv_freq(
                inv_freq,
                factor=scaling["factor"],
                low_freq_factor=scaling["low_freq_factor"],
                high_freq_factor=scaling["high_freq_factor"],
                original_max_position_embeddings=scaling.get("original_max_position_embeddings", max_position),
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)  # [max_pos, 1, head_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @maybe_compile
    def forward(
            self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


class MergedLinear(nn.Linear):
    """
    Several bias-free projections fused into one GEMM (q|k|v, gate|up). The checkpoint stores
    them separately, so each is loaded into its slice of the output rows.
    """

    def __init__(self, input_size: int, output_sizes: list[int]) -> None:
        super().__init__(input_size, sum(output_sizes), bias=False)
        self.output_sizes = output_sizes

    def load_shard(self, shard_id: int, loaded_weight: torch.Tensor) -> None:
        offset = sum(self.output_sizes[:shard_id])
        target = self.weight.data.narrow(0, offset, self.output_sizes[shard_id])
        assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
        target.copy_(loaded_weight)


class LlamaAttention(nn.Module):

    def __init__(self, config: Any, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        assert not getattr(config, "attention_bias", False)
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = MergedLinear(config.hidden_size, [self.q_size, self.kv_size, self.kv_size])
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.rotary_emb = rotary_emb

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = attn(self.layer_idx, q, k, v)
        return self.o_proj(o.flatten(1, -1))


class LlamaMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        assert hidden_act == "silu"
        self.gate_up_proj = MergedLinear(hidden_size, [intermediate_size] * 2)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: Any, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        assert not getattr(config, "mlp_bias", False)
        self.self_attn = LlamaAttention(config, layer_idx, rotary_emb)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
            attn: AttentionBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The residual add is deferred into the next norm so it fuses with it.
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, attn)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self, config: Any) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        theta, scaling = rope_settings(config)
        # One table shared by every layer; it is identical across layers.
        rotary_emb = RotaryEmbedding(head_dim, config.max_position_embeddings, theta, scaling)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i, rotary_emb) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, attn)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    # HF checkpoint projection name -> (our fused module, its shard index)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        """[num_tokens] ids and per-sequence positions -> [num_tokens, hidden] final hidden states."""
        return self.model(input_ids, positions, attn)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Separate from forward so the caller can project only the rows it samples from."""
        return self.lm_head(hidden_states)
