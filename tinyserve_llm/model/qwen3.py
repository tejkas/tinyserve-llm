# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), MIT License,
# Copyright (c) 2025 Xingkai Yu. Based on nanovllm/models/qwen3.py.

"""
Qwen3 forward pass over flat packed batches, the same contract as the Llama model. The architecture
is Llama's plus QK-norm: an RMSNorm over each head of q and k, applied before RoPE. head_dim is set
in the config rather than derived (Qwen3-0.6B: 16 heads of 128 on a 1024-wide residual stream).
The norm, RoPE and fused-projection layers are shared with llama.py.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tinyserve_llm.model.attention import AttentionBackend
from tinyserve_llm.model.llama import LlamaMLP, MergedLinear, RMSNorm, RotaryEmbedding, rope_settings


class Qwen3Attention(nn.Module):

    def __init__(self, config: Any, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        assert not getattr(config, "attention_bias", False)
        assert not getattr(config, "use_sliding_window", False), "sliding-window layers are not supported"
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = MergedLinear(config.hidden_size, [self.q_size, self.kv_size, self.kv_size])
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        # Normalised over head_dim, one weight vector shared by every head.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = attn(self.layer_idx, q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Any, layer_idx: int, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx, rotary_emb)
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


class Qwen3Model(nn.Module):

    def __init__(self, config: Any) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        theta, scaling = rope_settings(config)
        rotary_emb = RotaryEmbedding(head_dim, config.max_position_embeddings, theta, scaling)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i, rotary_emb) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, attn)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
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
        self.model = Qwen3Model(config)
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
