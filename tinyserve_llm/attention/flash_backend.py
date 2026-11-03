"""
flash-attn backend: paged KV writes and attention over block tables.

tinyserve_llm delegates attention math to flash-attn. This module is the
adapter between our block tables and flash-attn's paged KV interface:

    store_kv               write new K/V into their paged slots
    paged_attention_flash  flash_attn_varlen_func over block tables, for prefill and decode alike

flash-attn is CUDA-only, so it is imported lazily inside the functions that need it.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _store_kv_kernel(
        k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
        k_token_stride, v_token_stride, # elements between consecutive tokens of k and v
        block_stride, slot_stride,       # elements between blocks and between slots of the cache view
        block_size: tl.constexpr,
        row: tl.constexpr,
    ):
        token = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + token) # int64
        if slot == -1:
            return     # padding
        cols = tl.arange(0, row)
        k = tl.load(k_ptr + token * k_token_stride + cols)
        v = tl.load(v_ptr + token * v_token_stride + cols)
        dst = (slot // block_size) * block_stride + (slot % block_size) * slot_stride + cols
        tl.store(k_cache_ptr + dst, k)
        tl.store(v_cache_ptr + dst, v)
    
def _store_kv_triton(k, v, k_cache, v_cache, slot_mapping) -> None:
    num_tokens, num_kv_heads, head_dim = k.shape
    row = num_kv_heads * head_dim
    # A token's heads must sit back to back, in k/v and in the cache, so one row of values covers them.
    assert k.stride(-1) == v.stride(-1) == 1 and k.stride(1) == v.stride(1) == head_dim
    assert k_cache.stride() == v_cache.stride()
    assert k_cache.stride(3) == 1 and k_cache.stride(2) == head_dim and k_cache.stride(1) == row
    assert row & (row - 1) == 0, "tl.arange needs a power-of-two length"
    assert slot_mapping.dtype == torch.int64, "int32 offsets overflow on a large pool"
    _store_kv_kernel[(num_tokens,)](
        k, v, k_cache, v_cache, slot_mapping,
        k.stride(0), v.stride(0), k_cache.stride(0), k_cache.stride(1),
        block_size=k_cache.shape[1], row=row,
    )
    
def store_kv(
        k: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor, # [num_tokens, num_kv_heads, head_dim]
        k_cache: torch.Tensor, # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: torch.Tensor, # [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: torch.Tensor, # [num_tokens], int64
) -> None:
    """
    Write new K/V vectors into their paged cache slots, in place.

    ``slot_mapping[i]`` gives the flat slot index for token ``i``:

        slot = physical_block_id * block_size + offset_within_block

    The caller computes it once per step and reuses it for every layer: it depends
    only on block tables and token positions, not on layer contents.

    Padded rows carry slot ``-1`` and are skipped. On CUDA this is a per-thread early
    return in the Triton store kernel; on CPU the fallback below masks them out, where
    a data-dependent shape costs nothing.
    Both caches are modified in place; nothing is returned.
    """
    if k.is_cuda:
        return _store_kv_triton(k, v, k_cache, v_cache, slot_mapping)
    # Padded rows carry slot -1 and must not be written. Masking has a data-dependent
    # output shape, which on CUDA would force a host sync -- that is why the CUDA path
    # is a Triton kernel with a per-thread early return instead. On CPU there is no
    # device to synchronise with, so masking here is free.
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    block_size = k_cache.shape[1]
    blocks, offsets = slots // block_size, slots % block_size

    # Index block and offset separately instead of flattening: the cache is a strided layer view of the
    # block-first pool, which view(-1, ...) cannot flatten. Indexed assignment writes through the view.
    k_cache[blocks, offsets] = k[valid]
    v_cache[blocks, offsets] = v[valid]


def check_paged_layout(k_cache: torch.Tensor, block_tables: torch.Tensor) -> None:
    """flash-attn's paged-KV requirements, checked here so a violation reads as a sentence, not a C++ assert."""
    block_size = k_cache.shape[1]
    if block_size % 256:
        raise ValueError(f"flash-attn paged KV needs block_size % 256 == 0, got {block_size}")
    if block_tables.dtype != torch.int32:
        raise TypeError(f"block_tables must be int32, got {block_tables.dtype}")
    if block_tables.stride(-1) != 1:
        raise ValueError("block_tables must be contiguous in its last dimension")
    if k_cache.stride(-1) != 1:
        raise ValueError("the KV cache must be contiguous in head_dim")


def paged_attention_flash(
        q: torch.Tensor,              # [num_tokens, num_heads, head_dim], sequences packed back to back
        k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim], may be a strided view
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,   # [batch, max_blocks_per_seq] int32
        cu_seqlens_q: torch.Tensor,   # [batch + 1] int32, query offsets into q
        cu_seqlens_k: torch.Tensor,   # [batch + 1] int32, running total of keys per sequence
        max_seqlen_q: int,
        max_seqlen_k: int,
        scale: float | None = None,
) -> torch.Tensor:                    # [num_tokens, num_heads, head_dim]
    """
    Every query attends over its sequence's keys in the paged cache, prefill and decode alike, in one call.

    This step's K/V must already be in the cache (store_kv runs first). flash-attn aligns the causal
    mask to the bottom right, so a sequence's queries are its last tokens: query j of a sequence with
    q_len queries and k_len keys sees keys up to k_len - q_len + j. That one rule covers a full
    prefill, a prefill over a cached prefix, and a decode.
    """
    from flash_attn import flash_attn_varlen_func

    check_paged_layout(k_cache, block_tables)
    return flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=scale, causal=True, block_table=block_tables,
    )
