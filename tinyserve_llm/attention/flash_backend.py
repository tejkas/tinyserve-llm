"""
flash-attn backend: paged KV writes and attention over block tables.

tinyserve_llm delegates attention math to flash-attn. This module is the
adapter between our block tables and flash-attn's paged KV interface:

    store_kv           write new K/V into their paged slots
    prefill_attention  flash_attn_varlen_func
    decode_attention   flash_attn_with_kvcache

flash-attn is CUDA-only, so it is imported lazily inside the functions that need it.
"""
from __future__ import annotations

import torch

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
    a data-dependent shape costs nothing. There is no sentinel block.

    Both caches are modified in place; nothing is returned.
    """
    num_kv_heads, head_dim = k.shape[1], k.shape[2]

    # Padded rows carry slot -1 and must not be written. Masking has a data-dependent
    # output shape, which on CUDA would force a host sync -- that is why the CUDA path
    # is a Triton kernel with a per-thread early return instead. On CPU there is no
    # device to synchronise with, so masking here is free.
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]

    # view(), not reshape(): the pool is contiguous, and view() raises if that ever
    # stops being true. reshape() would silently write into a copy and drop the data.
    k_cache.view(-1, num_kv_heads, head_dim)[slots] = k[valid]
    v_cache.view(-1, num_kv_heads, head_dim)[slots] = v[valid]
