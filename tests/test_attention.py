from __future__ import annotations

import torch

from tinyserve_llm.attention.flash_backend import store_kv

# Deliberately tiny and not a multiple of 256: these exercise slot arithmetic, not
# flash-attn's constraints. The 256 rule is enforced by the flash wrappers, not here.
NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM = 6, 4, 2, 8


def make_pool():
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return torch.zeros(shape), torch.zeros(shape)


def flat(cache):
    return cache.view(-1, NUM_KV_HEADS, HEAD_DIM)


def test_store_kv_writes_to_the_named_slots_and_nowhere_else():
    """A sequence on blocks [3, 1]: slots jump backwards, which is the point."""
    k_cache, v_cache = make_pool()
    slot_mapping = torch.tensor([12, 13, 14, 15, 4, 5, 6])
    k = torch.randn(7, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(7, NUM_KV_HEADS, HEAD_DIM)

    store_kv(k, v, k_cache, v_cache, slot_mapping)

    torch.testing.assert_close(flat(k_cache)[slot_mapping], k)
    torch.testing.assert_close(flat(v_cache)[slot_mapping], v)

    untouched = torch.ones(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.bool)
    untouched[slot_mapping] = False
    assert flat(k_cache)[untouched].eq(0).all()
    assert flat(v_cache)[untouched].eq(0).all()


def test_store_kv_mutates_the_callers_pool():
    """Assert through the original 4-D tensor, not a view we made ourselves.

    A reshape() that silently copied would satisfy any assertion made through the
    copy. Indexing k_cache[block, offset] can only pass if the caller's tensor changed.
    """
    k_cache, v_cache = make_pool()
    k = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)

    store_kv(k, k, k_cache, v_cache, torch.tensor([13]))

    torch.testing.assert_close(k_cache[3, 1], k[0])   # slot 13 == block 3, offset 1


def test_store_kv_skips_padded_rows():
    """-1 means 'no token here'. It must write nothing -- in particular it must not
    wrap around to the last slot of the pool, which PyTorch indexing would do."""
    k_cache, v_cache = make_pool()
    k = torch.randn(3, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(3, NUM_KV_HEADS, HEAD_DIM)

    store_kv(k, v, k_cache, v_cache, torch.tensor([12, -1, 5]))

    torch.testing.assert_close(flat(k_cache)[12], k[0])
    torch.testing.assert_close(flat(k_cache)[5], k[2])
    assert flat(k_cache)[-1].eq(0).all()


def test_store_kv_pairs_rows_with_slots_positionally():
    """Permuting tokens and their slots together must produce an identical pool."""
    k_cache_a, v_cache_a = make_pool()
    k_cache_b, v_cache_b = make_pool()
    slot_mapping = torch.tensor([12, 13, 14, 15, 4, 5, 6])
    k = torch.randn(7, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(7, NUM_KV_HEADS, HEAD_DIM)

    store_kv(k, v, k_cache_a, v_cache_a, slot_mapping)

    perm = torch.randperm(7)
    store_kv(k[perm], v[perm], k_cache_b, v_cache_b, slot_mapping[perm])

    torch.testing.assert_close(k_cache_a, k_cache_b)
