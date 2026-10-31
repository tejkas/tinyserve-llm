from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from tinyserve_llm.attention.flash_backend import store_kv
from tinyserve_llm.engine.interface import (
    CacheConfig, KVCacheView, ModelShape, PagedKVCacheView, build, bytes_per_block,
)
from tinyserve_llm.scheduler.scheduler import Scheduler, SchedulerConfig
from tinyserve_llm.scheduler.sequence import SamplingParams

SHAPE = ModelShape(num_layers=3, num_kv_heads=2, head_dim=8, dtype=torch.float32)
B = 4


def make(num_blocks: int = 16, caching: bool = True) -> tuple[Scheduler, KVCacheView]:
    config = SchedulerConfig(max_model_len=32, max_num_seqs=4, max_num_batched_tokens=36)
    return build(num_blocks, SHAPE, config, CacheConfig(block_size=B, caching=caching, device="cpu"))


def test_build_wires_one_manager_behind_the_scheduler_and_the_view():
    scheduler, view = make(caching=False)
    assert isinstance(view, PagedKVCacheView) and view.block_size == B
    assert scheduler.kv.num_blocks == 16 and scheduler.kv.caching is False
    k, v = view.layer_cache(2)
    assert k.shape == v.shape == (16, B, SHAPE.num_kv_heads, SHAPE.head_dim)
    assert k.dtype == torch.float32 and k.device.type == "cpu"


def test_bytes_per_block_matches_the_allocated_pools():
    scheduler, _ = make(num_blocks=5)
    storage = scheduler.kv.storage
    allocated = sum(p.numel() * p.element_size() for p in (storage.k_pool, storage.v_pool))
    assert bytes_per_block(SHAPE, B) * 5 == allocated


def test_the_view_reads_what_the_scheduler_reserved_and_writes_land_in_the_pool():
    scheduler, view = make()
    scheduler.add_request([2, 3, 4, 5, 6], SamplingParams(max_tokens=2), 0.0)
    ((seq, fed, _),) = scheduler.schedule(0.0)
    table = view.block_table(seq.seq_id)
    assert list(table) == scheduler.kv.seqs[seq.seq_id].block_table and len(table) == 2
    # The engine's slot mapping for this prefill, then one layer's write through the view
    slots = torch.tensor([table[p // B] * B + p % B for p in range(len(fed))])
    k = torch.randn(len(fed), SHAPE.num_kv_heads, SHAPE.head_dim)
    k_cache, v_cache = view.layer_cache(1)
    store_kv(k, -k, k_cache, v_cache, slots)
    pool = scheduler.kv.storage.k_pool
    assert torch.equal(pool[table[0], 1], k[:B]) and torch.equal(pool[table[1], 1, :1], k[B:])
    assert not pool[:, 0].any() and not pool[:, 2].any(), "other layers are untouched"


def test_a_block_table_can_change_in_place_at_commit():
    """Two identical prompts admitted in one step both compute their first block; the second commit
    dedupes it onto the first's page. A table kept from before postprocess would point at a freed block."""
    scheduler, view = make()
    prompt = [2, 3, 4, 5, 6]
    for _ in range(2):
        scheduler.add_request(prompt, SamplingParams(max_tokens=3), 0.0)
    batch = scheduler.schedule(0.0)
    assert [len(fed) for _, fed, _ in batch] == [5, 5], "nothing was cached yet"
    first, second = (list(view.block_table(q.seq.seq_id)) for q in batch)
    assert first[0] != second[0]
    scheduler.postprocess(batch, [[7], [7]], 0.0)
    assert view.block_table(batch[1].seq.seq_id)[0] == first[0] != second[0]
    assert view.stats().dedupes == 1


def test_stats_pass_through():
    scheduler, view = make()
    scheduler.add_request([2] * 6, SamplingParams(max_tokens=1), 0.0)
    batch = scheduler.schedule(0.0)
    assert view.stats().blocks_in_use == 2
    scheduler.postprocess(batch, [[9]], 0.0)
    assert view.stats() == scheduler.kv.stats()


def test_a_view_missing_a_method_cannot_be_created():
    class Partial(KVCacheView):
        def block_table(self, seq_id):
            return []

    with pytest.raises(TypeError):
        Partial()


@pytest.mark.parametrize("head_dim", [None, 64])
def test_model_shape_from_hf_config(head_dim):
    config = LlamaConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=2, num_hidden_layers=5,
                         head_dim=head_dim)
    shape = ModelShape.from_hf_config(config, dtype=torch.float16)
    assert shape == ModelShape(5, 2, head_dim or 256 // 8, torch.float16)
