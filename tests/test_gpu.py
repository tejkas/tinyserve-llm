"""
GPU-only parity tests: the CUDA paths must compute what the CPU paths compute. Skipped without
CUDA and flash-attn; run them through scripts/gpu_tests.sbatch.

bf16 and fp16 round differently from the fp32 references, so comparisons use loose tolerances.
Greedy token parity is deliberately not checked here: a tiny random model has near-tied logits,
and one rounding difference flips an argmax without anything being wrong.
"""

from __future__ import annotations

import pytest
import torch

flash_attn = pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tests.test_llama import hf_model, save_checkpoint, tiny_config  # noqa: E402
from tinyserve_llm.attention.flash_backend import paged_attention_flash, store_kv  # noqa: E402
from tinyserve_llm.attention.torch_paged import paged_attention_torch  # noqa: E402
from tinyserve_llm.engine.interface import CacheConfig, ModelShape, build  # noqa: E402
from tinyserve_llm.engine.llm_engine import LLM  # noqa: E402
from tinyserve_llm.engine.model_runner import PagedAttention, prepare  # noqa: E402
from tinyserve_llm.kvcache.storage import KVStorage  # noqa: E402
from tinyserve_llm.model.attention import DenseCausalAttention  # noqa: E402
from tinyserve_llm.model.loader import load_llama  # noqa: E402
from tinyserve_llm.scheduler.scheduler import SchedulerConfig  # noqa: E402
from tinyserve_llm.scheduler.sequence import SamplingParams  # noqa: E402

CUDA = torch.device("cuda")
B = 256
TOL = {torch.bfloat16: dict(atol=2e-2, rtol=2e-2), torch.float16: dict(atol=2e-3, rtol=2e-3)}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_store_kv_on_cuda_matches_cpu_through_a_strided_layer_view(dtype):
    torch.manual_seed(0)
    num_blocks, layers, kv_heads, head_dim = 6, 3, 8, 64
    pools = {d: KVStorage(num_blocks, layers, kv_heads, head_dim, B, dtype, d) for d in ("cpu", "cuda")}
    n = 700
    k, v = torch.randn(n, kv_heads, head_dim, dtype=dtype), torch.randn(n, kv_heads, head_dim, dtype=dtype)
    slots = torch.randperm(num_blocks * B)[:n]
    slots[::7] = -1  # padding rows write nothing
    for device, storage in pools.items():
        k_cache, v_cache = storage.layer(1)
        store_kv(k.to(device), v.to(device), k_cache, v_cache, slots.to(device))
    assert torch.equal(pools["cuda"].k_pool.cpu(), pools["cpu"].k_pool)
    assert torch.equal(pools["cuda"].v_pool.cpu(), pools["cpu"].v_pool)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_matches_torch_paged_on_a_mixed_batch(dtype):
    """One batch with a full prefill, a prefill over a cached prefix and two decodes, over a strided layer view."""
    torch.manual_seed(1)
    heads, kv_heads, head_dim = 16, 4, 64
    storage = KVStorage(16, 2, kv_heads, head_dim, B, dtype, "cuda")
    k_cache, v_cache = storage.layer(1)
    k_cache.normal_()
    v_cache.normal_()
    tables = [[3, 9], [12, 0, 5], [7], [1, 14, 2, 11]]
    context_lens = [300, 700, 90, 1000]
    query_lens = [300, 180, 1, 1]
    cu_q = [0]
    for n in query_lens:
        cu_q.append(cu_q[-1] + n)
    cu_k = [0]
    for n in context_lens:
        cu_k.append(cu_k[-1] + n)
    width = max(map(len, tables))
    block_tables = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32, device=CUDA)
    q = torch.randn(cu_q[-1], heads, head_dim, dtype=dtype, device=CUDA)

    expected = paged_attention_torch(q.float(), k_cache.float(), v_cache.float(), block_tables, cu_q, context_lens)
    got = paged_attention_flash(q, k_cache, v_cache, block_tables,
                                torch.tensor(cu_q, dtype=torch.int32, device=CUDA),
                                torch.tensor(cu_k, dtype=torch.int32, device=CUDA),
                                max(query_lens), max(context_lens))
    torch.testing.assert_close(got.float(), expected, **TOL[dtype])


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_llama_gpu")
    save_checkpoint(hf_model(tiny_config()), path)
    return path


def test_the_runner_with_flash_matches_dense_attention_through_prefill_and_decode(checkpoint):
    """Real scheduler, cache and packing; only the attention backend differs from the dense oracle."""
    dtype = torch.bfloat16
    model = load_llama(str(checkpoint), dtype=dtype, device=CUDA)
    shape = ModelShape.from_hf_config(model.config, dtype)
    config = SchedulerConfig(max_model_len=128, max_num_seqs=4, max_num_batched_tokens=132)
    scheduler, view = build(8, shape, config, CacheConfig(block_size=B, device="cuda"))
    prompts = [list(range(3, 40)), list(range(50, 61)), [7] * 70]
    for p in prompts:
        scheduler.add_request(p, SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True), 0.0)

    for step in range(2):  # the prefill step, then a decode step
        batch = scheduler.schedule(float(step))
        inputs = prepare(batch, view, CUDA)
        with torch.inference_mode():
            hidden = model(inputs.input_ids, inputs.positions, PagedAttention(view, inputs, "flash"))
            for i, (seq, fed, _) in enumerate(batch):
                full = torch.tensor(seq.all_token_ids, device=CUDA)
                dense = model(full, torch.arange(len(full), device=CUDA), DenseCausalAttention([0, len(full)]))
                rows = slice(inputs.cu_seqlens_q[i], inputs.cu_seqlens_q[i + 1])
                torch.testing.assert_close(hidden[rows].float(), dense[-len(fed):].float(), atol=5e-2, rtol=5e-2)
        scheduler.postprocess(batch, [[5] for _ in batch], float(step))


def test_the_engine_runs_on_cuda_sized_from_free_memory(checkpoint):
    llm = LLM(str(checkpoint), SchedulerConfig(max_model_len=128, max_num_seqs=8, max_num_batched_tokens=512),
              CacheConfig(block_size=B, device="cuda"), dtype=torch.bfloat16, load_tokenizer=False,
              gpu_memory_utilization=0.5)
    assert llm.scheduler.kv.num_blocks > 0
    prompts = [list(range(3, 3 + n)) for n in (5, 60, 120, 33)]
    results = llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.8, ignore_eos=True))
    assert [len(r.token_ids[0]) for r in results] == [8] * 4
    assert llm.view.stats().blocks_in_use == 0
