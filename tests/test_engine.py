"""
The engine end to end on CPU, on tiny random Llama checkpoints: greedy output must match
transformers token for token, and nothing the cache does (prefix hits, preemption, forks, caching
off) may change what is generated.
"""

from __future__ import annotations

import pytest
import torch

from tests.test_llama import hf_model, save_checkpoint, tiny_config
from tinyserve_llm.attention.torch_paged import paged_attention_torch
from tinyserve_llm.engine.interface import CacheConfig, ModelShape, build
from tinyserve_llm.engine.llm_engine import LLM, warmup_prompt_lengths
from tinyserve_llm.engine.model_runner import prepare
from tinyserve_llm.engine.sampler import sample
from tinyserve_llm.model import llama
from tinyserve_llm.model.attention import DenseCausalAttention
from tinyserve_llm.scheduler.scheduler import SchedulerConfig
from tinyserve_llm.scheduler.sequence import SamplingParams

B = 4  # small blocks so short prompts span several, with partial tails
GREEDY = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True)


@pytest.fixture(autouse=True)
def eager_kernels(monkeypatch):
    monkeypatch.setattr(llama, "COMPILE", False)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_llama")
    hf = hf_model(tiny_config())
    save_checkpoint(hf, path)
    return path, hf


def engine(path, num_blocks: int = 64, caching: bool = True, **overrides) -> LLM:
    config = SchedulerConfig(**{**dict(max_model_len=64, max_num_seqs=8, max_num_batched_tokens=72), **overrides})
    return LLM(str(path), config, CacheConfig(block_size=B, caching=caching, device="cpu"),
               dtype=torch.float32, num_blocks=num_blocks, load_tokenizer=False)


def hf_greedy(hf, prompt: list[int], max_tokens: int) -> list[int]:
    out = hf.generate(torch.tensor([prompt]), max_new_tokens=max_tokens, min_new_tokens=max_tokens,
                      do_sample=False, pad_token_id=0)
    return out[0, len(prompt):].tolist()


PREFIX = [5, 17, 3, 99, 42, 8, 61, 23, 7]  # 9 tokens: two full blocks and a partial one
PROMPTS = [PREFIX + [11, 12], PREFIX + [30], [77, 2, 9], PREFIX + [11, 12, 13, 14, 15], [64] * 6]


# --- attention ---------------------------------------------------------------------------------


def scattered_cache(k, v, table):
    """Write k/v for positions 0.. into the blocks named by table, in a pool with other junk in it."""
    k_cache = torch.randn(8, B, k.shape[1], k.shape[2])
    v_cache = torch.randn_like(k_cache)
    for p in range(k.shape[0]):
        k_cache[table[p // B], p % B] = k[p]
        v_cache[table[p // B], p % B] = v[p]
    return k_cache, v_cache


def test_paged_attention_matches_dense_for_a_whole_prefill_a_cached_prefix_and_a_decode():
    torch.manual_seed(0)
    n, heads, kv_heads, dim = 11, 4, 2, 8
    q, k, v = torch.randn(n, heads, dim), torch.randn(n, kv_heads, dim), torch.randn(n, kv_heads, dim)
    dense = DenseCausalAttention([0, n])(0, q, k, v)
    table = [6, 1, 4]  # out of order on purpose
    k_cache, v_cache = scattered_cache(k, v, table)
    tables = torch.tensor([table + [0], table + [0], table + [0]], dtype=torch.int32)
    # One batch: the whole prompt; its last 4 tokens over a cached prefix; its last token as a decode.
    queries = torch.cat([q, q[n - 4:], q[n - 1:]])
    out = paged_attention_torch(queries, k_cache, v_cache, tables, [0, n, n + 4, n + 5], [n, n, n])
    torch.testing.assert_close(out[:n], dense)
    torch.testing.assert_close(out[n: n + 4], dense[n - 4:])
    torch.testing.assert_close(out[n + 4:], dense[n - 1:])


def test_paged_attention_reads_a_strided_layer_view():
    torch.manual_seed(1)
    n, heads, kv_heads, dim = 6, 2, 1, 4
    q, k, v = torch.randn(n, heads, dim), torch.randn(n, kv_heads, dim), torch.randn(n, kv_heads, dim)
    pool_k = torch.randn(4, 3, B, kv_heads, dim)  # [blocks, layers, block_size, heads, dim]
    pool_v = torch.randn_like(pool_k)
    # Layer 1 holds the sequence on blocks [2, 0]; everything else is junk.
    pool_k[2, 1, :B], pool_k[0, 1, :2] = k[:B], k[B:]
    pool_v[2, 1, :B], pool_v[0, 1, :2] = v[:B], v[B:]
    out = paged_attention_torch(q, pool_k[:, 1], pool_v[:, 1], torch.tensor([[2, 0]], dtype=torch.int32), [0, n], [n])
    torch.testing.assert_close(out, DenseCausalAttention([0, n])(0, q, k, v))


# --- packing -----------------------------------------------------------------------------------


def test_prepare_packs_a_mixed_step():
    shape = ModelShape(num_layers=1, num_kv_heads=1, head_dim=2, dtype=torch.float32)
    config = SchedulerConfig(max_model_len=32, max_num_seqs=4, max_num_batched_tokens=36, eos_token_id=-1)
    scheduler, view = build(16, shape, config, CacheConfig(block_size=B, device="cpu"))
    first = scheduler.add_request(list(range(2, 8)), SamplingParams(max_tokens=4), 0.0)
    scheduler.postprocess(scheduler.schedule(0.0), [[50]], 0.0)  # 6 prompt tokens committed, 50 pending
    scheduler.add_request(list(range(2, 7)) + [9], SamplingParams(max_tokens=4, n=2, temperature=0.5), 0.0)
    batch = scheduler.schedule(0.1)  # first decodes; the second request hits the first's full block
    inputs = prepare(batch, view, "cpu")
    t0, t1 = view.block_table(batch[0].seq.seq_id), view.block_table(batch[1].seq.seq_id)
    assert batch[0].seq.request_id == first and t1[0] == t0[0], "the full prompt block is shared"
    assert inputs.input_ids.tolist() == [50, 6, 9]  # the pending token, then the uncached tail
    assert inputs.positions.tolist() == [6, 4, 5]
    assert inputs.slot_mapping.tolist() == [t0[1] * B + 2, t1[1] * B + 0, t1[1] * B + 1]
    assert inputs.cu_seqlens_q == [0, 1, 3] and inputs.context_lens == [7, 6]
    assert inputs.block_tables.tolist() == [list(t0), list(t1)] and inputs.block_tables.dtype == torch.int32
    assert inputs.sample_rows.tolist() == [0, 2, 2]  # the group draws twice from its last row
    assert inputs.temperatures.tolist() == [1.0, 0.5, 0.5]


def test_sampler():
    logits = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]])
    assert sample(logits, torch.zeros(2)).tolist() == [1, 0]
    g = torch.Generator().manual_seed(0)
    draws = sample(logits[:1].expand(20000, 3), torch.ones(20000), g)
    freq = torch.bincount(draws, minlength=3).float() / 20000
    torch.testing.assert_close(freq, torch.softmax(logits[0], -1), atol=0.01, rtol=0)
    a = sample(logits, torch.ones(2), torch.Generator().manual_seed(7))
    assert torch.equal(a, sample(logits, torch.ones(2), torch.Generator().manual_seed(7)))


def test_warmup_fills_the_budget_and_the_sequence_cap():
    for budget, seqs, max_len in ((16384, 512, 4096), (72, 8, 64), (36, 4, 32), (600, 1, 600)):
        lengths = warmup_prompt_lengths(budget, seqs, max_len)
        assert len(lengths) == seqs and all(1 <= n <= max_len - 1 for n in lengths)
        assert sum(lengths) == min(budget, seqs * (max_len - 1))


# --- end to end --------------------------------------------------------------------------------


def test_greedy_matches_transformers(checkpoint):
    path, hf = checkpoint
    llm = engine(path)
    results = llm.generate(PROMPTS, GREEDY)
    for prompt, result in zip(PROMPTS, results):
        assert result.token_ids == [hf_greedy(hf, prompt, GREEDY.max_tokens)]
        assert result.finish_reasons == ["length"]
    # All five were admitted in one step, before any block was committed, so nothing hit yet.
    assert llm.scheduler.stats().cached_tokens == 0
    later = PREFIX + [11, 40]
    (result,) = llm.generate([later], GREEDY)
    assert result.token_ids == [hf_greedy(hf, later, GREEDY.max_tokens)]
    assert llm.scheduler.stats().cached_tokens == len(PREFIX) + 1, "the released prefix pages were reused"


@pytest.mark.parametrize("caching", [True, False])
def test_preemption_and_caching_do_not_change_output(checkpoint, caching):
    path, hf = checkpoint
    expected = engine(path).generate(PROMPTS, GREEDY)
    llm = engine(path, num_blocks=14, caching=caching)  # too small to run all five at once
    results = llm.generate(PROMPTS, GREEDY)
    assert [r.token_ids for r in results] == [r.token_ids for r in expected]
    stats = llm.scheduler.stats()
    assert stats.preemptions > 0
    assert (stats.cached_tokens > 0) == caching


def test_parallel_samples_fork_and_greedy_samples_agree(checkpoint):
    path, hf = checkpoint
    llm = engine(path)
    (result,) = llm.generate([PROMPTS[0]], SamplingParams(max_tokens=6, temperature=0.0, n=3, ignore_eos=True))
    assert result.token_ids == [hf_greedy(hf, PROMPTS[0], 6)] * 3
    assert llm.view.stats().blocks_in_use == 0


def test_seeded_sampling_is_reproducible(checkpoint):
    path, _ = checkpoint
    params = SamplingParams(max_tokens=10, temperature=1.0, ignore_eos=True)
    runs = [engine(path) for _ in range(2)]
    for llm in runs:
        llm.generator = llm.runner.generator = torch.Generator().manual_seed(3)
    a, b = (llm.generate(PROMPTS, params) for llm in runs)
    assert [r.token_ids for r in a] == [r.token_ids for r in b]


def test_the_end_token_stops_generation(checkpoint):
    path, hf = checkpoint
    greedy = hf_greedy(hf, PROMPTS[0], GREEDY.max_tokens)
    eos = greedy[2]
    llm = engine(path, eos_token_id=eos)
    (result,) = llm.generate(PROMPTS[:1], SamplingParams(max_tokens=12, temperature=0.0))
    assert result.token_ids == [greedy[: greedy.index(eos) + 1]] and result.finish_reasons == ["eos"]


def test_step_records_and_timing(checkpoint):
    path, _ = checkpoint
    llm = engine(path)
    (result,) = llm.generate([PROMPTS[0]], SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True))
    assert len(llm.steps) == 3 and [s.num_tokens for s in llm.steps] == [len(PROMPTS[0]), 1, 1]
    assert all(s.start <= s.end for s in llm.steps)
    assert llm.steps[-1].scheduler.decode_tokens == 2 and llm.steps[-1].cache.publishes > 0
    assert result.arrival_time <= result.first_token_time <= result.finish_time


def test_num_blocks_is_required_off_the_gpu(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="num_blocks"):
        LLM(str(path), SchedulerConfig(max_model_len=64, max_num_seqs=8, max_num_batched_tokens=72),
            CacheConfig(block_size=B, device="cpu"), dtype=torch.float32, load_tokenizer=False)
