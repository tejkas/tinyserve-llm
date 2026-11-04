from __future__ import annotations

import random

import pytest

from tests.test_kvcache_manager import B, FakeKV, check_invariants, forward
from tinyserve_llm.kvcache.manager import KVCacheManager
from tinyserve_llm.scheduler.scheduler import RequestOutput, ScheduledSeq, Scheduler, SchedulerConfig, SchedulerStats
from tinyserve_llm.scheduler.sequence import SamplingParams, Sequence, SequenceStatus

VOCAB = 8
EOS = 1


def model(history: list[int], sample: int) -> int:
    """Fake model: the next token is a fixed function of everything before it (and which sample is drawn)."""
    return hash((tuple(history), sample)) % VOCAB + 2  # never EOS unless a test asks for it


def config(**overrides) -> SchedulerConfig:
    base = dict(max_model_len=64, max_num_seqs=8, max_num_batched_tokens=72, eos_token_id=EOS, watermark=0.0)
    return SchedulerConfig(**{**base, **overrides})


def make(num_blocks: int = 32, caching: bool = True, **overrides) -> Scheduler:
    return Scheduler(config(**overrides), KVCacheManager(FakeKV(num_blocks), caching=caching))


def check_scheduler(s: Scheduler) -> None:
    """What the scheduler promises between steps."""
    m = s.kv
    check_invariants(m)
    assert {q.seq_id for q in s.running}.isdisjoint(q.seq_id for q in s.waiting)
    assert set(m.seqs) == {q.seq_id for q in s.running}, "exactly the running sequences hold KV"
    assert len(s.running) <= s.config.max_num_seqs
    for q in s.running:
        assert q.status is SequenceStatus.RUNNING
        # Every token but the pending one has KV
        assert m.seqs[q.seq_id].token_ids == q.all_token_ids[:-1]
    for q in s.waiting:
        assert q.status is SequenceStatus.WAITING
    for group in s.groups.values():
        assert len(group) <= group[0].params.n


def check_step_stats(s: Scheduler, before: SchedulerStats, after: SchedulerStats, was_running: list[Sequence],
                     preemptions_before: int, batch: list[ScheduledSeq]) -> None:
    """The counters moved by exactly what this step's batch shows."""
    running_ids = {q.seq_id for q in was_running}
    decodes = [q for q in batch if q.seq.seq_id in running_ids]
    admitted = [q for q in batch if q.seq.seq_id not in running_ids]
    delta = {f: getattr(after, f) - getattr(before, f) for f in
             ("admissions", "readmissions", "preemptions", "cached_tokens", "prefill_tokens", "decode_tokens")}
    assert delta == {
        "admissions": sum(not q.seq.num_preemptions for q in admitted),
        "readmissions": sum(bool(q.seq.num_preemptions) for q in admitted),
        "preemptions": sum(q.num_preemptions for q in was_running) - preemptions_before,
        "cached_tokens": sum(q.seq.num_tokens - len(q.token_ids) for q in admitted),
        "prefill_tokens": sum(len(q.token_ids) for q in admitted),
        "decode_tokens": len(decodes),
    }
    assert (after.num_waiting, after.num_running) == (len(s.waiting), len(s.running))


def run_step(s: Scheduler, now: float, token_for=model) -> tuple[list[ScheduledSeq], list[RequestOutput]]:
    """Fake engine step. The model reads its history from the KV slots, so a wrong feed changes the output."""
    m = s.kv
    before, was_running = s.stats(), list(s.running)
    preemptions_before = sum(q.num_preemptions for q in was_running)
    batch = s.schedule(now)
    check_step_stats(s, before, s.stats(), was_running, preemptions_before, batch)
    assert sum(len(fed) for _, fed, _ in batch) <= s.config.max_num_batched_tokens
    assert len({q.seq_id for q, _, _ in batch}) == len(batch)
    sampled = []
    for seq, fed, num_samples in batch:
        st = m.seqs[seq.seq_id]
        assert st.committed == seq.num_tokens - len(fed), "positions start where the KV ends"
        assert st.token_ids[st.committed:] == fed
        forward(m, seq.seq_id)
        history = [m.storage.slots[st.block_table[p // B]][p % B] for p in range(st.reserved)]
        sampled.append([token_for(history, k) for k in range(num_samples)])
    outputs = s.postprocess(batch, sampled, now)
    check_scheduler(s)
    return batch, outputs


def reference(prompt: list[int], params: SamplingParams, sample: int) -> list[int]:
    """What one sample generates with no batching and no preemption."""
    out = [model(prompt, sample)]
    while not ((out[-1] == EOS and not params.ignore_eos) or len(out) == params.max_tokens):
        out.append(model(prompt + out, 0))
    return out


def run_to_completion(s: Scheduler, max_steps: int = 2000) -> dict[int, RequestOutput]:
    done: dict[int, RequestOutput] = {}
    now = 0.0
    for _ in range(max_steps):
        if not s.has_unfinished():
            return done
        now += 0.1
        _, outputs = run_step(s, now)
        for o in outputs:
            assert o.request_id not in done, "a request is returned once"
            done[o.request_id] = o
    raise AssertionError("the scheduler stopped making progress")


# --- config and add_request --------------------------------------------------------------------


def test_budget_must_fit_a_maximum_prompt_beside_a_full_decode_batch():
    with pytest.raises(AssertionError):
        SchedulerConfig(max_model_len=64, max_num_seqs=8, max_num_batched_tokens=71)
    SchedulerConfig(max_model_len=64, max_num_seqs=8, max_num_batched_tokens=72)


def test_add_request_rejects_what_could_never_run():
    s = make(num_blocks=4)  # 16 slots
    with pytest.raises(ValueError, match="max_model_len"):
        s.add_request([2] * 60, SamplingParams(max_tokens=5), 0.0)
    with pytest.raises(ValueError, match="blocks"):
        s.add_request([2] * 10, SamplingParams(max_tokens=10), 0.0)  # 20 tokens > 16 slots
    with pytest.raises(ValueError, match="blocks"):
        s.add_request([2] * 8, SamplingParams(max_tokens=4, n=3), 0.0)  # 3 blocks + 2 fork copies
    with pytest.raises(ValueError, match="max_num_seqs"):
        make().add_request([2], SamplingParams(max_tokens=1, n=9), 0.0)  # 9 samples never fit a cap of 8
    assert not s.waiting



def test_a_request_accepted_at_the_bound_can_always_be_admitted():
    """
    A partial copy out of a free cached page needs the source and a fresh block at once, so a 2-block
    request can cost 3. add_request must count that, or a request accepted into a 2-block pool could
    never be admitted.
    """
    with pytest.raises(ValueError, match="blocks"):
        make(num_blocks=2).add_request([5, 6, 7, 8, 9, 12], SamplingParams(max_tokens=1), 0.0)

    s = make(num_blocks=3)
    s.add_request([5, 6, 7, 8, 9, 10, 11], SamplingParams(max_tokens=1, ignore_eos=True), 0.0)
    run_to_completion(s)  # leaves a full page [5, 6, 7, 8] and a free partial page [9, 10, 11]
    s.add_request([5, 6, 7, 8, 9, 12], SamplingParams(max_tokens=1, ignore_eos=True), 1.0)
    batch, _ = run_step(s, 1.0)
    assert len(batch) == 1 and batch[0].token_ids == [12]  # 5 tokens cached, one of them copied
    assert s.kv.stats().partial_copies == 1

# --- one request -------------------------------------------------------------------------------


def test_prefill_feeds_the_prompt_then_decode_feeds_the_pending_token():
    s = make()
    s.add_request([2, 3, 4, 5, 6], SamplingParams(max_tokens=3), 0.0)
    (prefill,), _ = run_step(s, 0.0)
    assert prefill.token_ids == [2, 3, 4, 5, 6] and prefill.num_samples == 1
    seq = prefill.seq
    (decode,), _ = run_step(s, 0.1)
    assert decode.seq is seq and decode.token_ids == [seq.output_token_ids[0]]
    _, outputs = run_step(s, 0.2)
    (out,) = outputs
    assert out.sequences == [seq] and seq.finish_reason == "length"
    assert seq.output_token_ids == reference([2, 3, 4, 5, 6], seq.params, 0)
    assert (seq.first_token_time, seq.finish_time) == (0.0, 0.2)
    assert not s.has_unfinished() and not s.kv.seqs and not s.groups


def test_eos_stops_unless_ignored():
    for ignore_eos, expected, reason in ((False, [5, EOS], "eos"), (True, [5, EOS, 5, EOS], "length")):
        s = make()
        s.add_request([2, 3], SamplingParams(max_tokens=4, ignore_eos=ignore_eos), 0.0)
        tokens = iter([5, EOS, 5, EOS])
        outputs, now = [], 0.0
        while s.has_unfinished():
            now += 0.1
            outputs += run_step(s, now, token_for=lambda history, k: next(tokens))[1]
        (out,) = outputs
        assert out.sequences[0].output_token_ids == expected and out.sequences[0].finish_reason == reason


def test_a_cached_prefix_is_not_fed_again():
    s = make()
    prompt = list(range(2, 12))  # 10 tokens: two full blocks and a partial one
    s.add_request(prompt, SamplingParams(max_tokens=1), 0.0)
    run_to_completion(s)
    s.add_request(prompt + [7, 7], SamplingParams(max_tokens=2), 10.0)
    (prefill,), _ = run_step(s, 10.0)
    # release published the partial last page, so all 10 prompt tokens are cached
    assert prefill.token_ids == [7, 7]
    stats = s.stats()
    assert (stats.admissions, stats.cached_tokens, stats.prefill_tokens) == (2, 10, 10 + 2)


# --- admission ---------------------------------------------------------------------------------


def warm(s: Scheduler, prompt: list[int]) -> None:
    """Puts a prompt's pages in the prefix cache."""
    s.add_request(prompt, SamplingParams(max_tokens=1), 0.0)
    run_to_completion(s)


def test_higher_hit_is_admitted_first():
    s = make(max_num_seqs=1, max_num_batched_tokens=65, wait_weight=0.0)
    shared = list(range(2, 14))
    warm(s, shared)
    cold = s.add_request([9] * 12, SamplingParams(max_tokens=2), 0.0)
    hot = s.add_request(shared + [3], SamplingParams(max_tokens=2), 0.0)
    (first,), _ = run_step(s, 1.0)
    assert first.seq.request_id == hot


def test_waiting_time_outranks_hits_when_weighted_heavily():
    s = make(max_num_seqs=1, max_num_batched_tokens=65, wait_weight=1000.0)
    shared = list(range(2, 14))
    warm(s, shared)
    old = s.add_request([9] * 12, SamplingParams(max_tokens=2), 0.0)
    s.add_request(shared + [3], SamplingParams(max_tokens=2), 0.5)
    (first,), _ = run_step(s, 1.0)
    assert first.seq.request_id == old


def test_hit_modes():
    s = make(hit_mode="absolute")
    warm(s, list(range(2, 10)))  # 8 tokens cached
    seq = s.groups[s.add_request(list(range(2, 10)) + [3], SamplingParams(max_tokens=1), 0.0)][0]
    (_, cost, hit), = s._rank_waiting(0.0)
    assert hit == pytest.approx((9 - cost.prefill_tokens) / 64)
    s = make(hit_mode="ratio")
    warm(s, list(range(2, 10)))
    s.add_request(list(range(2, 10)) + [3], SamplingParams(max_tokens=1), 0.0)
    (_, cost, hit), = s._rank_waiting(0.0)
    assert hit == pytest.approx((9 - cost.prefill_tokens) / 9)


def blocker(s: Scheduler, num_tokens: int) -> None:
    """A running sequence that holds num_tokens worth of blocks."""
    s.add_request([5] * num_tokens, SamplingParams(max_tokens=8, ignore_eos=True), 0.0)
    run_step(s, 0.0)


def test_a_misfit_is_skipped_until_it_has_waited_max_skip_wait():
    for now, skip_expected in ((1.0, True), (6.0, False)):
        s = make(num_blocks=8, wait_weight=1.0, max_skip_wait=5.0)  # 32 slots
        blocker(s, 16)  # 4 blocks held, 4 free
        s.add_request([6] * 20, SamplingParams(max_tokens=1), 0.0)  # needs 5 blocks: never fits now
        small = s.add_request([7] * 3, SamplingParams(max_tokens=1), 0.5)
        batch, _ = run_step(s, now)
        admitted = [q.seq.request_id for q in batch if q.seq.request_id != 0]
        assert admitted == ([small] if skip_expected else [])


def test_token_budget_limits_prefills():
    s = make(max_num_seqs=2, max_num_batched_tokens=66)
    s.add_request([3] * 40, SamplingParams(max_tokens=2), 0.0)
    s.add_request([4] * 40, SamplingParams(max_tokens=2), 0.1)
    batch, _ = run_step(s, 1.0)
    assert len(batch) == 1  # 80 prompt tokens exceed the 66-token budget


def test_sequence_cap_counts_a_groups_future_samples():
    s = make(max_num_seqs=3, max_num_batched_tokens=67)
    s.add_request([3] * 5, SamplingParams(max_tokens=3, n=3), 0.0)
    s.add_request([4] * 5, SamplingParams(max_tokens=3), 0.1)
    batch, _ = run_step(s, 1.0)
    assert len(batch) == 1 and batch[0].num_samples == 3
    assert len(s.running) == 3


def test_forks_owed_by_earlier_admissions_count_against_the_cap():
    s = make(max_num_seqs=4, max_num_batched_tokens=68)
    s.add_request([3] * 5, SamplingParams(max_tokens=3, n=3), 0.0)
    s.add_request([4] * 5, SamplingParams(max_tokens=3, n=2), 0.1)  # 1 + 2 owed + 1 + 1 > 4
    batch, _ = run_step(s, 1.0)
    assert len(batch) == 1 and len(s.running) == 3


def test_admission_respects_the_watermark_but_decode_may_use_it():
    s = make(num_blocks=10, watermark=0.2)  # 2 blocks held back from admission
    assert s.watermark_blocks == 2
    a = s.add_request([5] * 16, SamplingParams(max_tokens=4, ignore_eos=True), 0.0)  # 4 blocks
    b = s.add_request([6] * 12, SamplingParams(max_tokens=4, ignore_eos=True), 0.0)  # 3 blocks
    batch, _ = run_step(s, 0.0)
    assert len(batch) == 2 and s.kv.num_available_blocks() == 3
    s.add_request([7] * 5, SamplingParams(max_tokens=1), 0.0)  # 2 blocks + 2 watermark > 3 free
    batch, _ = run_step(s, 0.1)
    assert sorted(q.seq.request_id for q in batch) == [a, b]  # decodes only
    assert s.kv.num_available_blocks() == 1  # both decodes took a block from the reserve


# --- preemption --------------------------------------------------------------------------------


def test_the_lowest_score_is_preempted_and_resumes_with_the_same_output():
    s = make(num_blocks=6, wait_weight=1.0)  # 24 slots
    first = s.add_request([2] * 8, SamplingParams(max_tokens=10, ignore_eos=True), 0.0)
    second = s.add_request([3] * 8, SamplingParams(max_tokens=10, ignore_eos=True), 1.0)
    done = run_to_completion(s)
    victims = [q for o in done.values() for q in o.sequences if q.num_preemptions]
    assert victims and all(q.request_id == second for q in victims)  # the later arrival scores lower
    for rid, prompt in ((first, [2] * 8), (second, [3] * 8)):
        (q,) = done[rid].sequences
        assert q.output_token_ids == reference(prompt, q.params, 0)


def preempted_after_one_decode(s: Scheduler, prompts: list[list[int]], arrivals: list[float]) -> list[int]:
    """Admits the prompts together, runs one decode step, returns the request ids that were preempted."""
    rids = [s.add_request(p, SamplingParams(max_tokens=4, ignore_eos=True), t) for p, t in zip(prompts, arrivals)]
    batch, _ = run_step(s, 3.0)
    assert len(batch) == len(prompts) and s.kv.num_available_blocks() == 1
    run_step(s, 3.1)  # every sequence needs a new block; only one is free
    return [q.request_id for q in s.waiting]


def test_the_victim_is_the_lowest_score_by_wait():
    s = make(num_blocks=9, wait_weight=1.0)  # four 2-block prompts leave one block free
    victims = preempted_after_one_decode(s, [[2] * 8, [3] * 8, [4] * 8, [5] * 8], [0.0, 1.0, 2.0, 2.5])
    assert victims == [3]  # the latest arrival, not merely the lowest of those still unscheduled


def test_the_victim_is_the_lowest_score_by_admission_hit():
    s = make(num_blocks=9, wait_weight=0.0)
    warm(s, [2] * 8)  # two cached pages
    # hits of 8, 4 and 0 tokens; 8 distinct blocks in all, one left free
    victims = preempted_after_one_decode(s, [[2] * 8 + [3] * 4, [2] * 4 + [4] * 8, [5] * 12], [0.0] * 3)
    assert victims == [3]  # request 0 was the warm-up


def test_a_sequence_preempts_itself_when_nothing_ranks_lower():
    s = make(num_blocks=5)  # the request's bound: 4 blocks at full length, plus 1 for a partial copy
    s.add_request([2] * 12, SamplingParams(max_tokens=4, ignore_eos=True), 0.0)  # 3 full blocks
    run_step(s, 0.0)
    s.kv.allocate(999, [9] * 8)  # something outside the scheduler takes the last two blocks
    forward(s.kv, 999)
    s.kv.commit(999, 8)
    assert s.schedule(0.1) == []
    (victim,) = s.waiting
    assert victim.num_preemptions == 1 and not s.running and victim.seq_id not in s.kv.seqs
    s.kv.release(999)
    (readmit,), _ = run_step(s, 0.2)
    assert readmit.seq is victim and readmit.token_ids[-1] == victim.output_token_ids[0]  # the old pending token
    stats = s.stats()
    assert (stats.admissions, stats.readmissions, stats.preemptions) == (1, 1, 1)
    assert stats.cached_tokens == 12  # release published all three full prompt blocks; only the pending token is fed


def test_prefill_first_runs_separate_steps():
    s = make(step_mode="prefill_first")
    s.add_request([2] * 5, SamplingParams(max_tokens=5, ignore_eos=True), 0.0)
    run_step(s, 0.0)
    s.add_request([3] * 5, SamplingParams(max_tokens=5, ignore_eos=True), 0.1)
    batch, _ = run_step(s, 0.1)
    assert [len(q.token_ids) for q in batch] == [5]  # prefill only; the running sequence waits
    batch, _ = run_step(s, 0.2)
    assert [len(q.token_ids) for q in batch] == [1, 1]


# --- parallel sampling -------------------------------------------------------------------------


def test_a_group_forks_after_its_prefill_and_returns_once():
    s = make()
    prompt = [2, 3, 4, 5, 6, 7]
    rid = s.add_request(prompt, SamplingParams(max_tokens=4, n=3, ignore_eos=True), 0.0)
    (prefill,), _ = run_step(s, 0.0)
    assert prefill.num_samples == 3 and len(s.running) == 3
    stats = s.stats()
    assert (stats.admissions, stats.num_running) == (1, 3)  # forked samples are not admissions
    tables = [s.kv.seqs[q.seq_id].block_table for q in s.running]
    assert tables[0][0] == tables[1][0] == tables[2][0]  # the full prompt block is shared
    done = run_to_completion(s)
    (out,) = done.values()
    assert out.request_id == rid and len(out.sequences) == 3
    for k, q in enumerate(out.sequences):
        assert q.output_token_ids == reference(prompt, q.params, k)


# --- everything at once ------------------------------------------------------------------------


@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize("seed", range(30))
def test_random_workload_matches_unbatched_generation(seed, caching):
    """With caching on or off, every output must equal generation with no batching and no preemption:
    the prefix cache may change how much is computed, never what is generated."""
    rng = random.Random(seed)
    s = make(
        caching=caching,
        num_blocks=rng.choice([12, 16, 24]),
        wait_weight=rng.choice([0.0, 0.1, 1000.0]),
        max_skip_wait=rng.choice([0.0, 0.5, 5.0]),
        hit_mode=rng.choice(["absolute", "ratio"]),
        step_mode=rng.choice(["mixed", "prefill_first"]),
        watermark=rng.choice([0.0, 0.1]),
    )
    prefixes = [[rng.randrange(2, 2 + VOCAB) for _ in range(rng.randrange(1, 10))] for _ in range(3)]
    pending = []
    for i in range(25):
        prompt = rng.choice(prefixes) + [rng.randrange(2, 2 + VOCAB) for _ in range(rng.randrange(1, 8))]
        params = SamplingParams(max_tokens=rng.randrange(1, 12), n=rng.choice([1, 1, 1, 2, 3]))
        pending.append((i * rng.choice([0.0, 0.05, 0.3]), prompt, params))

    requests, done, now = {}, {}, 0.0
    for _ in range(3000):
        while pending and pending[0][0] <= now:
            arrival, prompt, params = pending.pop(0)
            try:
                requests[s.add_request(prompt, params, arrival)] = (prompt, params)
            except ValueError:
                pass
        if not pending and not s.has_unfinished():
            break
        now += 0.1
        for o in run_step(s, now)[1]:
            assert o.request_id not in done
            done[o.request_id] = o
    else:
        raise AssertionError("the scheduler stopped making progress")

    assert done.keys() == requests.keys()
    for rid, (prompt, params) in requests.items():
        assert len(done[rid].sequences) == params.n
        for k, q in enumerate(done[rid].sequences):
            assert q.output_token_ids == reference(prompt, params, k), (rid, k)
    m = s.kv
    assert not m.seqs and not s.groups and m.num_available_blocks() == m.num_blocks, "no block leaks"
    if not caching:
        assert s.stats().cached_tokens == 0


def test_stats_returns_an_independent_snapshot():
    s = make()
    s.add_request([2, 3, 4], SamplingParams(max_tokens=2), 0.0)
    snapshot = s.stats()
    assert (snapshot.num_waiting, snapshot.prefill_tokens) == (1, 0)
    run_to_completion(s)
    assert (snapshot.num_waiting, snapshot.prefill_tokens) == (1, 0)
    assert (s.stats().prefill_tokens, s.stats().decode_tokens, s.stats().num_waiting) == (3, 1, 0)
