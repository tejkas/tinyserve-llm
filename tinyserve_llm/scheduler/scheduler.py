from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

from tinyserve_llm.scheduler.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve_llm.kvcache.manager import AdmissionCost, KVCacheManager


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_model_len: int = 4096
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    eos_token_id: int = -1
    # Ranking: score = hit + wait_weight * wait_seconds. A large wait_weight gives first come, first served.
    wait_weight: float = 0.1  # 10 s of waiting counts as much as a fully cached prompt
    hit_mode: Literal["absolute", "ratio"] = "absolute" # Use longest cached token length
    max_skip_wait: float = 5.0 # A top candidate that does not fit is skipped until it has waited this long
    watermark: float = 0.01  # fraction of all blocks kept free at admission
    step_mode: Literal["mixed", "prefill_first"] = "mixed"

    def __post_init__(self) -> None:
        # Decodes take their share of the budget first, so a maximum-length prompt must still fit beside them
        assert self.max_num_batched_tokens >= self.max_model_len + self.max_num_seqs, "a maximum-length prompt could never be scheduled"

@dataclass(slots=True)
class SchedulerStats:
    """
    Running totals since startup. The engine copies these each step and subtracts consecutive copies.
    """
    admissions: int = 0      # first admissions
    readmissions: int = 0    # admissions after a preemption
    preemptions: int = 0
    cached_tokens: int = 0   # tokens served from the KV cache at admission
    prefill_tokens: int = 0  # tokens fed through the model at admission
    decode_tokens: int = 0   # one per scheduled decode
    num_waiting: int = 0     # current queue sizes
    num_running: int = 0

class ScheduledSeq(NamedTuple):
    seq: Sequence
    token_ids: list[int]  # per step token ids, their positions start at seq.num_tokens - len(token_ids)
    num_samples: int = 1  # n on a group's first prefill, so the sampler draws every sample's first token

class RequestOutput(NamedTuple):
    request_id: int
    sequences: list[Sequence]  # all n samples, each with outputs, finish reason and timestamps

class Scheduler:
    def __init__(self, config: SchedulerConfig, kv: KVCacheManager) -> None:
        self.config = config
        self.kv = kv
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.groups: dict[int, list[Sequence]] = {}  # request_id -> its sequences; grows to n at fork
        self.watermark_blocks = math.ceil(config.watermark * kv.num_blocks)
        self._seq_ids = itertools.count()
        self._request_ids = itertools.count()
        self.counters = SchedulerStats()

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def add_request(self, prompt_token_ids: list[int], params: SamplingParams, arrival_time: float) -> int:
        """
        Queues a request and returns its request_id. Rejects a request that could never run.
        """
        assert prompt_token_ids and params.max_tokens >= 1 and params.n >= 1
        max_len = len(prompt_token_ids) + params.max_tokens
        if max_len > self.config.max_model_len:
            raise ValueError(f"prompt + max_tokens = {max_len} exceeds max_model_len")
        # Conservative: one sequence at full length, plus the group's fork copies, plus the watermark
        max_blocks = -(-max_len // self.kv.block_size) + (params.n - 1) + self.watermark_blocks
        if max_blocks > self.kv.num_blocks:
            raise ValueError(f"needs up to {max_blocks} blocks; the KV cache has {self.kv.num_blocks}")
        if params.n > self.config.max_num_seqs:
            raise ValueError(f"n = {params.n} samples exceed max_num_seqs")

        request_id = next(self._request_ids)
        seq = Sequence(next(self._seq_ids), request_id, tuple(prompt_token_ids), params, arrival_time)
        self.groups[request_id] = [seq]
        self.waiting.append(seq)
        return request_id

    def _hit(self, num_tokens: int, cost: AdmissionCost) -> float:
        """Cache hit used for ranking: cached tokens over max_model_len (absolute) or over the prompt (ratio)."""
        cached = num_tokens - cost.prefill_tokens
        denominator = self.config.max_model_len if self.config.hit_mode == "absolute" else num_tokens
        return cached / denominator

    def _score(self, hit: float, seq: Sequence, now: float) -> float:
        """
        Weighted score of cache hit length and wait time for this sequence.
        """
        return hit + self.config.wait_weight * (now - seq.arrival_time)

    def _rank_waiting(self, now: float) -> list[tuple[Sequence, AdmissionCost, float]]:
        """
        Waiting sequences, best first, each with its admission cost and cache hit.
        Recomputed every step: publishes and evictions since the last step change the hits.
        """
        ranked = []
        for seq in self.waiting:
            token_ids = seq.all_token_ids
            cost = self.kv.calculate_admission_cost(token_ids)
            hit = self._hit(len(token_ids), cost)
            ranked.append((self._score(hit, seq, now), seq, cost, hit))
        ranked.sort(key=lambda r: r[0], reverse=True)
        return [(seq, cost, hit) for _, seq, cost, hit in ranked]

    def _running_score(self, seq: Sequence, now: float) -> float:
        """A running sequence keeps its admission-time hit; its wait keeps growing."""
        return self._score(seq.admission_hit, seq, now)

    def schedule(self, now: float) -> list[ScheduledSeq]:
        """
        Builds one step. Mixed prefill-decode: every running sequence decodes, then prefills fill the remaining token
        budget.
        """
        budget = self.config.max_num_batched_tokens
        if self.config.step_mode == "prefill_first": # a prefill-only step whenever anything waiting fits, else a decode-only step.
            return self._admit(now, budget) or self._schedule_decodes(now)
        decodes = self._schedule_decodes(now)
        return decodes + self._admit(now, budget - len(decodes))

    def _schedule_decodes(self, now: float) -> list[ScheduledSeq]:
        """
        Gives each running sequence one slot, highest score first. When the pool runs short, preempts the
        lowest-scoring sequence not yet scheduled this step, or the sequence itself if none is left.
        """
        queue = sorted(self.running, key=lambda s: self._running_score(s, now))  # lowest score first
        scheduled: list[ScheduledSeq] = []
        self.running = []
        while queue:
            seq = queue.pop()  # highest score left
            # Recheck after every preemption: releasing a sibling can drop this sequence's copy-on-write need
            while self.kv.blocks_needed_for_append(seq.seq_id, 1) > self.kv.num_available_blocks() and queue:
                self._preempt(queue.pop(0))
            if self.kv.blocks_needed_for_append(seq.seq_id, 1) > self.kv.num_available_blocks():
                self._preempt(seq)  # queue is empty, nothing ranked lower is left to give up its blocks
                break
            self.kv.append_slots(seq.seq_id, [seq.pending_token])
            scheduled.append(ScheduledSeq(seq, [seq.pending_token]))
            self.counters.decode_tokens += 1
            self.running.append(seq)
        return scheduled

    def _preempt(self, seq: Sequence) -> None:
        """Preempt by dropping the KV, requeue with outputs and arrival time kept."""
        self.kv.release(seq.seq_id)
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.counters.preemptions += 1
        self.waiting.append(seq)

    def _admit(self, now: float, budget: int) -> list[ScheduledSeq]:
        """
        Admits waiting sequences, best score first, within the token budget, the sequence cap and the
        watermark. A candidate that does not fit is skipped, unless it has waited max_skip_wait seconds,
        then admission stops for this step, so the memory that frees up goes to it.
        """
        admitted: list[ScheduledSeq] = []
        owed_forks = 0 # forks of groups admitted earlier in this step (forking happens in postprocess())
        for seq, cost, hit in self._rank_waiting(now):
            token_ids = seq.all_token_ids
            if admitted:  # an earlier admission this step may have evicted pages this hit counted on
                cost = self.kv.calculate_admission_cost(token_ids)
                hit = self._hit(len(token_ids), cost)
            # A group's parent forks after its first prefill -> n sequences and n - 1 copies of the last block
            first_of_group = seq.params.n > 1 and not seq.output_token_ids
            num_forks = seq.params.n - 1 if first_of_group else 0
            fits = (
                cost.prefill_tokens <= budget
                and len(self.running) + owed_forks + 1 + num_forks <= self.config.max_num_seqs
                and cost.blocks_needed + owed_forks + num_forks + self.watermark_blocks <= self.kv.num_available_blocks()
            )
            if not fits:
                if now - seq.arrival_time >= self.config.max_skip_wait:
                    break  # stop skipping past a sequence that has waited this long
                continue
            assert self.kv.allocate(seq.seq_id, token_ids) is not None  # the fit check used a fresh cost
            fed = token_ids[self.kv.num_cached_tokens(seq.seq_id):]

            if seq.num_preemptions:
                self.counters.readmissions += 1
            else:
                self.counters.admissions += 1
            self.counters.cached_tokens += len(token_ids) - len(fed)
            self.counters.prefill_tokens += len(fed)

            budget -= len(fed)
            owed_forks += num_forks
            seq.status = SequenceStatus.RUNNING
            seq.admission_hit = hit
            self.running.append(seq)
            admitted.append(ScheduledSeq(seq, fed, seq.params.n if first_of_group else 1))
        if admitted:
            chosen = {s.seq.seq_id for s in admitted}
            self.waiting = [s for s in self.waiting if s.seq_id not in chosen]
        return admitted

    def _fork(self, parent: Sequence) -> Sequence:
        """
        A new sample of the parent's request, sharing its prompt KV
        """
        assert not parent.output_token_ids
        child = Sequence(next(self._seq_ids), parent.request_id, parent.prompt_token_ids, parent.params,
                         parent.arrival_time, status=SequenceStatus.RUNNING, admission_hit=parent.admission_hit)
        self.kv.fork(parent.seq_id, child.seq_id)
        self.running.append(child)
        self.groups[parent.request_id].append(child)
        return child

    def _finish_reason(self, seq: Sequence, token: int) -> Literal["eos", "length"] | None:
        if token == self.config.eos_token_id and not seq.params.ignore_eos:
            return "eos"
        if len(seq.output_token_ids) >= seq.params.max_tokens:
            return "length"
        return None

    def _collect(self, finished: list[Sequence]) -> list[RequestOutput]:
        """
        Requests whose samples have all finished.
        """
        outputs = []
        for request_id in dict.fromkeys(s.request_id for s in finished):  # dedupe, keep order
            group = self.groups[request_id]
            if all(s.status is SequenceStatus.FINISHED for s in group):
                outputs.append(RequestOutput(request_id, self.groups.pop(request_id)))
        return outputs

    def postprocess(self, scheduled: list[ScheduledSeq], sampled: list[list[int]], now: float) -> list[RequestOutput]:
        """
        Finishes the step by:
        - Committing each sequence's slots
        - Forking a group's samples after the first prefill
        - Recording the new tokens
        - Releasing finished sequences
        """
        finished: list[Sequence] = []
        for (seq, token_ids, num_samples), tokens in zip(scheduled, sampled, strict=True):
            assert len(tokens) == num_samples
            self.kv.commit(seq.seq_id, len(token_ids)) # 1 for decode, the uncached prompt for prefill
            samples = [seq] + [self._fork(seq) for _ in range(num_samples - 1)] # fork parent into n-1 samples
            for sample, token in zip(samples, tokens):
                sample.output_token_ids.append(token)
                if sample.first_token_time is None:
                    sample.first_token_time = now
                reason = self._finish_reason(sample, token)
                if reason is not None:
                    self.kv.release(sample.seq_id)
                    sample.status = SequenceStatus.FINISHED
                    sample.finish_reason = reason
                    sample.finish_time = now
                    finished.append(sample)
        if finished:
            done = {s.seq_id for s in finished}
            self.running = [s for s in self.running if s.seq_id not in done]
        return self._collect(finished)

    def stats(self) -> SchedulerStats:
        """A snapshot: the running totals plus current queue sizes."""
        return replace(self.counters, num_waiting=len(self.waiting), num_running=len(self.running))