from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal

from tinyserve_llm.kvcache.manager import KVCacheManager
from tinyserve_llm.scheduler.sequence import SamplingParams, Sequence
from tinyserve_llm.kvcache.manager import AdmissionCost, KVCacheManager


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_model_len: int = 4096
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    eos_token_id: int = -1
    # Ranking: score = hit + wait_weight * wait_seconds. A large wait_weight gives first come, first served.
    wait_weight: float = 0.1  # 10 s of waiting counts as much as a fully cached prompt. In other words, 10s is the cap on wait
    hit_mode: Literal["absolute", "ratio"] = "absolute" # Use longest cached token length
    max_skip_wait: float = 5.0 # A top candidate that does not fit is skipped until it has waited this long
    watermark: float = 0.01  # fraction of all blocks kept free at admission
    step_mode: Literal["mixed", "prefill_first"] = "mixed"

    def __post_init__(self) -> None:
        # Decodes take their share of the budget first, so a maximum-length prompt must still fit beside them
        assert self.max_num_batched_tokens >= self.max_model_len + self.max_num_seqs, "a maximum-length prompt could never be scheduled"


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