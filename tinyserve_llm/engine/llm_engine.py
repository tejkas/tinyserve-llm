"""
The engine loop: schedule, run the model, sample, hand the tokens back to the scheduler.

Bootstrap on a GPU: load the weights, run one worst-case step to measure peak activation memory,
then give the KV cache whatever memory is left under the utilisation limit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Literal

import torch

from tinyserve_llm.engine.interface import CacheConfig, CacheStats, ModelShape, build, bytes_per_block
from tinyserve_llm.engine.model_runner import AttentionImpl, ModelRunner
from tinyserve_llm.model.loader import load_model
from tinyserve_llm.scheduler.scheduler import RequestOutput, Scheduler, SchedulerConfig, SchedulerStats
from tinyserve_llm.scheduler.sequence import SamplingParams

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StepRecord:
    """One engine step. The stats are running totals: subtract consecutive records for per-step numbers."""
    start: float
    end: float
    num_seqs: int
    num_tokens: int
    scheduler: SchedulerStats
    cache: CacheStats


@dataclass(slots=True)
class Completion:
    """A finished request: its n samples in fork order, with the timing benchmarks need."""
    request_id: int
    prompt_token_ids: list[int]
    token_ids: list[list[int]]
    texts: list[str] | None
    finish_reasons: list[Literal["eos", "length"]]
    arrival_time: float
    first_token_time: float  # earliest across the samples
    finish_time: float       # latest across the samples


def warmup_prompt_lengths(max_num_batched_tokens: int, max_num_seqs: int, max_model_len: int) -> list[int]:
    """
    Prompt lengths for the worst-case step: as many tokens as the budget allows (peak activations)
    spread over as many sequences as the cap allows (peak sampled rows).
    """
    longest = max_model_len - 1  # each warm-up request samples one token
    remaining, lengths = max_num_batched_tokens, []
    while len(lengths) < max_num_seqs - 1 and remaining - (max_num_seqs - 1 - len(lengths)) >= longest:
        lengths.append(longest)
        remaining -= longest
    if len(lengths) < max_num_seqs:
        lengths.append(max(1, min(longest, remaining - (max_num_seqs - 1 - len(lengths)))))
    lengths += [1] * (max_num_seqs - len(lengths))
    return lengths


class LLM:
    def __init__(
            self,
            model_path: str,
            scheduler_config: SchedulerConfig = SchedulerConfig(),
            cache_config: CacheConfig = CacheConfig(),
            *,
            dtype: torch.dtype = torch.bfloat16,
            num_blocks: int | None = None,  # None: size from free GPU memory after a warm-up step
            gpu_memory_utilization: float = 0.9,
            attention: AttentionImpl | None = None,  # default: flash on CUDA, torch elsewhere
            seed: int | None = None,
            load_tokenizer: bool = True,
    ) -> None:
        self.device = torch.device(cache_config.device)
        self.model = load_model(model_path, dtype=dtype, device=self.device)
        config = self.model.config
        assert scheduler_config.max_model_len <= config.max_position_embeddings
        self.tokenizer = None
        if load_tokenizer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if scheduler_config.eos_token_id == -1:
            scheduler_config = replace(scheduler_config, eos_token_id=self._eos_token_id())
        self.shape = ModelShape.from_hf_config(config, dtype)
        self.attention = attention or ("flash" if self.device.type == "cuda" else "torch")
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(self.device).manual_seed(seed)

        if num_blocks is None:
            if self.device.type != "cuda":
                raise ValueError("num_blocks is required off the GPU: there is no free memory to size from")
            num_blocks = self._profile_num_blocks(scheduler_config, cache_config, gpu_memory_utilization)
        self.scheduler, self.view = build(num_blocks, self.shape, scheduler_config, cache_config)
        self.runner = ModelRunner(self.model, self.view, self.device, self.attention, self.generator)
        self.steps: list[StepRecord] = []
        logger.info("KV cache: %d blocks of %d tokens", num_blocks, cache_config.block_size)

    def _eos_token_id(self) -> int:
        # Llama 3 Instruct lists several end tokens; the tokenizer's is the chat turn end (<|eot_id|>).
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        eos = self.model.config.eos_token_id
        return eos[0] if isinstance(eos, list) else (eos if eos is not None else -1)

    def _profile_num_blocks(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig,
                            utilisation: float) -> int:
        """Run one worst-case step through a throwaway cache, then size the pool from what is left."""
        lengths = warmup_prompt_lengths(scheduler_config.max_num_batched_tokens, scheduler_config.max_num_seqs,
                                        scheduler_config.max_model_len)
        block_size = cache_config.block_size
        warmup_blocks = sum(-(-n // block_size) for n in lengths) + 1
        scheduler, view = build(warmup_blocks, self.shape, replace(scheduler_config, watermark=0.0), cache_config)
        for n in lengths:
            scheduler.add_request([0] * n, SamplingParams(max_tokens=1, ignore_eos=True), 0.0)
        batch = scheduler.schedule(0.0)
        assert len(batch) == len(lengths), "the warm-up step must admit every warm-up prompt"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()  # weights + the throwaway cache
        ModelRunner(self.model, view, self.device, self.attention).run(batch)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del scheduler, view, batch
        torch.cuda.empty_cache()

        free, total = torch.cuda.mem_get_info()
        budget = total * utilisation - (total - free) - peak
        num_blocks = int(budget // bytes_per_block(self.shape, block_size))
        logger.info("warm-up: %d tokens over %d sequences, peak activations %.2f GiB",
                    sum(lengths), len(lengths), peak / 2**30)
        if num_blocks <= 0:
            raise RuntimeError("no memory left for the KV cache; lower max_num_batched_tokens or raise utilisation")
        return num_blocks

    def add_request(self, prompt: str | list[int], params: SamplingParams, arrival_time: float | None = None) -> int:
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        return self.scheduler.add_request(token_ids, params, time.perf_counter() if arrival_time is None else arrival_time)

    def step(self) -> list[RequestOutput]:
        start = time.perf_counter()
        batch = self.scheduler.schedule(start)
        if not batch:
            # Only possible if everything left waiting is blocked; add_request rejects what can never fit.
            raise RuntimeError("the scheduler produced an empty step")
        sampled = self.runner.run(batch)
        end = time.perf_counter()
        outputs = self.scheduler.postprocess(batch, sampled, end)
        self.steps.append(StepRecord(start, end, len(batch), sum(len(fed) for _, fed, _ in batch),
                                     self.scheduler.stats(), self.view.stats()))
        return outputs

    def generate(self, prompts: list[str] | list[list[int]],
                 params: SamplingParams | list[SamplingParams] = SamplingParams()) -> list[Completion]:
        """Run the prompts to completion; results come back in prompt order."""
        params = params if isinstance(params, list) else [params] * len(prompts)
        order = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        done: dict[int, RequestOutput] = {}
        while self.scheduler.has_unfinished():
            for output in self.step():
                done[output.request_id] = output
        return [self._completion(done[request_id]) for request_id in order]

    def _completion(self, output: RequestOutput) -> Completion:
        seqs = output.sequences
        token_ids = [list(s.output_token_ids) for s in seqs]
        texts = None
        if self.tokenizer is not None:
            texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_ids]
        return Completion(
            request_id=output.request_id,
            prompt_token_ids=list(seqs[0].prompt_token_ids),
            token_ids=token_ids,
            texts=texts,
            finish_reasons=[s.finish_reason for s in seqs],
            arrival_time=seqs[0].arrival_time,
            first_token_time=min(s.first_token_time for s in seqs),
            finish_time=max(s.finish_time for s in seqs),
        )
