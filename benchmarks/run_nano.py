"""
Replay a multi-turn trace through stock nano-vllm in one process, and write one JSON record in the
same format as run_ours. Runs in nano-vllm's own environment; nano-vllm itself is not modified.

    python -m benchmarks.run_nano --model /path/to/Qwen3-0.6B --trace benchmarks/data/trace_qwen3.json \
        --conversations 300 --rho 1 --out benchmarks/results/nano_rho1.json

nano-vllm reports none of the counts we need, so this file reads them from outside, by wrapping
three methods on the live objects: LLMEngine.step (whose return value already carries each prefill
step's token count), Scheduler.preempt, and Scheduler.postprocess (where a sequence's first token
is appended). Its KV capacity is pinned the same way: nano-vllm always sizes its cache from free
memory, then builds its block manager from config.num_kvcache_blocks, so the count is lowered to
ours in between. Its KV tensor stays larger than needed, but it can only hand out that many blocks.

--rho must be a number: "full" means each engine's own memory accounting, which would not match.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner

from benchmarks.multiturn import (BLOCK_SIZE, WaveRecord, WaveResult, blocks_for, load_trace, peak_demand, replay,
                                  summarize)

MAX_NUM_SEQS = 512


def pin_kv_blocks(num_blocks: int) -> None:
    """Make nano-vllm's block manager use exactly num_blocks; must run before LLM() is built."""
    original = ModelRunner.allocate_kv_cache

    def allocate_then_pin(self):
        original(self)
        available = self.config.num_kvcache_blocks
        if available < num_blocks:
            raise RuntimeError(f"nano-vllm fits only {available} KV blocks; {num_blocks} were requested")
        self.config.num_kvcache_blocks = num_blocks

    ModelRunner.allocate_kv_cache = allocate_then_pin


class NanoEngine:
    """One generate() call per wave, with counts read through wrappers on the live engine."""

    def __init__(self, llm: LLM, temperature: float) -> None:
        self.llm = llm
        self.temperature = temperature
        self.prefill_tokens = 0
        self.preemptions = 0
        self.first_token_at: dict[int, float] = {}

        step, preempt, postprocess = llm.step, llm.scheduler.preempt, llm.scheduler.postprocess

        def counted_step():
            outputs, num_tokens = step()
            if num_tokens > 0:  # positive: a prefill step's scheduled tokens; negative: a decode step's batch size
                self.prefill_tokens += num_tokens
            return outputs, num_tokens

        def counted_preempt(seq):
            self.preemptions += 1
            return preempt(seq)

        def timed_postprocess(seqs, token_ids, is_prefill):
            result = postprocess(seqs, token_ids, is_prefill)
            now = time.perf_counter()
            for seq in seqs:
                if seq.num_completion_tokens >= 1 and seq.seq_id not in self.first_token_at:
                    self.first_token_at[seq.seq_id] = now
            return result

        llm.step = counted_step
        llm.scheduler.preempt = counted_preempt
        llm.scheduler.postprocess = timed_postprocess

    def generate(self, prompts: list[list[int]], max_tokens: list[int]) -> WaveResult:
        prefill, preemptions = self.prefill_tokens, self.preemptions
        first_seq_id = next_seq_id()
        params = [SamplingParams(temperature=self.temperature, max_tokens=n, ignore_eos=True) for n in max_tokens]
        start = time.perf_counter()
        results = self.llm.generate(prompts, params, use_tqdm=False)  # sorted by seq_id, which is prompt order
        ids = range(first_seq_id, first_seq_id + len(prompts))
        return WaveResult(
            outputs=[r["token_ids"] for r in results],
            first_token_seconds=[self.first_token_at.pop(i) - start for i in ids],
            prefill_tokens=self.prefill_tokens - prefill,  # includes chunked and recomputed prefill
            preemptions=self.preemptions - preemptions,
        )


def next_seq_id() -> int:
    """nano-vllm numbers sequences from one global counter; peek at the next id without using it up."""
    from itertools import count
    from nanovllm.engine.sequence import Sequence
    current = next(Sequence.counter)
    Sequence.counter = count(current)
    return current


def warm_up(engine: NanoEngine, max_model_len: int) -> None:
    """The same prompts as run_ours' warm-up, so both engines compile the same shapes before timing."""
    for n in (17, 300, 1100):
        engine.generate([[7] * min(n, max_model_len - 4)], [4])
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--conversations", type=int, required=True)
    parser.add_argument("--rho", required=True, help="target memory pressure; must be a number")
    parser.add_argument("--max-waves", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", choices=["on", "off"], default="on", help="off enables CUDA graphs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta, trace = load_trace(args.trace)
    trace = trace[:args.conversations]
    demand, demand_wave = peak_demand(trace)
    num_blocks = blocks_for(demand, float(args.rho), meta["max_model_len"])
    torch.manual_seed(args.seed)  # nano-vllm's sampler draws from torch's global generator

    pin_kv_blocks(num_blocks)
    llm = LLM(args.model, enforce_eager=args.enforce_eager == "on", max_model_len=meta["max_model_len"],
              max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=meta["max_model_len"] + MAX_NUM_SEQS,
              gpu_memory_utilization=args.gpu_memory_utilization, kvcache_block_size=BLOCK_SIZE)
    actual = len(llm.scheduler.block_manager.blocks)
    capacity = actual * BLOCK_SIZE
    print(f"{len(trace)} conversations, peak demand {demand:,} tokens at wave {demand_wave}; "
          f"KV {actual} blocks = {capacity:,} tokens, rho {demand / capacity:.2f}", flush=True)

    engine = NanoEngine(llm, args.temperature)
    warm_up(engine, meta["max_model_len"])

    def progress(r: WaveRecord) -> None:
        print(f"wave {r.wave:>3}: {r.num_prompts:>4} prompts, {r.prompt_tokens:>9,} prompt tokens, "
              f"{r.prefill_tokens:>9,} prefilled, {r.preemptions:>4} preempted, {r.seconds:7.2f} s", flush=True)

    started = time.time()
    records = replay(trace, engine, args.max_waves, progress)
    summary = summarize(records)
    result = {
        "engine": "nano-vllm",
        "model": args.model,
        "trace": args.trace,
        "trace_meta": meta,
        "conversations": len(trace),
        "max_waves": args.max_waves,
        "caching": "on",
        "enforce_eager": args.enforce_eager,
        "rho_requested": args.rho,
        "num_blocks": actual,
        "capacity_tokens": capacity,
        "peak_demand_tokens": demand,
        "rho": demand / capacity,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": meta["max_model_len"] + MAX_NUM_SEQS,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "started": started,
        "summary": summary,
        "waves": [asdict(r) for r in records],
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}: prefill {summary['prefill_tokens']:,} of {summary['prompt_tokens']:,} prompt tokens "
          f"({summary['prefill_fraction']:.1%}), {summary['output_tokens_per_second']:,.0f} output tok/s", flush=True)
    # nano-vllm registers its own exit with atexit, so it is not called here.


if __name__ == "__main__":
    main()
