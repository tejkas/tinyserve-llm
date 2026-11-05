"""
Replay a multi-turn trace through tinyserve-llm in one process, and write one JSON record.

    python -m benchmarks.run_ours --model /path/to/checkpoint --trace benchmarks/data/trace_llama.json \
        --conversations 300 --rho 1 --out benchmarks/results/ours_rho1.json

--rho sets the KV capacity: peak demand / rho tokens, rounded up to whole blocks. "full" sizes the
cache from free GPU memory instead and reports the rho that results.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from benchmarks.multiturn import (BLOCK_SIZE, WaveRecord, WaveResult, blocks_for, load_trace, peak_demand,
                                  replay, summarize)
from tinyserve_llm.engine.interface import CacheConfig
from tinyserve_llm.engine.llm_engine import LLM
from tinyserve_llm.scheduler.scheduler import SchedulerConfig
from tinyserve_llm.scheduler.sequence import SamplingParams

MAX_NUM_SEQS = 512


class OursEngine:
    """One generate() call per wave; counts come from the scheduler's running totals."""

    def __init__(self, llm: LLM, temperature: float) -> None:
        self.llm = llm
        self.temperature = temperature

    def generate(self, prompts: list[list[int]], max_tokens: list[int]) -> WaveResult:
        before = self.llm.scheduler.stats()
        params = [SamplingParams(max_tokens=n, temperature=self.temperature, ignore_eos=True) for n in max_tokens]
        results = self.llm.generate(prompts, params)
        after = self.llm.scheduler.stats()
        return WaveResult(
            outputs=[r.token_ids[0] for r in results],
            first_token_seconds=[r.first_token_time - r.arrival_time for r in results],
            prefill_tokens=after.prefill_tokens - before.prefill_tokens,  # includes recompute after preemption
            preemptions=after.preemptions - before.preemptions,
        )


def warm_up(engine: OursEngine, max_model_len: int) -> None:
    """Compile and autotune before timing, one prompt at a time; several lengths, so dynamic shapes are seen too."""
    for n in (17, 300, 1100):
        n = min(n, max_model_len - 4)
        engine.generate([[7] * n], [4])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--conversations", type=int, required=True, help="the first n of the trace")
    parser.add_argument("--rho", default="full", help="target memory pressure, or 'full' for all free memory")
    parser.add_argument("--caching", choices=["on", "off"], default="on")
    parser.add_argument("--max-waves", type=int, default=None, help="stop early; 1 gives the calibration run")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta, trace = load_trace(args.trace)
    trace = trace[:args.conversations]
    demand, demand_wave = peak_demand(trace)
    num_blocks = None if args.rho == "full" else blocks_for(demand, float(args.rho), meta["max_model_len"])

    scheduler_config = SchedulerConfig(max_model_len=meta["max_model_len"], max_num_seqs=MAX_NUM_SEQS,
                                       max_num_batched_tokens=meta["max_model_len"] + MAX_NUM_SEQS)
    cache_config = CacheConfig(block_size=BLOCK_SIZE, caching=args.caching == "on", device=args.device)
    llm = LLM(args.model, scheduler_config, cache_config, dtype=torch.bfloat16, num_blocks=num_blocks,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, load_tokenizer=False)
    capacity = llm.scheduler.kv.num_blocks * BLOCK_SIZE
    print(f"{len(trace)} conversations, peak demand {demand:,} tokens at wave {demand_wave}; "
          f"KV {llm.scheduler.kv.num_blocks} blocks = {capacity:,} tokens, rho {demand / capacity:.2f}", flush=True)

    engine = OursEngine(llm, args.temperature)
    warm_up(engine, meta["max_model_len"])

    def progress(r: WaveRecord) -> None:
        print(f"wave {r.wave:>3}: {r.num_prompts:>4} prompts, {r.prompt_tokens:>9,} prompt tokens, "
              f"{r.prefill_tokens:>9,} prefilled, {r.preemptions:>4} preempted, {r.seconds:7.2f} s", flush=True)

    started = time.time()
    records = replay(trace, engine, args.max_waves, progress)
    summary = summarize(records)

    result = {
        "engine": "tinyserve-llm",
        "model": args.model,
        "trace": args.trace,
        "trace_meta": meta,
        "conversations": len(trace),
        "max_waves": args.max_waves,
        "caching": args.caching,
        "rho_requested": args.rho,
        "num_blocks": llm.scheduler.kv.num_blocks,
        "capacity_tokens": capacity,
        "peak_demand_tokens": demand,
        "rho": demand / capacity,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
        "torch": torch.__version__,
        "started": started,
        "summary": summary,
        "cache_stats": asdict(llm.view.stats()),
        "waves": [asdict(r) for r in records],
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}: prefill {summary['prefill_tokens']:,} of {summary['prompt_tokens']:,} prompt tokens "
          f"({summary['prefill_fraction']:.1%}), {summary['output_tokens_per_second']:,.0f} output tok/s", flush=True)


if __name__ == "__main__":
    main()
