"""
Build a multi-turn replay trace from ShareGPT V3 for one tokenizer, and report how much KV it needs.

    python -m benchmarks.build_trace --sharegpt benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
        --tokenizer benchmarks/data/tokenizers/llama --format llama --out benchmarks/data/trace_llama.json

The source file comes from huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered. Its
redistribution rights are unclear, so traces are rebuilt locally from it rather than published.
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoConfig, AutoTokenizer

from benchmarks.multiturn import ChatFormat, build_trace, load_sharegpt, peak_demand, save_trace

# Pin whatever the template would otherwise vary between runs, and pick each model's chat mode.
FORMATS = {
    "llama": {"date_string": "26 Jul 2024"},  # the template's own default date
    "qwen3": {"enable_thinking": False},
}


def kv_bytes_per_token(config, bytes_per_value: int = 2) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * bytes_per_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharegpt", required=True)
    parser.add_argument("--tokenizer", required=True, help="a directory with the tokenizer and config.json")
    parser.add_argument("--format", required=True, choices=sorted(FORMATS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kv-capacity-gib", type=float, default=70.0,
                        help="KV memory assumed for the rho table, e.g. an 80 GB card at 0.9 minus weights")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    fmt = ChatFormat(tokenizer, **FORMATS[args.format])
    conversations = load_sharegpt(args.sharegpt)
    trace = build_trace(conversations, fmt, args.max_model_len, args.min_turns, args.seed)
    meta = {"source": "ShareGPT_V3_unfiltered_cleaned_split.json", "tokenizer": args.tokenizer,
            "format": args.format, "template_kwargs": FORMATS[args.format],
            "max_model_len": args.max_model_len, "min_turns": args.min_turns, "seed": args.seed}
    save_trace(args.out, trace, meta)

    turns = sorted(len(c.turns) for c in trace)
    lengths = sorted(c.history_len(len(c.turns)) for c in trace)
    print(f"{len(conversations)} conversations after merging; {len(trace)} kept with >= {args.min_turns} turns")
    print(f"turns: median {turns[len(turns) // 2]}, p90 {turns[int(len(turns) * 0.9)]}, max {turns[-1]}")
    print(f"final length in tokens: median {lengths[len(lengths) // 2]}, p90 {lengths[int(len(lengths) * 0.9)]}")

    per_token = kv_bytes_per_token(AutoConfig.from_pretrained(args.tokenizer))
    capacity = int(args.kv_capacity_gib * 2**30 // per_token)
    print(f"KV: {per_token / 1024:.0f} KiB per token; {args.kv_capacity_gib:.0f} GiB holds {capacity:,} tokens")
    print(f"{'conversations':>14} {'peak demand':>12} {'at wave':>8} {'rho at capacity':>16}")
    for n in (25, 50, 100, 200, 400, 800, len(trace)):
        if n > len(trace):
            continue
        demand, wave = peak_demand(trace[:n])
        print(f"{n:>14} {demand:>12,} {wave:>8} {demand / capacity:>16.2f}")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
