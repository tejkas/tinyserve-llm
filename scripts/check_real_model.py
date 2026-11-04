"""
Real-weights sanity check: our engine's greedy output on a real checkpoint, judged by transformers.

Exact token parity is the wrong test in bf16: one near-tie sends two generations down different
paths without anything being wrong. Instead, every token we pick is scored against HF's logits at
the same position of our own output (one teacher-forced HF forward per completion). A near-tie
shows up as a tiny gap between HF's top logit and the logit of our token; a real bug (wrong RoPE,
a missed norm, a bad slot write) shows up as tokens HF rates far below its own choice.

A second wave reuses a long system prompt, so its prompts are served partly from the prefix cache.

    python scripts/check_real_model.py /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM

from tinyserve_llm.engine.interface import CacheConfig
from tinyserve_llm.engine.llm_engine import LLM
from tinyserve_llm.scheduler.scheduler import SchedulerConfig
from tinyserve_llm.scheduler.sequence import SamplingParams

# Long enough to fill at least one 256-token block, so the second wave gets full-block hits.
SYSTEM = " ".join([
    "You are a careful technical assistant for engineers who build and operate machine learning systems.",
    "Answer in two or three plain sentences. Prefer concrete numbers and mechanisms over adjectives.",
    "If a question is ambiguous, state the assumption you are making before you answer it.",
    "Never invent citations, benchmark results or version numbers; say that you are unsure instead.",
    "When you describe a tradeoff, name what is gained, what is paid for it, and when the balance flips.",
    "Use standard terminology: a token is one unit of model input, the KV cache holds the attention keys",
    "and values of tokens already processed, prefill processes a prompt, and decode generates one token",
    "at a time. Memory is measured in bytes with binary prefixes, and time in milliseconds or seconds.",
    "Assume the reader knows linear algebra and basic GPU programming but not the internals of any",
    "particular inference engine. Do not use bullet points, headings or code blocks unless asked.",
    "Keep each answer self-contained, since readers often see a single answer without the conversation.",
] * 2)

WAVE_1 = [
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Why does decode get slower as the context grows?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is grouped-query attention for?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "When is recomputing a prefill cheaper than storing it?"}],
    [{"role": "user", "content": "Write a haiku about a GPU waiting on memory."}],
    [{"role": "user", "content": "List the planets of the solar system in order from the Sun."}],
    [{"role": "user", "content": "Summarise the following in one sentence:\n\n" + " ".join(
        f"Step {i}: the scheduler admits waiting requests until the token budget or the block pool runs out, "
        f"then every running sequence decodes one token." for i in range(1, 41))}],
]
# Admitted after wave 1 has committed its blocks, so the shared system prompt is cached by then.
WAVE_2 = [
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What does a paged KV cache fix?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Why do inference engines batch requests?"}],
]

MAX_TOKENS = 96
MAX_GAP = 1.0        # fail if HF rates any token we picked this far below its own top choice
MIN_TOP1 = 0.95      # fail if fewer than this share of our tokens are HF's top choice


def encode(tokenizer, messages) -> list[int]:
    # The chat template already starts with <|begin_of_text|>, so add no special tokens again.
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(text, add_special_tokens=False)


@torch.no_grad()
def score(hf, prompt: list[int], output: list[int]) -> tuple[int, float]:
    """(how many of our tokens are HF's argmax, the largest logit gap) over one completion."""
    ids = torch.tensor([prompt + output], device="cuda")
    logits = hf(input_ids=ids).logits[0].float()
    rows = logits[len(prompt) - 1: len(prompt) - 1 + len(output)]  # row j predicts output[j]
    ours = torch.tensor(output, device="cuda")
    gaps = rows.max(dim=-1).values - rows.gather(1, ours[:, None]).squeeze(1)
    return int((gaps == 0).sum()), float(gaps.max())


@torch.no_grad()
def hf_greedy(hf, prompt: list[int], eos: int) -> list[int]:
    out = hf.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=MAX_TOKENS, do_sample=False,
                      eos_token_id=eos, pad_token_id=eos)
    return out[0, len(prompt):].tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    llm = LLM(args.checkpoint, SchedulerConfig(), CacheConfig(device="cuda"), dtype=torch.bfloat16,
              gpu_memory_utilization=0.6)  # leaves room for HF's copy of the model
    tokenizer = llm.tokenizer
    eos = llm.scheduler.config.eos_token_id
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    prompts_1 = [encode(tokenizer, m) for m in WAVE_1]
    prompts_2 = [encode(tokenizer, m) for m in WAVE_2]
    print(f"KV cache: {llm.scheduler.kv.num_blocks} blocks; prompt lengths {[len(p) for p in prompts_1 + prompts_2]}")

    results = llm.generate(prompts_1, params)
    cached_before = llm.scheduler.stats().cached_tokens
    results += llm.generate(prompts_2, params)
    cached_wave_2 = llm.scheduler.stats().cached_tokens - cached_before

    hf = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).cuda().eval()
    total = top1 = 0
    worst = 0.0
    for prompt, result in zip(prompts_1 + prompts_2, results):
        output = result.token_ids[0]
        hits, gap = score(hf, prompt, output)
        reference = hf_greedy(hf, prompt, eos)
        same = next((i for i, (a, b) in enumerate(zip(output, reference)) if a != b), min(len(output), len(reference)))
        total, top1, worst = total + len(output), top1 + hits, max(worst, gap)
        print(f"\n--- {len(prompt)} prompt tokens, {len(output)} generated, finish={result.finish_reasons[0]}")
        print(f"    HF's top choice {hits}/{len(output)}, largest gap {gap:.3f}, "
              f"identical to HF greedy for the first {same} tokens")
        print("    " + tokenizer.decode(output, skip_special_tokens=True).replace("\n", "\n    "))

    share = top1 / total
    print(f"\nOverall: HF's top choice for {top1}/{total} tokens ({share:.1%}), largest gap {worst:.3f}")
    print(f"Wave 2 prefix-cache hits: {cached_wave_2} tokens over {len(prompts_2)} prompts")
    print(f"Cache: {llm.view.stats()}")
    ok = worst <= MAX_GAP and share >= MIN_TOP1 and cached_wave_2 >= 256 * len(prompts_2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
