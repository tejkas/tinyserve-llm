"""
Multi-turn chat replay, shared by every engine's runner so all of them see exactly the same workload.

Conversations come from ShareGPT. Each turn's prompt is the conversation so far as token ids: the
previous prompt, the engine's own generated reply, then the turn-boundary tokens and the next user
message. Prompts are never rebuilt by re-applying the chat template, because decode-then-encode is
not an identity and templates rewrite history (Llama 3's inserts the date, Qwen3's strips thinking
blocks from earlier assistant turns); either would break prefix reuse for reasons unrelated to the
cache. Every reply is generated with ignore_eos to exactly the dataset reply's length, so turn k's
prompt has the same length on every engine even though the replies differ.

Turns run in waves: wave k sends turn k of every conversation that has one, in a single generate()
call, which is the only way to drive an offline engine such as nano-vllm. Between waves nothing is
running, so how much of each history survives in the prefix cache until its next turn is exactly
what the cache policy decides.

This module imports nothing from tinyserve_llm, so nano-vllm's environment can use it. A built trace
is plain JSON of token ids; building one needs a Hugging Face tokenizer, replaying one does not.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol


# --- the trace ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class Turn:
    user_ids: list[int]  # fed before this turn's reply: the whole first prompt, or the boundary + next user message
    reply_len: int       # tokens to generate: the dataset reply's length in this tokenizer


@dataclass(slots=True)
class Conversation:
    id: str
    turns: list[Turn]

    def history_len(self, k: int) -> int:
        """Tokens in the conversation after turn k's reply (k counts from 1)."""
        return sum(len(t.user_ids) + t.reply_len for t in self.turns[:k])


def load_sharegpt(path: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    ShareGPT V3 as (conversation id, [(user, assistant), ...]). The file splits long conversations
    into chunks `<id>_<offset>`, where offset is the index of the chunk's first message in the whole
    conversation; neighbouring chunks can overlap by a message. Each message is put back at its
    position, and the conversation ends at the first position no chunk covers, since joining across
    a gap would splice unrelated text together. It is then kept up to its first malformed message:
    not alternating user/assistant starting with the user, or empty.
    """
    with open(path) as f:
        rows = json.load(f)
    positions: dict[str, dict[int, dict]] = {}
    for row in rows:
        base, _, offset = row["id"].rpartition("_")
        slots = positions.setdefault(base, {})
        for j, message in enumerate(row["conversations"]):
            slots.setdefault(int(offset) + j, message)

    out = []
    for base, slots in positions.items():
        messages: list[dict] = []
        while len(messages) in slots:
            messages.append(slots[len(messages)])
        pairs: list[tuple[str, str]] = []
        for j in range(0, len(messages) - 1, 2):
            user, reply = messages[j], messages[j + 1]
            if user["from"] != "human" or reply["from"] != "gpt" or not user["value"].strip() or not reply["value"].strip():
                break
            pairs.append((user["value"], reply["value"]))
        if pairs:
            out.append((base, pairs))
    return out


class ChatFormat:
    """
    A tokenizer's chat template, cut into the two pieces a token-level conversation needs: the first
    prompt, and the continuation that follows a generated reply (end of the assistant turn, the next
    user message, the start of the next assistant turn). Both come from rendering the template around
    sentinel strings, so no template is hand-copied.
    """

    SENTINEL_REPLY = "⁣REPLY⁣"
    SENTINEL_USER = "⁣USER⁣"

    def __init__(self, tokenizer, **template_kwargs) -> None:
        # template_kwargs pin anything the template would otherwise vary, e.g. Llama 3's date_string,
        # and select modes such as Qwen3's enable_thinking=False.
        self.tokenizer = tokenizer
        self.kwargs = template_kwargs
        text = self._render([("user", "x"), ("assistant", self.SENTINEL_REPLY), ("user", self.SENTINEL_USER)])
        after_reply = text.split(self.SENTINEL_REPLY, 1)[1]
        self.before_user, self.after_user = after_reply.split(self.SENTINEL_USER, 1)

    def _render(self, messages: list[tuple[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": r, "content": c} for r, c in messages], tokenize=False, add_generation_prompt=True,
            **self.kwargs)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)  # the template carries its own BOS

    def first(self, user: str) -> list[int]:
        return self._encode(self._render([("user", user)]))

    def continuation(self, user: str) -> list[int]:
        return self._encode(self.before_user + user + self.after_user)


def build_trace(conversations: list[tuple[str, list[tuple[str, str]]]], fmt: ChatFormat, max_model_len: int,
                min_turns: int, seed: int = 0) -> list[Conversation]:
    """
    Tokenize and filter. A conversation is cut at the last turn whose prompt plus reply still fits in
    max_model_len, rather than dropped: the long ones are where caching matters most. Kept if at least
    min_turns survive. Returned in a seeded random order, so the first n of a trace are a random
    sample of n and every engine reading it takes the same n.
    """
    order = list(range(len(conversations)))
    random.Random(seed).shuffle(order)
    trace = []
    for i in order:
        cid, pairs = conversations[i]
        turns: list[Turn] = []
        total = 0
        for k, (user, reply) in enumerate(pairs):
            user_ids = fmt.first(user) if k == 0 else fmt.continuation(user)
            reply_len = len(fmt._encode(reply))
            if total + len(user_ids) + reply_len > max_model_len:
                break
            turns.append(Turn(user_ids, reply_len))
            total += len(user_ids) + reply_len
        if len(turns) >= min_turns:
            trace.append(Conversation(cid, turns))
    return trace


def save_trace(path: str, trace: list[Conversation], meta: dict) -> None:
    with open(path, "w") as f:
        json.dump({"meta": meta, "conversations": [asdict(c) for c in trace]}, f)


def load_trace(path: str) -> tuple[dict, list[Conversation]]:
    with open(path) as f:
        raw = json.load(f)
    convs = [Conversation(c["id"], [Turn(t["user_ids"], t["reply_len"]) for t in c["turns"]]) for c in raw["conversations"]]
    return raw["meta"], convs


def peak_demand(trace: list[Conversation]) -> tuple[int, int]:
    """
    (tokens, wave) at the point where keeping every live history costs the most KV: after wave k,
    every conversation that still has a turn k+1 holds k turns of history. That is the memory the
    cache needs to serve every next turn from cache. Capacity divided by this gives the pressure rho.
    """
    best, best_k = 0, 0
    for k in range(1, max(len(c.turns) for c in trace)):
        tokens = sum(c.history_len(k) for c in trace if len(c.turns) > k)
        if tokens > best:
            best, best_k = tokens, k
    return best, best_k


# --- the replay --------------------------------------------------------------------------------------


@dataclass(slots=True)
class WaveResult:
    """What an engine reports for one generate() call, measured inside its runner."""
    outputs: list[list[int]]                 # generated ids, in prompt order
    first_token_seconds: list[float]         # per prompt, from the wave's start to its first token
    prefill_tokens: int                      # tokens the engine actually ran through prefill
    preemptions: int


class Engine(Protocol):
    def generate(self, prompts: list[list[int]], max_tokens: list[int]) -> WaveResult: ...


@dataclass(slots=True)
class WaveRecord:
    wave: int
    num_prompts: int
    prompt_tokens: int
    output_tokens: int
    prefill_tokens: int
    preemptions: int
    seconds: float
    first_token_seconds: list[float] = field(default_factory=list)


def replay(trace: list[Conversation], engine: Engine, max_waves: int | None = None,
           on_wave: Callable[[WaveRecord], None] | None = None) -> list[WaveRecord]:
    """Run every conversation turn by turn, in waves, feeding each engine its own replies back."""
    history: list[list[int]] = [[] for _ in trace]
    waves = max(len(c.turns) for c in trace)
    records = []
    for k in range(waves if max_waves is None else min(waves, max_waves)):
        live = [i for i, c in enumerate(trace) if len(c.turns) > k]
        prompts = [history[i] + trace[i].turns[k].user_ids for i in live]
        max_tokens = [trace[i].turns[k].reply_len for i in live]
        start = time.perf_counter()
        result = engine.generate(prompts, max_tokens)
        seconds = time.perf_counter() - start
        for i, prompt, out, n in zip(live, prompts, result.outputs, max_tokens, strict=True):
            assert len(out) == n, f"conversation {trace[i].id} turn {k + 1}: asked for {n} tokens, got {len(out)}"
            history[i] = prompt + out
        record = WaveRecord(k + 1, len(live), sum(map(len, prompts)), sum(max_tokens), result.prefill_tokens,
                            result.preemptions, seconds, result.first_token_seconds)
        records.append(record)
        if on_wave:
            on_wave(record)
    return records


def summarize(records: list[WaveRecord]) -> dict:
    prompt = sum(r.prompt_tokens for r in records)
    prefill = sum(r.prefill_tokens for r in records)
    output = sum(r.output_tokens for r in records)
    seconds = sum(r.seconds for r in records)
    return {
        "prompt_tokens": prompt,
        "prefill_tokens": prefill,
        "prefill_fraction": prefill / prompt,
        "cached_tokens": prompt - prefill,
        "output_tokens": output,
        "seconds": seconds,
        "output_tokens_per_second": output / seconds,
        "preemptions": sum(r.preemptions for r in records),
        "median_first_token_seconds_by_wave": [_median(r.first_token_seconds) for r in records],
    }


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if s else float("nan")
