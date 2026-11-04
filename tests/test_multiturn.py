"""The multi-turn replay's pure logic: ShareGPT merging, trace building, demand, and the wave loop."""

from __future__ import annotations

import json
import os

import pytest

from benchmarks.multiturn import (ChatFormat, Conversation, Turn, WaveResult, build_trace, load_sharegpt,
                                  peak_demand, replay, summarize)


def msg(who, text):
    return {"from": who, "value": text}


def test_chunks_are_merged_by_message_offset_and_stop_at_a_gap(tmp_path):
    rows = [
        # offsets are message indices; the chunk at 3 overlaps message 3 of the first one
        {"id": "a_0", "conversations": [msg("human", "u0"), msg("gpt", "r0"), msg("human", "u1"), msg("gpt", "r1")]},
        {"id": "a_3", "conversations": [msg("gpt", "r1"), msg("human", "u2"), msg("gpt", "r2")]},
        {"id": "a_9", "conversations": [msg("human", "after a gap"), msg("gpt", "never joined")]},
        {"id": "b_0", "conversations": [msg("human", "u0"), msg("gpt", "r0"), msg("gpt", "same side twice"), msg("human", "x")]},
        {"id": "c_0", "conversations": [msg("gpt", "starts with the assistant"), msg("human", "x")]},
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(rows))
    convs = dict(load_sharegpt(str(path)))
    assert convs["a"] == [("u0", "r0"), ("u1", "r1"), ("u2", "r2")]
    assert convs["b"] == [("u0", "r0")]  # cut at the first malformed message
    assert "c" not in convs


class FakeFormat:
    """Each word is one token; the first prompt and the continuation add 2 and 3 template tokens."""

    def first(self, user):
        return [1, 1] + [len(w) for w in user.split()]

    def continuation(self, user):
        return [2, 2, 2] + [len(w) for w in user.split()]

    def _encode(self, text):
        return [len(w) for w in text.split()]


def test_trace_truncates_at_the_last_turn_that_fits_and_drops_short_ones():
    convs = [("long", [("a b", "c d e")] * 5), ("short", [("a", "b")] * 2)]
    # turn 1: 2 + 2 user + 3 reply = 7 tokens; later turns 3 + 2 + 3 = 8 each
    trace = build_trace(convs, FakeFormat(), max_model_len=7 + 8 + 8 + 7, min_turns=3)
    assert [c.id for c in trace] == ["long"]  # "short" has only 2 turns
    assert len(trace[0].turns) == 3 and trace[0].history_len(3) == 23  # a 4th turn would need 31
    assert sorted(c.id for c in build_trace(convs, FakeFormat(), max_model_len=1000, min_turns=2)) == ["long", "short"]


def test_peak_demand_counts_only_histories_with_a_next_turn():
    a = Conversation("a", [Turn([0] * 10, 5), Turn([0] * 2, 3), Turn([0] * 2, 1)])
    b = Conversation("b", [Turn([0] * 4, 4), Turn([0] * 1, 1)])
    # after wave 1: a holds 15 and b holds 8; after wave 2 only a goes on, holding 20
    assert peak_demand([a, b]) == (23, 1)


class EchoEngine:
    """Replies with its own sequence of ids and pretends it cached every prompt token it saw before."""

    def __init__(self):
        self.seen: list[list[int]] = []
        self.counter = 1000

    def generate(self, prompts, max_tokens):
        self.seen.append([list(p) for p in prompts])
        outputs = []
        for n in max_tokens:
            outputs.append(list(range(self.counter, self.counter + n)))
            self.counter += n
        return WaveResult(outputs, [0.1] * len(prompts), prefill_tokens=sum(map(len, prompts)), preemptions=0)


def test_replay_feeds_each_conversation_its_own_replies_in_waves():
    a = Conversation("a", [Turn([1, 1], 2), Turn([2], 1), Turn([3], 1)])
    b = Conversation("b", [Turn([5], 1), Turn([6, 6], 2)])
    engine = EchoEngine()
    records = replay([a, b], engine)
    first, second, third = engine.seen
    assert first == [[1, 1], [5]]
    assert second == [[1, 1, 1000, 1001, 2], [5, 1002, 6, 6]]  # history + own reply + next user turn
    assert third == [[1, 1, 1000, 1001, 2, 1003, 3]]           # b has no third turn
    assert [r.num_prompts for r in records] == [2, 2, 1]
    summary = summarize(records)
    assert summary["prompt_tokens"] == 3 + 9 + 7 and summary["output_tokens"] == 7


def test_replay_rejects_an_engine_that_stops_early():
    class Short(EchoEngine):
        def generate(self, prompts, max_tokens):
            return super().generate(prompts, [n - 1 for n in max_tokens])

    with pytest.raises(AssertionError, match="asked for 2 tokens"):
        replay([Conversation("a", [Turn([1], 2)])], Short())


LLAMA = "benchmarks/data/tokenizers/llama"


@pytest.mark.skipif(not os.path.isdir(LLAMA), reason="needs the local Llama tokenizer files")
def test_token_level_prompts_match_the_llama_template():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA)
    fmt = ChatFormat(tok, date_string="26 Jul 2024")
    ours = fmt.first("What is 2+2?") + tok.encode("It is 4.", add_special_tokens=False) + fmt.continuation("And 3+3?")
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "It is 4."},
         {"role": "user", "content": "And 3+3?"}], tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024")
    assert ours == tok.encode(text, add_special_tokens=False)


# --- our engine through the replay, on CPU ---------------------------------------------------------


def tiny_trace(num_conversations: int = 4, turns: int = 3) -> list[Conversation]:
    import random as _random
    rng = _random.Random(0)
    return [Conversation(str(c), [Turn([rng.randrange(3, 120) for _ in range(rng.randrange(5, 20))], rng.randrange(2, 9))
                                  for _ in range(turns)]) for c in range(num_conversations)]


def ours_engine(tmp_path, caching: bool, num_blocks: int = 64):
    import torch
    from benchmarks.run_ours import OursEngine
    from tests.test_llama import hf_model, save_checkpoint, tiny_config
    from tinyserve_llm.engine.interface import CacheConfig
    from tinyserve_llm.engine.llm_engine import LLM
    from tinyserve_llm.scheduler.scheduler import SchedulerConfig

    save_checkpoint(hf_model(tiny_config()), tmp_path)
    llm = LLM(str(tmp_path), SchedulerConfig(max_model_len=128, max_num_seqs=8, max_num_batched_tokens=136),
              CacheConfig(block_size=4, caching=caching, device="cpu"), dtype=torch.float32,
              num_blocks=num_blocks, seed=0, load_tokenizer=False)
    return OursEngine(llm, temperature=0.6)


@pytest.fixture
def eager(monkeypatch):
    from tinyserve_llm.model import llama
    monkeypatch.setattr(llama, "COMPILE", False)


def test_our_engine_prefills_only_new_tokens_after_the_first_wave(tmp_path, eager):
    trace = tiny_trace()
    records = replay(trace, ours_engine(tmp_path, caching=True))
    assert records[0].prefill_tokens == records[0].prompt_tokens  # nothing cached yet
    for r in records[1:]:
        new = sum(len(c.turns[r.wave - 1].user_ids) for c in trace)
        # history is served from cache except the tail of its last, partly filled block
        assert new <= r.prefill_tokens < r.prompt_tokens
        assert r.preemptions == 0


def test_with_caching_off_every_prompt_token_is_prefilled(tmp_path, eager):
    records = replay(tiny_trace(), ours_engine(tmp_path, caching=False))
    assert all(r.prefill_tokens == r.prompt_tokens for r in records)
    assert summarize(records)["prefill_fraction"] == 1.0


def test_run_ours_writes_a_record(tmp_path, eager):
    import subprocess
    import sys
    from benchmarks.multiturn import save_trace
    from tests.test_llama import hf_model, save_checkpoint, tiny_config

    model = tmp_path / "model"
    model.mkdir()
    save_checkpoint(hf_model(tiny_config()), model)
    trace_path = tmp_path / "trace.json"
    save_trace(str(trace_path), tiny_trace(), {"max_model_len": 120})
    out = tmp_path / "out.json"
    subprocess.run([sys.executable, "-m", "benchmarks.run_ours", "--model", str(model), "--trace", str(trace_path),
                    "--conversations", "3", "--rho", "1", "--device", "cpu", "--out", str(out)],
                   check=True, env={**os.environ, "TINYSERVE_COMPILE": "0"})
    result = json.loads(out.read_text())
    assert result["conversations"] == 3 and len(result["waves"]) == 3
    # the tiny demand is below one maximum-length request, so the floor sets capacity, not rho
    assert result["num_blocks"] == 4 and result["summary"]["output_tokens"] > 0
