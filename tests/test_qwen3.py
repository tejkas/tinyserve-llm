"""
Our Qwen3 must produce the same logits as transformers' on tiny random configs, and the engine must
generate HF's greedy tokens with it. head_dim is decoupled from hidden_size / num_heads, as in the
real checkpoints. No weights are downloaded.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from tests.test_llama import our_logits, save_checkpoint
from tinyserve_llm.engine.interface import CacheConfig
from tinyserve_llm.engine.llm_engine import LLM
from tinyserve_llm.model import llama
from tinyserve_llm.model.loader import load_model
from tinyserve_llm.model.qwen3 import Qwen3ForCausalLM
from tinyserve_llm.scheduler.scheduler import SchedulerConfig
from tinyserve_llm.scheduler.sequence import SamplingParams


@pytest.fixture(autouse=True)
def eager_kernels(monkeypatch):
    monkeypatch.setattr(llama, "COMPILE", False)


def tiny_config(tie: bool = True) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # 4 heads x 32 = 128, wider than the 64-wide residual stream
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        attn_implementation="eager",
    )


def hf_model(config: Qwen3Config) -> HFQwen3ForCausalLM:
    torch.manual_seed(0)
    model = HFQwen3ForCausalLM(config).eval()
    # HF initialises norms to ones; randomise them so a missed norm weight (q_norm, k_norm) would show up.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model


def ours_from(hf: HFQwen3ForCausalLM, path) -> Qwen3ForCausalLM:
    save_checkpoint(hf, path)
    return load_model(str(path), dtype=torch.float32, device="cpu")


@torch.no_grad()
def hf_logits(model: HFQwen3ForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids[None]).logits[0]


@pytest.mark.parametrize("tie", [True, False])
def test_packed_sequences_match_hf_run_separately(tmp_path, tie):
    hf = hf_model(tiny_config(tie=tie))
    ours = ours_from(hf, tmp_path)
    assert isinstance(ours, Qwen3ForCausalLM)
    assert (ours.lm_head.weight is ours.model.embed_tokens.weight) == tie
    g = torch.Generator().manual_seed(2)
    a = torch.randint(0, 128, (37,), generator=g)
    b = torch.randint(0, 128, (11,), generator=g)
    expected = torch.cat([hf_logits(hf, a), hf_logits(hf, b)])
    torch.testing.assert_close(our_logits(ours, [a, b]), expected, rtol=1e-5, atol=1e-5)


def test_qk_norm_weights_are_loaded(tmp_path):
    hf = hf_model(tiny_config())
    ours = ours_from(hf, tmp_path)
    torch.testing.assert_close(ours.model.layers[1].self_attn.k_norm.weight,
                               hf.model.layers[1].self_attn.k_norm.weight, rtol=0, atol=0)


def test_compiled_kernels_match_eager(tmp_path, monkeypatch):
    """QK-norm sends a 3-D [tokens, heads, head_dim] tensor through the compiled RMSNorm."""
    ours = ours_from(hf_model(tiny_config()), tmp_path)
    seqs = [torch.arange(9) % 128, torch.arange(5) % 128]
    eager = our_logits(ours, seqs)
    monkeypatch.setattr(llama, "COMPILE", True)
    torch.testing.assert_close(our_logits(ours, seqs), eager, rtol=1e-5, atol=1e-5)


def test_engine_greedy_matches_hf_with_prefix_hits(tmp_path):
    hf = hf_model(tiny_config())
    save_checkpoint(hf, tmp_path)
    llm = LLM(str(tmp_path), SchedulerConfig(max_model_len=64, max_num_seqs=4, max_num_batched_tokens=72),
              CacheConfig(block_size=4, device="cpu"), dtype=torch.float32, num_blocks=48, load_tokenizer=False)
    params = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
    prefix = [5, 17, 3, 99, 42, 8, 61, 23, 7]
    prompts = [prefix + [11], [77, 2, 9]]
    second = prefix + [30, 31]  # admitted after the first wave committed its blocks
    results = llm.generate(prompts, params) + llm.generate([second], params)
    for prompt, result in zip(prompts + [second], results):
        out = hf.generate(torch.tensor([prompt]), max_new_tokens=10, min_new_tokens=10, do_sample=False, pad_token_id=0)
        assert result.token_ids[0] == out[0, len(prompt):].tolist()
    assert llm.scheduler.stats().cached_tokens >= 8  # the prefix's two full blocks


def test_loader_rejects_an_unknown_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    with pytest.raises(NotImplementedError, match="gpt2"):
        load_model(str(tmp_path), dtype=torch.float32, device="cpu")
