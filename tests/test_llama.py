"""
Our Llama must produce the same logits as transformers' reference implementation on tiny random
configs, including Llama 3 rope scaling and packed batches. No weights are downloaded.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, LlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tinyserve_llm.model import llama
from tinyserve_llm.model.attention import DenseCausalAttention
from tinyserve_llm.model.llama import LlamaForCausalLM, RotaryEmbedding, rope_settings
from tinyserve_llm.model.loader import load_llama, load_weights

# Small original window so the llama3 scaling touches low, medium and high frequencies alike.
ROPE = {
    "rope_type": "llama3",
    "rope_theta": 10000.0,
    "factor": 4.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 32,
}


@pytest.fixture(autouse=True)
def eager_kernels(monkeypatch):
    # torch.compile costs seconds per kernel on CPU; test_compiled_kernels_match_eager covers it.
    monkeypatch.setattr(llama, "COMPILE", False)


def tiny_config(tie: bool = True, rope: dict | None = ROPE) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie,
        rope_parameters=dict(rope) if rope else {"rope_type": "default", "rope_theta": 10000.0},
        attn_implementation="eager",
    )


def hf_model(config: LlamaConfig) -> HFLlamaForCausalLM:
    torch.manual_seed(0)
    model = HFLlamaForCausalLM(config).eval()
    # HF initialises norms to ones; randomise them so a missed norm weight would show up.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model


def save_checkpoint(model: HFLlamaForCausalLM, path) -> None:
    """Write the HF weights as two safetensors shards plus config.json, the layout on the Hub."""
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    names = sorted(state)
    half = len(names) // 2
    save_file({k: state[k] for k in names[:half]}, str(path / "model-00001-of-00002.safetensors"))
    save_file({k: state[k] for k in names[half:]}, str(path / "model-00002-of-00002.safetensors"))
    model.config.save_pretrained(str(path))


def ours_from(hf: HFLlamaForCausalLM, path) -> LlamaForCausalLM:
    save_checkpoint(hf, path)
    return load_llama(str(path), dtype=torch.float32, device="cpu")


@torch.no_grad()
def hf_logits(model: HFLlamaForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids[None]).logits[0]


@torch.no_grad()
def our_logits(model: LlamaForCausalLM, seqs: list[torch.Tensor]) -> torch.Tensor:
    input_ids = torch.cat(seqs)
    positions = torch.cat([torch.arange(len(s)) for s in seqs])
    cu_seqlens = [0]
    for s in seqs:
        cu_seqlens.append(cu_seqlens[-1] + len(s))
    hidden = model(input_ids, positions, DenseCausalAttention(cu_seqlens))
    return model.compute_logits(hidden)


def test_inv_freq_matches_hf_llama3_scaling():
    config = tiny_config()
    expected, attention_factor = ROPE_INIT_FUNCTIONS["llama3"](config)
    theta, scaling = rope_settings(config)
    ours = RotaryEmbedding(16, 128, theta, scaling).inv_freq
    unscaled = RotaryEmbedding(16, 128, theta, None).inv_freq
    torch.testing.assert_close(ours, expected, rtol=0, atol=0)
    assert attention_factor == 1.0
    # The config is chosen so scaling leaves some frequencies alone and changes others.
    changed = ~torch.isclose(ours, unscaled)
    assert changed.any() and not changed.all()


def test_rope_settings_reads_hub_style_config(tmp_path):
    # Llama 3.2 1B's config.json as published: rope_theta and rope_scaling at the top level.
    hub_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(hub_config))
    config = AutoConfig.from_pretrained(str(tmp_path))
    theta, scaling = rope_settings(config)
    assert theta == 500000.0 and scaling["factor"] == 32.0
    expected, _ = ROPE_INIT_FUNCTIONS["llama3"](config)
    torch.testing.assert_close(RotaryEmbedding(64, 16, theta, scaling).inv_freq, expected, rtol=0, atol=0)
    # The same dict, read as transformers 4 and raw JSON present it.
    legacy = type("Cfg", (), {"rope_theta": 500000.0, "rope_scaling": hub_config["rope_scaling"]})()
    assert rope_settings(legacy) == (theta, hub_config["rope_scaling"])


def test_single_sequence_matches_hf(tmp_path):
    hf = hf_model(tiny_config())
    ours = ours_from(hf, tmp_path)
    ids = torch.randint(0, 128, (50,), generator=torch.Generator().manual_seed(1))  # beyond the 32-token window
    torch.testing.assert_close(our_logits(ours, [ids]), hf_logits(hf, ids), rtol=1e-5, atol=1e-5)


def test_packed_sequences_match_hf_run_separately(tmp_path):
    hf = hf_model(tiny_config())
    ours = ours_from(hf, tmp_path)
    g = torch.Generator().manual_seed(2)
    a = torch.randint(0, 128, (37,), generator=g)
    b = torch.randint(0, 128, (11,), generator=g)
    expected = torch.cat([hf_logits(hf, a), hf_logits(hf, b)])
    torch.testing.assert_close(our_logits(ours, [a, b]), expected, rtol=1e-5, atol=1e-5)


def test_tied_embeddings_share_one_parameter(tmp_path):
    hf = hf_model(tiny_config(tie=True))
    ours = ours_from(hf, tmp_path)
    assert ours.lm_head.weight is ours.model.embed_tokens.weight
    ids = torch.arange(20) % 128
    torch.testing.assert_close(our_logits(ours, [ids]), hf_logits(hf, ids), rtol=1e-5, atol=1e-5)


def test_untied_embeddings_load_lm_head(tmp_path):
    hf = hf_model(tiny_config(tie=False))
    ours = ours_from(hf, tmp_path)
    assert ours.lm_head.weight is not ours.model.embed_tokens.weight
    torch.testing.assert_close(ours.lm_head.weight, hf.lm_head.weight, rtol=0, atol=0)
    ids = torch.arange(20) % 128
    torch.testing.assert_close(our_logits(ours, [ids]), hf_logits(hf, ids), rtol=1e-5, atol=1e-5)


def test_default_rope_matches_hf(tmp_path):
    hf = hf_model(tiny_config(rope=None))
    ours = ours_from(hf, tmp_path)
    ids = torch.arange(30) % 128
    torch.testing.assert_close(our_logits(ours, [ids]), hf_logits(hf, ids), rtol=1e-5, atol=1e-5)


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    hf = hf_model(tiny_config(tie=False))
    save_checkpoint(hf, tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="did not fill"):
        load_weights(LlamaForCausalLM(tiny_config(tie=False)), str(tmp_path))


def test_compiled_kernels_match_eager(tmp_path, monkeypatch):
    hf = hf_model(tiny_config())
    ours = ours_from(hf, tmp_path)
    seqs = [torch.arange(9) % 128, torch.arange(5) % 128]
    eager = our_logits(ours, seqs)
    monkeypatch.setattr(llama, "COMPILE", True)
    torch.testing.assert_close(our_logits(ours, seqs), eager, rtol=1e-5, atol=1e-5)
