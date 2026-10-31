# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), MIT License,
# Copyright (c) 2025 Xingkai Yu. Based on nanovllm/utils/loader.py.

"""Load a Hugging Face Llama checkpoint directory (config.json + model*.safetensors)."""

from __future__ import annotations

import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from tinyserve_llm.model.llama import LlamaForCausalLM, MergedLinear

# Buffers some older checkpoints serialise; we recompute them.
IGNORED_SUFFIXES = ("rotary_emb.inv_freq",)


def load_weights(model: nn.Module, path: str) -> None:
    """
    Copy every tensor in path/model*.safetensors into `model`, splitting HF's separate q/k/v and
    gate/up projections into our fused ones. Raises if any of our parameters was not filled.
    """
    files = sorted(glob(os.path.join(path, "model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {path}")
    packed = getattr(model, "packed_modules_mapping", {})
    tied = getattr(model, "tie_word_embeddings", False)
    loads: dict[str, int] = {}  # our param name -> tensors copied into it

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                if name.endswith(IGNORED_SUFFIXES):
                    continue
                # Tied checkpoints normally omit lm_head; if one is present it duplicates the embedding.
                if tied and name == "lm_head.weight":
                    continue
                tensor = f.get_tensor(name)
                for hf_part, (our_part, shard_id) in packed.items():
                    if f".{hf_part}." in name:
                        param_name = name.replace(f".{hf_part}.", f".{our_part}.")
                        module = model.get_submodule(param_name.rpartition(".")[0])
                        assert isinstance(module, MergedLinear)
                        module.load_shard(shard_id, tensor)
                        break
                else:
                    param_name = name
                    param = model.get_parameter(name)
                    assert param.shape == tensor.shape, (name, param.shape, tensor.shape)
                    param.data.copy_(tensor)
                loads[param_name] = loads.get(param_name, 0) + 1

    missing = []
    for name, _ in model.named_parameters():  # tied weights appear once, under the embedding
        module = model.get_submodule(name.rpartition(".")[0])
        needed = len(module.output_sizes) if isinstance(module, MergedLinear) else 1
        if loads.get(name, 0) != needed:
            missing.append(name)
    if missing:
        raise ValueError(f"checkpoint did not fill: {missing}")


def load_llama(
        path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
) -> LlamaForCausalLM:
    """
    Build the model directly in `dtype` on `device` and load its weights. Building under the
    default dtype (not .to(dtype) afterwards) keeps the RoPE table in fp32.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(previous)
    load_weights(model, path)
    return model.eval()
