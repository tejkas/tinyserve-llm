"""
Turns one scheduled step into tensors, runs the model and samples.

Each step's batch is flat: every sequence's fed tokens back to back, decodes and prefills alike.
Block tables are re-read from the cache view every step, never kept: a commit can swap a block
for a duplicate page and a copy-on-write can replace a sequence's last block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from tinyserve_llm.attention.flash_backend import paged_attention_flash, store_kv
from tinyserve_llm.attention.torch_paged import paged_attention_torch
from tinyserve_llm.engine.interface import KVCacheView
from tinyserve_llm.engine.sampler import sample
from tinyserve_llm.scheduler.scheduler import ScheduledSeq

AttentionImpl = Literal["torch", "flash"]


@dataclass(slots=True)
class StepInputs:
    """One step, packed. Index lists stay on the host; tensors live on the model's device."""
    input_ids: torch.Tensor     # [num_tokens] int64
    positions: torch.Tensor     # [num_tokens] int64
    slot_mapping: torch.Tensor  # [num_tokens] int64, block_id * block_size + offset
    block_tables: torch.Tensor  # [batch, max_blocks] int32, padded with 0 past each sequence's blocks
    cu_seqlens_q: list[int]     # [batch + 1] where each sequence's fed tokens start in the flat batch
    context_lens: list[int]     # [batch] keys per sequence after this step's writes
    sample_rows: torch.Tensor   # [sum of num_samples] int64, the flat row whose logits each draw uses
    temperatures: torch.Tensor  # [sum of num_samples] float32
    # flash-attn's varlen layout: device copies of the offsets, plus the longest query and key runs
    cu_seqlens_q_dev: torch.Tensor  # [batch + 1] int32
    cu_seqlens_k_dev: torch.Tensor  # [batch + 1] int32
    max_seqlen_q: int
    max_seqlen_k: int


def prepare(batch: list[ScheduledSeq], view: KVCacheView, device: torch.device | str) -> StepInputs:
    """Pack a scheduled step. Must run after schedule() and before postprocess()."""
    block_size = view.block_size
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    tables: list[list[int]] = []
    cu_seqlens_q = [0]
    context_lens: list[int] = []
    sample_rows: list[int] = []
    temperatures: list[float] = []
    for seq, fed, num_samples in batch:
        table = list(view.block_table(seq.seq_id))
        end = seq.num_tokens
        start = end - len(fed)  # the cache already holds everything before the fed tokens
        input_ids += fed
        positions += range(start, end)
        slot_mapping += [table[p // block_size] * block_size + p % block_size for p in range(start, end)]
        tables.append(table)
        cu_seqlens_q.append(cu_seqlens_q[-1] + len(fed))
        context_lens.append(end)
        # Only the last fed token's logits are sampled; a group's first prefill draws n times from them.
        sample_rows += [cu_seqlens_q[-1] - 1] * num_samples
        temperatures += [seq.params.temperature] * num_samples
    width = max(len(t) for t in tables)
    padded = [t + [0] * (width - len(t)) for t in tables]

    def to(values, dtype):
        return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

    cu_seqlens_k = [0]
    for n in context_lens:
        cu_seqlens_k.append(cu_seqlens_k[-1] + n)
    return StepInputs(
        cu_seqlens_q_dev=to(cu_seqlens_q, torch.int32),
        cu_seqlens_k_dev=to(cu_seqlens_k, torch.int32),
        max_seqlen_q=max(b - a for a, b in zip(cu_seqlens_q, cu_seqlens_q[1:])),
        max_seqlen_k=max(context_lens),
        input_ids=to(input_ids, torch.int64),
        positions=to(positions, torch.int64),
        slot_mapping=to(slot_mapping, torch.int64),
        block_tables=to(padded, torch.int32),
        cu_seqlens_q=cu_seqlens_q,
        context_lens=context_lens,
        sample_rows=to(sample_rows, torch.int64),
        temperatures=to(temperatures, torch.float32),
    )


class PagedAttention:
    """The model's attention backend for one step: write this step's K/V, then attend over the cache."""

    def __init__(self, view: KVCacheView, inputs: StepInputs, impl: AttentionImpl = "torch") -> None:
        self.view = view
        self.inputs = inputs
        self.impl = impl

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_cache, v_cache = self.view.layer_cache(layer_idx)
        store_kv(k, v, k_cache, v_cache, self.inputs.slot_mapping)
        inputs = self.inputs
        if self.impl == "flash":
            return paged_attention_flash(q, k_cache, v_cache, inputs.block_tables, inputs.cu_seqlens_q_dev,
                                         inputs.cu_seqlens_k_dev, inputs.max_seqlen_q, inputs.max_seqlen_k)
        return paged_attention_torch(q, k_cache, v_cache, self.inputs.block_tables,
                                     self.inputs.cu_seqlens_q, self.inputs.context_lens)


class ModelRunner:
    def __init__(self, model: torch.nn.Module, view: KVCacheView, device: torch.device | str,
                 impl: AttentionImpl = "torch", generator: torch.Generator | None = None) -> None:
        self.model = model
        self.view = view
        self.device = device
        self.impl = impl
        self.generator = generator

    @torch.inference_mode()
    def run(self, batch: list[ScheduledSeq]) -> list[list[int]]:
        """One forward pass; returns each scheduled sequence's num_samples new tokens."""
        inputs = prepare(batch, self.view, self.device)
        attn = PagedAttention(self.view, inputs, self.impl)
        hidden = self.model(inputs.input_ids, inputs.positions, attn)
        # Project only the sampled rows: the vocabulary projection is the widest matmul in the model.
        logits = self.model.compute_logits(hidden[inputs.sample_rows])
        tokens = sample(logits, inputs.temperatures, self.generator).tolist()  # the step's one host sync
        out, i = [], 0
        for _, _, num_samples in batch:
            out.append(tokens[i: i + num_samples])
            i += num_samples
        return out
