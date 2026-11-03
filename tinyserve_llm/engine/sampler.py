"""Token sampling: greedy at temperature 0, otherwise a draw from softmax(logits / temperature)."""

from __future__ import annotations

import torch


def sample(
        logits: torch.Tensor,        # [rows, vocab]
        temperatures: torch.Tensor,  # [rows], float32; 0 means greedy
        generator: torch.Generator | None = None,
) -> torch.Tensor:                   # [rows], int64
    logits = logits.float()
    greedy = logits.argmax(dim=-1)
    # Dividing by 1 for greedy rows keeps them finite; their draw is discarded below.
    t = torch.where(temperatures > 0, temperatures, torch.ones_like(temperatures))
    probs = torch.softmax(logits / t.unsqueeze(1), dim=-1)
    # Gumbel-max in exponential form: argmax(p_i / E_i) with E_i ~ Exp(1) draws i with probability
    # p_i, without the host sync that torch.multinomial needs.
    # The clamp guards against an exact 0 draw, which would divide by zero.
    noise = torch.empty_like(probs).exponential_(1, generator=generator).clamp_min_(1e-10)
    drawn = (probs / noise).argmax(dim=-1)
    return torch.where(temperatures > 0, drawn, greedy)
