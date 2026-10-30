from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 1.0
    n: int = 1  # parallel samples of the same prompt
    ignore_eos: bool = False  # lets benchmarks fix output lengths


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class Sequence:
    """
    The scheduler's view of one sequence. The KV cache manager keeps its own state under the same
    seq_id, while the scheduler needs this to rank, feed and finish the sequence.
    """
    seq_id: int
    request_id: int  # shared by the n samples of one request
    prompt_token_ids: tuple[int, ...]
    params: SamplingParams
    arrival_time: float  # in seconds. Preempted sequences keep their original arrival times
    status: SequenceStatus = SequenceStatus.WAITING # initialize in waiting
    output_token_ids: list[int] = field(default_factory=list)
    admission_hit: float = 0.0  # cache hit at the last admission; ranks the sequence while it runs
    num_preemptions: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    finish_reason: Literal["eos", "length"] | None = None

    @property
    def pending_token(self) -> int:
        """
        The latest sampled token. It has no KV yet, the next decode step feeds it.
        """
        return self.output_token_ids[-1]

    @property
    def all_token_ids(self) -> list[int]:
        """
        Prompt plus outputs: what an admission or re-admission prefills.
        """
        return [*self.prompt_token_ids, *self.output_token_ids]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)