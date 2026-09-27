from __future__ import annotations

from dataclasses import dataclass, field

# Analogue of a page frame, but similar to what vLLM does with GPU HBM memory
# discrete unit of memory that holds KVs
@dataclass(slots=True)
class PhysicalBlock:
    block_id: int
    ref_count: int  # no default: blocks are built free (0) and take ownership in reset()
    num_filled: int = 0

    def is_free(self) -> bool:
        return self.ref_count == 0
    
    def increment_ref(self) -> None:
        if self.ref_count == 0:
            raise ValueError(f"Cannot increment ref on a free block {self.block_id}")
        self.ref_count += 1

    def decrement_ref(self) -> None:
        if self.ref_count <= 0:
            raise ValueError(f"Cannot decrement ref on a free block {self.block_id}")
        self.ref_count -= 1

    def reset(self) -> None:
        """Re-arm a recycled block.

        Called when a block is handed out, never when it is freed. A freed block's
        KV stays valid until it is reused, so a later request with the same prefix
        can still claim it: freeing marks a block evictable, not invalid. Moving
        this into free() would destroy cache entries at the earliest possible
        moment instead of the latest.
        """
        self.ref_count = 1
        self.num_filled = 0

# Per-sequence page table, maps position of a token in a given sequence -> physical block
@dataclass
class BlockTable:
    sequence_id: int
    block_size: int
    blocks: list[PhysicalBlock] = field(default_factory=list)

    def num_tokens(self) -> int:
        if not self.blocks:
            return 0
        full_blocks = max(0, len(self.blocks) - 1)
        # The last block may be not totally filled
        return full_blocks * self.block_size + self.blocks[-1].num_filled

    def num_blocks(self) -> int:
        return len(self.blocks)
    
    def append_block(self, block: PhysicalBlock) -> None:
        self.blocks.append(block)

    def last_block_num_filled(self) -> int:
        """Slots used in the last block, or 0 when the table is empty."""
        return self.blocks[-1].num_filled if self.blocks else 0

    # Whether the last block still has room for one more token.
    def can_append_token(self) -> bool:
        return bool(self.blocks) and self.last_block_num_filled() < self.block_size

    def get_block_ids(self) -> list[int]:
        return [b.block_id for b in self.blocks]