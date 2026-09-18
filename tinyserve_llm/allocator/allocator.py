from __future__ import annotations

import logging
from collections import deque

import torch

from tinyserve_llm.allocator.block import PhysicalBlock

logger = logging.getLogger(__name__)

# BlockAllocator to create and free PhysicalBlocks
# Allocates memory directly in GPU HBM
class BlockAllocator:
    def __init__(
            self,
            num_blocks: int,
            block_size: int,
            num_heads: int,
            head_dim: int,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        # Allocate large contiguous block of num_blocks * block_size * num_heads * head_dim for k 
        self.k_pool = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        # Allocate large contiguous block of num_blocks * block_size * num_heads * head_dim for v
        self.v_pool = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

        # free block list, using a deque
        # Only need the block_ids (int), not the PhysicalBlock object since that is just metadata living on CPU
        self.free_blocks: deque[int] = deque(range(num_blocks))
        # map block_id -> PhysicalBlock
        self.blocks: dict[int, PhysicalBlock] = {}

    def allocate(self) -> PhysicalBlock:
        if not self.free_blocks:
            raise MemoryError("Out of Free Blocks!")
        
        block_id = self.free_blocks.popleft()
        block = PhysicalBlock(block_id=block_id, block_size=self.block_size)
        self.blocks[block_id] = block

        logger.debug("Allocated block %d (%d free)", block_id, len(self.free_blocks))
        return block

    def free(self, block: PhysicalBlock) -> None:
        block.decrement_ref()
        if block.is_free():
            # if this block is free, return it to the free list
            del self.blocks[block.block_id]
            self.free_blocks.append(block.block_id)
            logger.debug("Freed block %d (%d free)", block.block_id, len(self.free_blocks))
    
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
    
    def get_kv(self, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_pool[block_id], self.v_pool[block_id]

