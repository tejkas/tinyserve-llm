from __future__ import annotations

import torch

class KVStorage:
    """
    KV Cache Storage/DAO Layer
    """

    def __init__(
            self,
            num_blocks: int,
            num_kv_heads: int,
            head_dim: int,
            block_size: int = 256,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda" # CPU/GPU
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks, block_size, num_kv_heads, head_dim) # Contiguous pool
        self.k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=device)

    def copy_slots(self, src_block: int, dst_block: int, num_slots: int) -> None:
        """
        Used for copy-on-write and partial-block reuse.
        """
        assert num_slots <= self.block_size
        self.k_pool[dst_block, :num_slots].copy_(self.k_pool[src_block, :num_slots])
        self.v_pool[dst_block, :num_slots].copy_(self.v_pool[src_block, :num_slots])