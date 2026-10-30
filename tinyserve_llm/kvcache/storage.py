from __future__ import annotations

import torch

class KVStorage:
    """
    KV Cache Storage/DAO Layer for all model layers in block-first order
    [num_blocks, num_layers, block_size, kv_heads, head_dim].
    One block's KV for all layers is contiguous so copying/offloading is just 1 copy per pool.
    """

    def __init__(
            self,
            num_blocks: int,
            num_layers: int,
            num_kv_heads: int,
            head_dim: int,
            block_size: int = 256,
            dtype: torch.dtype = torch.bfloat16,
            device: str = "cuda" # CPU/GPU
    ) -> None:
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.block_size = block_size
        shape = (num_blocks, num_layers, block_size, num_kv_heads, head_dim) # Contiguous pool
        self.k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def bytes_per_block(num_layers: int, num_kv_heads: int, head_dim: int, block_size: int, dtype: torch.dtype) -> int:
        """
        K and V for one block across all layers, used for sizing the pool after the warm-up run.
        """
        return 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize

    def layer(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One layer's KV cache, [num_blocks, block_size, kv_heads, head_dim].
        Returns a strided view of the pool for this layer.
        """
        return self.k_pool[:, layer], self.v_pool[:, layer]
    
    def copy_slots(self, src_block: int, dst_block: int, num_slots: int) -> None:
        """
        Used for copy-on-write and partial-block reuse.
        The caller may release the source block as soon as this returns.
        """
        assert num_slots <= self.block_size
        self.k_pool[dst_block, :, :num_slots].copy_(self.k_pool[src_block, :, :num_slots])
        self.v_pool[dst_block, :, :num_slots].copy_(self.v_pool[src_block, :, :num_slots])