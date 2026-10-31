from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tinyserve_llm.kvcache.manager import CacheStats, KVCacheManager
from tinyserve_llm.kvcache.storage import KVStorage
from tinyserve_llm.scheduler.scheduler import Scheduler, SchedulerConfig

@dataclass(frozen=True, slots=True)
class ModelShape:
    """
    Model dimensions to size the KV cache.
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_hf_config(cls, config, dtype: torch.dtype = torch.bfloat16) -> ModelShape:
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(config.num_hidden_layers, config.num_key_value_heads, head_dim, dtype)

@dataclass(frozen=True, slots=True)
class CacheConfig:
    block_size: int = 256
    caching: bool = True  # False turns prefix caching off, but forks still share blocks
    device: str = "cuda"

class KVCacheView(ABC):
    """
    Defines what the engine needs to read from the KV cache.
    """
    block_size: int

    @abstractmethod
    def block_table(self, seq_id: int) -> Sequence[int]: ...

    @abstractmethod
    def layer_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def stats(self) -> CacheStats: ...

class PagedKVCacheView(KVCacheView):
    """
    Defines the interface over the GPU-residing paged cache. 
    block_table() is only valid from schedule() to postprocess().
    """

    def __init__(self, manager: KVCacheManager) -> None:
        self._manager = manager
        self.block_size = manager.block_size

    def block_table(self, seq_id: int) -> Sequence[int]:
        return self._manager.seqs[seq_id].block_table

    def layer_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._manager.storage.layer(layer)

    def stats(self) -> CacheStats:
        return self._manager.stats()

def bytes_per_block(shape: ModelShape, block_size: int) -> int:
    """
    Memory for one block across all layers.
    """
    return KVStorage.bytes_per_block(shape.num_layers, shape.num_kv_heads, shape.head_dim, block_size, shape.dtype)

def build(num_blocks: int, shape: ModelShape, scheduler_config: SchedulerConfig,
          cache_config: CacheConfig) -> tuple[Scheduler, KVCacheView]:
    """
    Build's the components required for the engine, namely:
    - Scheduler (interfaces with the KV Cache)
    - KVCacheView (read-only)
    """
    storage = KVStorage(num_blocks, shape.num_layers, shape.num_kv_heads, shape.head_dim,
                        cache_config.block_size, shape.dtype, cache_config.device)
    manager = KVCacheManager(storage, caching=cache_config.caching)
    return Scheduler(scheduler_config, manager), PagedKVCacheView(manager)