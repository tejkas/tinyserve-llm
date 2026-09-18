from __future__ import annotations

import pytest
import torch

from tinyserve_llm.allocator.allocator import BlockAllocator
from tinyserve_llm.allocator.block import BlockTable
from tinyserve_llm.allocator.cow import fork_sequence, cow_write

def make_allocator(num_blocks: int = 32) -> BlockAllocator:
    return BlockAllocator(
        num_blocks=num_blocks,
        num_kv_heads=4,
        block_size=16,
        head_dim=64,
        dtype=torch.float16,
        device="cpu" # Don't use GPU for logic tests
    )

def test_allocate_and_free():
    """Test basic allocate and free logic"""
    allocator = make_allocator(num_blocks=4)
    assert allocator.num_free_blocks() == 4

    b1 = allocator.allocate()
    b2 = allocator.allocate()
    assert allocator.num_free_blocks() == 2
    assert b1.block_id != b2.block_id
    assert b1.ref_count == 1

    allocator.free(b1)
    assert allocator.num_free_blocks() == 3
    assert b1.is_free()

def test_oom_raises():
    """Exhausting the pool raises cleanly rather than crashing."""
    allocator = make_allocator(num_blocks=2)

    allocator.allocate()
    allocator.allocate()
    assert allocator.num_free_blocks() == 0
    with pytest.raises(MemoryError):
        allocator.allocate()

def test_fork_shares_blocks():
    """Forking a sequence should result in their block tables also being shared (prefix sharing)"""
    allocator = make_allocator()
    table = BlockTable(sequence_id=0, block_size=16)

    b1 = allocator.allocate()
    b1.num_filled = 16
    b2 = allocator.allocate()
    b2.num_filled = 5
    table.append_block(b1)
    table.append_block(b2)

    forked = fork_sequence(table, allocator, new_sequence_id=1)

    assert forked.num_blocks() == 2
    assert forked.get_block_ids() == table.get_block_ids()
    assert b1.ref_count == 2
    assert b2.ref_count == 2

def test_cow_write_copies_on_shared():
    """When a shared sequence is written to, Copy-on-Write should be triggered"""
    allocator = make_allocator()
    table = BlockTable(sequence_id=0, block_size=16)

    block = allocator.allocate()
    block.num_filled = 10
    table.append_block(block)

    fork_sequence(table, allocator, new_sequence_id=1)
    assert block.ref_count == 2

    new_block = cow_write(block, allocator)

    assert new_block.block_id != block.block_id
    assert new_block.num_filled == 10
    assert block.ref_count == 1
    assert new_block.ref_count == 1

def test_cow_write_noop_on_unique():
    """If ref-count is 1, don't copy for no reason"""
    allocator = make_allocator()
    block = allocator.allocate()

    assert block.ref_count == 1
    same_block = cow_write(block, allocator)
    assert same_block is block