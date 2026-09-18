from __future__ import annotations

from tinyserve_llm.allocator.block import BlockTable, PhysicalBlock
from tinyserve_llm.allocator.allocator import BlockAllocator

# Creates a deep copy of a BlockTable
def fork_sequence(
        src_table: BlockTable,
        allocator: BlockAllocator,
        new_sequence_id: int,
) -> BlockTable:
    new_table = BlockTable(
        sequence_id=new_sequence_id,
        block_size=src_table.block_size,
    )
    for block in src_table.blocks:
        block.increment_ref()
        new_table.append_block(block)

    return new_table

def cow_write(
        block: PhysicalBlock,
        allocator: BlockAllocator,
) -> PhysicalBlock:
    # Single ref count implies just 1 request is using this block, so we don't even need CoW here
    if block.ref_count == 1:
        return block
    
    new_block = allocator.allocate()
    new_block.num_filled = block.num_filled

    src_k, src_v = allocator.get_kv(block.block_id)
    dst_k, dst_v = allocator.get_kv(new_block.block_id)
    dst_k.copy_(src_k)
    dst_v.copy_(src_v)

    block.decrement_ref()
    return new_block