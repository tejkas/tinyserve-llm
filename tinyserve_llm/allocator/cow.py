from __future__ import annotations

from tinyserve_llm.allocator.block import BlockTable, PhysicalBlock
from tinyserve_llm.allocator.allocator import BlockAllocator

# Branches a sequence without copying any KV: the new table points at the same
# physical blocks and every block gains a reference. Memory is only copied later,
# and only for a block that is actually written to -- see cow_write.
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

    # Release through the allocator rather than touching ref_count here, so the
    # rule for returning a block to the free list lives in exactly one place.
    allocator.free(block)
    return new_block