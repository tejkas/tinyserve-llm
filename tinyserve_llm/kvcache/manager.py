from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Sequence
from typing import TYPE_CHECKING

from tinyserve_llm.kvcache.page_tree import Match, PageNode, PageTree

if TYPE_CHECKING:
    from tinyserve_llm.kvcache.storage import KVStorage

@dataclass(slots=True)
class SeqState:
    """
    Per-sequence state tracking for the KVCacheManager.
    """
    block_table: list[int] = field(default_factory=list) # block_ids in token order
    token_ids: list[int] = field(default_factory=list) # one per reserved slot
    committed: int = 0 # pointer tracking leading slots whose KV is written and accepted

    @property
    def reserved(self) -> int:
        return len(self.token_ids)

@dataclass(frozen=True, slots=True)
class AdmissionCost:
    blocks_needed: int # can allocate if this is <= num_available_blocks()
    prefill_tokens: int # prompt tokens the forward pass needs to compute

@dataclass(slots=True)
class CacheStats:
    """
    Running totals since the manager was created; diff two snapshots for one step's counts.
    """
    evictions: int = 0
    cow_copies: int = 0      # shared last block copied before a write
    partial_copies: int = 0  # matched slots copied out of a partially matching page at admission
    publishes: int = 0       # new pages added to the tree
    dedupes: int = 0         # blocks swapped for an identical page already in the tree

    # Current values, not totals; filled in by stats()
    blocks_in_use: int = 0
    cached_pages: int = 0

class KVCacheManager:
    """
    Managed KV Cache metadata, including:
    - block ref counts
    - prefix cache (page tree + internal tries)
    - free, uncached blocks
    - per-sequence state
    """

    def __init__(self, storage: KVStorage, caching: bool = True) -> None:
        self.storage = storage
        self.block_size = storage.block_size
        self.tree = PageTree(self.block_size)
        self.seqs: dict[int, SeqState] = {}
        self.num_blocks = storage.num_blocks
        self.ref_counts = [0] * self.num_blocks
        self.free_uncached_blocks = list(range(self.num_blocks - 1, -1, -1)) # reverse-order, pop() vends block 0 first
        self.num_free_cached = 0
        self._clock = 0 # Tracks LRU with monotonically increasing global counter
        self.caching = caching # default True, but refuses trie publshing if False
        self.counters = CacheStats()

    def num_available_blocks(self) -> int:
        """
        Total number of blocks available for a new reservation (all free, uncached blocks + evictable free cached blocks)
        """
        return len(self.free_uncached_blocks) + self.num_free_cached

    def _take_block(self) -> int:
        """
        Vends a free block with ref 1. Prefers an uncached one if available, else evicts the LRU free cached leaf block in the page tree.
        """
        if self.free_uncached_blocks:
            block_id = self.free_uncached_blocks.pop()
        else:
            block_id = self._evict_lru_leaf()
        assert self.ref_counts[block_id] == 0
        self.ref_counts[block_id] = 1
        return block_id

    def _evict_lru_leaf(self) -> int:
        """
        Evicts a free, cached block from the prefix cache tree. The page holding the block must be a leaf node and must be free.
        Returns the block_id of the evicted node.
        """
        victim = min(
            (n for n in self.tree.nodes.values() if self.ref_counts[n.block_id] == 0 and not n.children),
            key=lambda n: n.last_used,
            default=None
        )
        assert victim is not None, "no free blocks. Caller must check num_available_blocks() first."
        self.num_free_cached -= 1
        self.counters.evictions += 1
        return self.tree.evict(victim)

    def _increment_ref(self, block_id: int) -> None:
        """
        Adds a holder to a block already in use, or cached.
        """
        if self.ref_counts[block_id] == 0:
            # check that it is a cached block
            assert block_id in self.tree.nodes, "free uncached blocks should be claimed with _take_block() first"
            self.num_free_cached -= 1
        self.ref_counts[block_id] += 1

    def _decrement_ref(self, block_id: int) -> None:
        """
        Drops a holder from a block. At ref_count == 0, the block is considered free.
        """
        assert self.ref_counts[block_id] > 0
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] > 0:
            return # in use
        node = self.tree.nodes.get(block_id)
        if node is None:
            self.free_uncached_blocks.append(block_id) # free
        else: 
            # free but still cached, update LRU
            self._clock += 1
            node.last_used = self._clock
            self.num_free_cached += 1

    def calculate_admission_cost(self, token_ids: Sequence[int]) -> AdmissionCost:
        """
        Calculates the cost of admitting this sequence. Read-only.
        """
        match = self.tree.match(token_ids=token_ids)
        return self._cost(match, self._select_partial_source(match), len(token_ids))

    def allocate(self, seq_id: int, token_ids: Sequence[int]) -> Sequence[int] | None:
        """
        Admit a sequence. Attaches its cached prefix, copies a partial match, and reserves slots for the rest of the 
        prompt. 
        Returns the sequence's block table, or None if there are not enough free blocks, in which case nothing changes.
        """
        assert token_ids, "need at least one token to start a prompt"
        assert seq_id not in self.seqs, "sequence is already allocated"
        match = self.tree.match(token_ids=token_ids)
        source = self._select_partial_source(match)
        if self._cost(match, source, len(token_ids)).blocks_needed > self.num_available_blocks():
            return None

        seq = SeqState()
        # Hold all matched pages by incrementing refcounts before calling _take_block() to prevent accidentally evicting them
        for page in match.pages:
            self._increment_ref(page.block_id)
            seq.block_table.append(page.block_id)
        if source is not None:
            self._increment_ref(source.block_id) # same as above - hold before take
            dest = self._take_block()
            self.storage.copy_slots(source.block_id, dest, match.partial_len)
            self.counters.partial_copies += 1
            self._decrement_ref(source.block_id) # copy_slots allows release the moment it returns
            seq.block_table.append(dest)
        seq.committed = match.num_tokens
        seq.token_ids = list(token_ids[:seq.committed])
        self._reserve(seq, token_ids[seq.committed:])
        self.seqs[seq_id] = seq
        return seq.block_table

    def num_cached_tokens(self, seq_id: int) -> int:
        """
        The number of leading tokens whose KV already exists. Tells prefill where to start computing.
        """
        return self.seqs[seq_id].committed

    def _select_partial_source(self, match: Match) -> PageNode | None:
        """
        Returns the page from which a partial match can be copied, or None if there is no partial match. 
        Every candidate in `match` holds the matched tokens, but one that is already in use is preferred.
        """
        if not match.partial_len:
            return None # No partial matches
        return min(match.partial_candidates, key=lambda p: self.ref_counts[p.block_id] == 0) # False < True, in-use blocks are preferred

    def _cost(self, match: Match, source: PageNode | None, num_tokens: int) -> AdmissionCost:
        """
        Checks what allocating this new sequence would take from the free block pool. 
        """
        refs = self.ref_counts
        pages = match.pages
        # calculates the incremental blocks needed beyond what is already matched in pages
        blocks_to_take = -(-num_tokens // self.block_size) - len(pages)
        # Partial page match, but over all recorded tokens in the last block 
        # If the partial match is already held, we will need to copy-on-write, so an additional block is demanded.
        # Eg Prompt [A, B, C, D, E, F] and blocks are [A, B, C, D] and [E, F, _, _]
        if pages and len(pages[-1].tokens) < self.block_size and refs[pages[-1].block_id] > 0:
            blocks_to_take += 1
        # The case when a partial match ends midway through a block
        # Eg [A, B, C, D] matching with block [A, B, X, Y] -> need to copy slots A, B somewhere
        # Need at least 2 (the source block and the destination to copy slots into)
        # Check if the source is free - if someone else holds the source, it isn't in the free pool and can't be evicted, so nothing extra is needed.
        if source is not None and refs[source.block_id] == 0:
            blocks_to_take = max(blocks_to_take, 2)
        # If a free matched page is attached, it will be taken out of the pool
        free_pages_attached = sum(1 for p in pages if refs[p.block_id] == 0)
        # Final count is total blocks we plan on taking + free pages that are matched and attached
        return AdmissionCost(blocks_to_take + free_pages_attached, num_tokens - match.num_tokens)

    def _reserve(self, seq: SeqState, token_ids: Sequence[int]) -> None:
        """
        Extend the block table for this sequence to cover `token_ids` after the reserved slots and record them.
        """
        block_table = seq.block_table
        last_block_used_slots = seq.reserved % self.block_size # number of slots filled in the last block
        if last_block_used_slots:
            last_block = block_table[-1]
            if self.ref_counts[last_block] > 1: # sharing case -> copy on write
                new_block = self._take_block()
                self.storage.copy_slots(last_block, new_block, last_block_used_slots)
                self.counters.cow_copies += 1
                self._decrement_ref(last_block)
                block_table[-1] = new_block
            else: # our block, write in place
                node = self.tree.nodes.get(last_block)
                assert node is None or last_block_used_slots >= len(node.tokens), "cannot overwrite a published slot"
        needed_blocks = -(-(seq.reserved + len(token_ids)) // self.block_size) # ceil division
        while len(block_table) < needed_blocks:
            block_table.append(self._take_block())
        seq.token_ids.extend(token_ids)

    def blocks_needed_for_append(self, seq_id: int, num_tokens: int) -> int:
        """
        Blocks needed for appending `num_tokens` to this sequence.
        When this value is <= num_available_blocks(), allocate_slots succeeds.
        """
        seq = self.seqs[seq_id]
        table = seq.block_table
        blocks = -(-(seq.reserved + num_tokens) // self.block_size) - len(table)
        if seq.reserved % self.block_size and self.ref_counts[table[-1]] > 1:
            blocks += 1 # If this sequence is sharing its last block, then we need an additional block to copy it before writing
        return blocks

    def append_slots(self, seq_id: int, token_ids: Sequence[int]) -> None:
        """
        Reserves slots needed for the tokens fed into this sequence for this step
        (for decode: 1; for speculative decoding: >1)
        Invoked once per step, per sequence, and needs to be finalized with commit().
        """
        assert token_ids, "a step needs at least one token"
        seq = self.seqs[seq_id]
        assert seq.reserved == seq.committed, "the previous step was not successfully committed"
        needed = self.blocks_needed_for_append(seq_id, len(token_ids))
        assert needed <= self.num_available_blocks(), "not enough free blocks; preemption needed"
        self._reserve(seq, token_ids)

    def commit(self, seq_id: int, num_accepted: int) -> None:
        """
        Closes a step by keeping the KV of the first `num_accepted` slots reserved since the last commit. Drops the rest.
        Used for speculative decoding (to drop tokens rejected by the verifier).
        Finally, publishes every block that is full.
        """
        seq = self.seqs[seq_id]
        assert 0 <= num_accepted <= seq.reserved - seq.committed, "cannot accept more slots than those reserved"
        old_committed = seq.committed
        seq.committed += num_accepted
        # Rollback rejected tokens
        del seq.token_ids[seq.committed:]
        blocks_kept = -(-seq.committed // self.block_size)
        while len(seq.block_table) > blocks_kept:
            self._decrement_ref(seq.block_table.pop()) # release holds on blocks that are no longer needed
        if self.caching:
            for i in range(old_committed // self.block_size, seq.committed // self.block_size):
                self._publish_block(seq, i) # Publish new full blocks

    def _publish_block(self, seq: SeqState, i: int):
        """
        Publish block i's committed tokens in the tree, under its parent page (aka block i-1 or root).
        If the tree already contains a page with these tokens, dedupe by releasing this block and switching to that one.
        """
        block_table = seq.block_table
        block_id = block_table[i]
        parent = self.tree.nodes[block_table[i-1]] if i else self.tree.root
        tokens = seq.token_ids[i * self.block_size : min((i + 1) * self.block_size, seq.committed)] # grab tokens from the start of the block until commit point/end of block
        node = self.tree.nodes.get(block_id)
        if node is None:
            page = self.tree.publish(parent, block_id, tokens)
            if page.block_id == block_id:
                self.counters.publishes += 1
        elif len(node.tokens) < len(tokens):
            # Attached partial page that we are writing into
            assert node.parent is parent
            page = self.tree.grow(node, tokens)
        else:
            return # if this block already exists and has the same tokens, nothing to do

        if page.block_id != block_id: # If there is an existing page that dupes this block, dedupe
            self.counters.dedupes += 1
            self._increment_ref(page.block_id)
            block_table[i] = page.block_id
            self._decrement_ref(block_id)

    def release(self, seq_id: int) -> None:
        """
        Drop a finished or preempted sequence.
        We still store its KV in the page tree for future matching.
        """
        seq = self.seqs[seq_id]
        assert seq.reserved == seq.committed, "release between steps: commit first"
        if self.caching and seq.committed % self.block_size:
            self._publish_block(seq, seq.committed // self.block_size) # publish the tail block to store the full KV in the page tree for matching
        del self.seqs[seq_id]
        for block_id in reversed(seq.block_table): # go in order of tail first
            self._decrement_ref(block_id)

    def fork(self, src_id: int, new_id: int) -> None:
        """
        Start `new_id` as a copy of `src_id`. Used for parallel sampling.
        """
        assert new_id not in self.seqs, "sequence is already allocated"
        src = self.seqs[src_id]
        assert src.reserved == src.committed, "fork between steps: commit first"
        for block_id in src.block_table:
            self._increment_ref(block_id)
        self.seqs[new_id] = SeqState(
            block_table=list(src.block_table), token_ids=list(src.token_ids), committed=src.committed
        )

    def stats(self) -> CacheStats:
        """
        A snapshot: the running totals plus current block usage.
        """
        return replace(self.counters, blocks_in_use=self.num_blocks - self.num_available_blocks(),
                       cached_pages=len(self.tree.nodes))
