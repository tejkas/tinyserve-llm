from __future__ import annotations

import random

import pytest
import torch

from tinyserve_llm.kvcache.manager import KVCacheManager
from tinyserve_llm.kvcache.storage import KVStorage

B = 4


class FakeKV:
    """Storage stand-in: each slot holds the token id whose KV it stores (None = never written)."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks, self.block_size = num_blocks, B
        self.slots = [[None] * B for _ in range(num_blocks)]
        self.copies: list[tuple[int, int, int]] = []

    def copy_slots(self, src: int, dst: int, num_slots: int) -> None:
        self.copies.append((src, dst, num_slots))
        self.slots[dst][:num_slots] = self.slots[src][:num_slots]


def make(num_blocks: int = 12) -> KVCacheManager:
    return KVCacheManager(FakeKV(num_blocks))


def forward(m: KVCacheManager, seq_id: int) -> None:
    """Fake forward pass: write KV for the slots reserved this step."""
    s = m.seqs[seq_id]
    for pos in range(s.committed, s.reserved):
        m.storage.slots[s.block_table[pos // B]][pos % B] = s.token_ids[pos]


def prefill(m: KVCacheManager, seq_id: int, prompt: list[int]) -> None:
    assert m.allocate(seq_id, prompt) is not None
    s = m.seqs[seq_id]
    forward(m, seq_id)
    m.commit(seq_id, s.reserved - s.committed)


def step(m: KVCacheManager, seq_id: int, tokens: list[int], accept: int | None = None) -> None:
    m.append_slots(seq_id, tokens)
    forward(m, seq_id)
    m.commit(seq_id, len(tokens) if accept is None else accept)


def check_invariants(m: KVCacheManager) -> None:
    """Every invariant the manager promises between steps."""
    tree = m.tree
    holders = [0] * m.num_blocks
    for s in m.seqs.values():
        assert s.reserved == s.committed
        assert len(s.block_table) == -(-s.committed // B)
        for pos in range(s.committed):  # our slots hold our tokens' KV
            assert m.storage.slots[s.block_table[pos // B]][pos % B] == s.token_ids[pos]
        for i in range(s.committed // B):  # every full committed block is in the tree, chained
            node = tree.nodes[s.block_table[i]]
            assert node.tokens == tuple(s.token_ids[i * B : (i + 1) * B])
            assert node.parent is (tree.nodes[s.block_table[i - 1]] if i else tree.root)
        for b in s.block_table:
            holders[b] += 1
    assert holders == m.ref_counts

    free = {b for b in range(m.num_blocks) if m.ref_counts[b] == 0}
    assert sorted(m.free_uncached_blocks) == sorted(b for b in free if b not in tree.nodes)
    assert m.num_free_cached == sum(1 for b in free if b in tree.nodes)

    for b, node in tree.nodes.items():
        assert node.block_id == b
        assert m.storage.slots[b][: len(node.tokens)] == list(node.tokens)
        assert node.parent.children[node.tokens] is node
        if len(node.tokens) < B:
            assert not node.children, "a partial page is always a leaf"
        if m.ref_counts[b] and node.parent is not tree.root:
            assert m.ref_counts[node.parent.block_id], "a held page's parent is held"
        depth, pages = node.parent.trie.longest_match(node.tokens)
        assert depth == len(node.tokens) and node in pages


# --- admission -------------------------------------------------------------------------------


def test_cold_start_takes_one_block_per_four_tokens():
    m = make()
    cost = m.calculate_admission_cost(list(range(10)))
    assert (cost.blocks_needed, cost.prefill_tokens) == (3, 10)
    assert list(m.allocate(1, list(range(10)))) == [0, 1, 2]
    assert m.num_cached_tokens(1) == 0


def test_a_fully_cached_prompt_still_prefills_its_last_token():
    m = make()
    prefill(m, 1, list(range(8)))
    m.release(1)
    cost = m.calculate_admission_cost(list(range(8)))
    assert cost.prefill_tokens == 1
    prefill(m, 2, list(range(8)))
    assert m.num_cached_tokens(2) == 8  # committed after prefill
    check_invariants(m)


def test_free_attached_partial_page_is_written_in_place():
    m = make()
    prefill(m, 1, list(range(10)))
    m.release(1)  # pages [0..3] [4..7] [8,9]
    prompt = list(range(10)) + [10]
    cost = m.calculate_admission_cost(prompt)
    assert (cost.blocks_needed, cost.prefill_tokens) == (3, 1)  # 3 free pages held, none new
    assert list(m.allocate(2, prompt)) == [0, 1, 2]
    assert m.num_cached_tokens(2) == 10 and not m.storage.copies


def test_shared_attached_partial_page_is_copied_in_allocate():
    m = make()
    prefill(m, 1, list(range(10)))
    m.release(1)
    prompt = list(range(10)) + [10]
    m.allocate(2, prompt)
    cost = m.calculate_admission_cost(prompt)
    assert (cost.blocks_needed, cost.prefill_tokens) == (1, 1)
    table = list(m.allocate(3, prompt))
    assert table[:2] == [0, 1] and table[2] != 2
    assert m.storage.copies[-1] == (2, table[2], 2) and m.ref_counts[2] == 1


def test_partial_match_copies_only_the_matched_slots():
    m = make()
    prefill(m, 1, list(range(10)))
    m.release(1)
    m.allocate(2, list(range(10)) + [10])  # holds the [8,9] page
    prompt = list(range(8)) + [8, 99, 5]
    cost = m.calculate_admission_cost(prompt)
    assert (cost.blocks_needed, cost.prefill_tokens) == (1, 2)  # source is held: nothing extra
    table = list(m.allocate(4, prompt))
    assert m.storage.copies[-1] == (2, table[2], 1) and m.num_cached_tokens(4) == 9


def test_free_copy_source_is_released_once_its_destination_is_taken():
    m = make(2)
    prefill(m, 1, [1, 2, 3])
    m.release(1)  # page [1,2,3] at block 0, free
    prompt = [1, 2, 9, 9, 9, 9]
    cost = m.calculate_admission_cost(prompt)
    assert (cost.blocks_needed, cost.prefill_tokens) == (2, 4)
    assert list(m.allocate(2, prompt)) == [1, 0]  # copy into 1, then evict the source for block 2
    assert m.storage.copies[-1] == (0, 1, 2) and 0 not in m.tree.nodes


def test_a_free_copy_source_cannot_be_its_own_destination():
    m = make(2)
    prefill(m, 1, [1, 2, 3])
    m.release(1)
    m.allocate(2, [5, 5])  # takes the only uncached block
    assert m.calculate_admission_cost([1, 2, 9]).blocks_needed == 2 > m.num_available_blocks()
    assert m.allocate(3, [1, 2, 9]) is None


def test_allocate_without_room_returns_none_and_changes_nothing():
    m = make(3)
    prefill(m, 1, list(range(10)))
    m.release(1)
    m.allocate(2, [50, 51, 52, 53, 54])

    def snapshot():
        return (list(m.ref_counts), list(m.free_uncached_blocks), m.num_free_cached, m._clock,
                {b: (n.tokens, n.last_used) for b, n in m.tree.nodes.items()}, set(m.seqs))

    before = snapshot()
    assert m.allocate(9, list(range(10)) + [10]) is None
    assert snapshot() == before


def test_matched_pages_are_held_before_any_block_is_taken():
    m = make(3)
    prefill(m, 1, [1, 2, 3, 4, 5])
    m.release(1)  # pages [1..4] (oldest stamp), [5]
    prefill(m, 2, [7, 7])
    m.release(2)  # page [7,7]
    table = list(m.allocate(3, [1, 2, 3, 4, 6, 6, 6]))  # needs one block: must not evict [1..4]
    assert table[0] == 0 and 0 in m.tree.nodes
    forward(m, 3)
    m.commit(3, 3)
    check_invariants(m)


def test_allocate_twice_asserts():
    m = make()
    m.allocate(1, [1, 2])
    with pytest.raises(AssertionError):
        m.allocate(1, [3, 4])


# --- free pool and eviction ------------------------------------------------------------------


def test_uncached_blocks_go_first_then_the_lru_free_leaf():
    m = make(4)
    b0, b1, b2, b3 = (m._take_block() for _ in range(4))
    p0 = m.tree.publish(m.tree.root, b0, [1, 2, 3, 4])
    p1 = m.tree.publish(p0, b1, [5, 6])
    p2 = m.tree.publish(m.tree.root, b2, [9, 9, 9, 9])
    for b in (b3, b2, b0, b1):
        m._decrement_ref(b)
    assert m.free_uncached_blocks == [3] and m.num_free_cached == 3
    assert (p2.last_used, p0.last_used, p1.last_used) == (1, 2, 3)  # stamped at ref 0
    assert m._take_block() == 3
    assert m._take_block() == 2  # p0 is older but not a leaf
    assert 2 not in m.tree.nodes and m.num_free_cached == 2


def test_increment_ref_on_a_free_uncached_block_asserts():
    m = make(2)
    b = m._take_block()
    m._decrement_ref(b)
    with pytest.raises(AssertionError):
        m._increment_ref(b)


# --- append ----------------------------------------------------------------------------------


def test_append_writes_in_place_then_takes_a_block_at_the_boundary():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5, 6])
    assert m.blocks_needed_for_append(1, 1) == 0
    step(m, 1, [7])
    step(m, 1, [8])
    assert m.seqs[1].block_table == [0, 1]
    assert m.blocks_needed_for_append(1, 1) == 1
    step(m, 1, [9])
    assert m.seqs[1].block_table == [0, 1, 2]
    check_invariants(m)


def test_append_without_commit_asserts():
    m = make()
    prefill(m, 1, [1, 2])
    m.append_slots(1, [3])
    with pytest.raises(AssertionError):
        m.append_slots(1, [4])


def test_append_without_room_asserts_and_changes_nothing():
    m = make(1)
    prefill(m, 1, [1, 2, 3, 4])
    assert m.blocks_needed_for_append(1, 1) == 1 > m.num_available_blocks()
    with pytest.raises(AssertionError):
        m.append_slots(1, [5])
    assert m.seqs[1].reserved == 4


def test_speculative_append_takes_exactly_what_it_predicts():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5, 6])
    needed = m.blocks_needed_for_append(1, 7)
    before = m.num_available_blocks()
    m.append_slots(1, list(range(10, 17)))
    assert needed == 2 and before - m.num_available_blocks() == 2


def test_fork_group_checked_then_appended_one_by_one_is_exact():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5])
    for c in (2, 3, 4):
        m.fork(1, c)
    needs = []
    for sid in (2, 3, 4, 1):
        needs.append(m.blocks_needed_for_append(sid, 1))
        step(m, sid, [10 * sid])
    assert needs == [1, 1, 1, 0]
    assert m.seqs[1].block_table == [0, 1] and m.ref_counts[:2] == [4, 1]
    check_invariants(m)


# --- commit ----------------------------------------------------------------------------------


def test_prefill_commit_publishes_only_full_blocks():
    m = make()
    prefill(m, 1, list(range(10)))
    assert set(m.tree.nodes) == {0, 1}
    step(m, 1, [10])
    step(m, 1, [11])
    assert m.tree.nodes[2].parent is m.tree.nodes[1]
    check_invariants(m)


def test_rejected_drafts_are_rolled_back():
    m = make()
    prefill(m, 1, list(range(10)))
    before = m.num_available_blocks()
    m.append_slots(1, list(range(10, 16)))  # spans two new blocks
    forward(m, 1)
    m.commit(1, 2)  # keep 2; the second new block held only rejected tokens
    s = m.seqs[1]
    assert s.committed == s.reserved == 12 and s.token_ids[-2:] == [10, 11]
    assert len(s.block_table) == 3 and m.num_available_blocks() == before
    check_invariants(m)
    step(m, 1, [99, 98])  # reuses the freed slots; check_invariants reads them back
    assert len(s.block_table) == 4 and s.token_ids[-2:] == [99, 98]
    check_invariants(m)


def test_commit_zero_abandons_the_step_and_over_accept_asserts():
    m = make()
    prefill(m, 1, [1, 2, 3])
    m.append_slots(1, [4, 5])
    with pytest.raises(AssertionError):
        m.commit(1, 3)
    m.commit(1, 0)
    assert m.seqs[1].committed == m.seqs[1].reserved == 3
    m.release(1)
    check_invariants(m)


def test_identical_prompts_prefilled_together_are_deduped():
    m = make()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    m.allocate(1, prompt)
    m.allocate(2, prompt)
    for sid in (1, 2):
        forward(m, sid)
    m.commit(1, 9)
    m.commit(2, 9)
    t1, t2 = m.seqs[1].block_table, m.seqs[2].block_table
    assert t1[:2] == t2[:2] and t1[2] != t2[2]
    assert m.num_available_blocks() == 12 - 4
    step(m, 2, [10, 11, 12])  # next block publishes under the deduped page
    assert m.tree.nodes[t2[2]].parent is m.tree.nodes[t1[1]]
    check_invariants(m)


def test_attached_partial_page_grows_in_place():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5, 6])
    m.release(1)
    prefill(m, 2, [1, 2, 3, 4, 5, 6, 7])
    step(m, 2, [8])
    assert m.seqs[2].block_table == [0, 1] and m.tree.nodes[1].tokens == (5, 6, 7, 8)
    check_invariants(m)


def test_grow_collision_switches_to_the_existing_page():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5, 6])
    m.release(1)  # partial page [5,6] at block 1
    page = m.tree.nodes[1]
    prefill(m, 2, [1, 2, 3, 4, 5, 6, 7])  # attaches the page alone, writes in place
    prefill(m, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])  # page shared -> copies, publishes [5,6,7,8]
    other = m.seqs[3].block_table[1]
    step(m, 2, [8])
    assert m.seqs[2].block_table == [0, other] and m.ref_counts[other] == 2
    assert m.tree.nodes[1] is page and page.tokens == (5, 6) and m.ref_counts[1] == 0
    check_invariants(m)


def test_partial_match_destination_is_published_when_full():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5, 6, 7])
    m.release(1)
    prefill(m, 2, [1, 2, 3, 4, 5, 6, 9, 9, 9])
    dest = m.seqs[2].block_table[1]
    assert dest != 1 and m.tree.nodes[dest].tokens == (5, 6, 9, 9)
    assert m.tree.nodes[1].tokens == (5, 6, 7)
    check_invariants(m)


# --- release and fork ------------------------------------------------------------------------


def test_release_publishes_the_partial_block_then_frees_tail_first():
    m = make(8)
    prefill(m, 1, list(range(1, 10)))
    step(m, 1, [10])
    m.release(1)
    n = m.tree.nodes
    assert not m.seqs and m.num_free_cached == 3 and m.num_available_blocks() == 8
    assert n[2].tokens == (9, 10) and n[2].parent is n[1]
    assert n[2].last_used < n[1].last_used < n[0].last_used
    check_invariants(m)


def test_release_mid_step_asserts():
    m = make()
    prefill(m, 1, [1, 2, 3])
    m.append_slots(1, [4])
    with pytest.raises(AssertionError):
        m.release(1)


def test_preempted_sequence_resumes_from_its_cached_kv():
    m = make(8)
    prefill(m, 1, [1, 2, 3, 4, 5, 6])
    step(m, 1, [7])
    resume = list(m.seqs[1].token_ids) + [8]  # + the token sampled last step
    m.release(1)
    assert m.calculate_admission_cost(resume).prefill_tokens == 1
    prefill(m, 1, resume)
    assert m.seqs[1].block_table == [0, 1] and m.tree.nodes[1].tokens == (5, 6, 7, 8)
    check_invariants(m)


def test_identical_partial_blocks_released_together_are_deduped():
    m = make(8)
    m.allocate(1, [1, 2, 3, 4, 5, 6])
    m.allocate(2, [1, 2, 3, 4, 5, 6])
    for sid in (1, 2):
        forward(m, sid)
        m.commit(sid, 6)
    m.release(1)
    m.release(2)
    assert len(m.tree.nodes) == 2 and m.num_free_cached == 2
    check_invariants(m)


def test_fork_takes_no_blocks_and_everything_is_freed_after_release():
    m = make()
    prefill(m, 1, [1, 2, 3, 4, 5, 6])
    available = m.num_available_blocks()
    for c in (2, 3, 4):
        m.fork(1, c)
    assert m.num_available_blocks() == available and m.ref_counts[:2] == [4, 4]
    with pytest.raises(AssertionError):
        m.fork(1, 2)
    for sid in (2, 3, 4, 1):
        step(m, sid, [sid, sid])
    assert len(m.tree.nodes[0].children) == 4
    check_invariants(m)
    for sid in (3, 1, 4, 2):
        m.release(sid)
        check_invariants(m)
    assert m.num_available_blocks() == 12 and not any(m.ref_counts)


def test_fork_mid_step_asserts():
    m = make()
    prefill(m, 1, [1, 2])
    m.append_slots(1, [3])
    with pytest.raises(AssertionError):
        m.fork(1, 2)


def test_fork_of_a_held_tree_page_copies_before_writing():
    m = make(8)
    prefill(m, 1, [1, 2, 3, 4, 5])
    m.release(1)
    prefill(m, 2, [1, 2, 3, 4, 5, 6])  # attaches the partial page [5]
    m.fork(2, 3)
    assert m.blocks_needed_for_append(3, 1) == 1
    step(m, 3, [7])
    step(m, 2, [8])
    assert m.seqs[2].block_table == [0, 1] and m.seqs[3].block_table[1] != 1
    m.release(2)
    m.release(3)
    check_invariants(m)


def test_release_after_fork_grows_a_page_its_siblings_share():
    """Writing past an attached page and then forking shares the written slots; release grows
    the page while a sibling still holds it."""
    m = make(8)
    prefill(m, 1, [1, 2, 3, 4, 5])
    m.release(1)  # partial page [5] at block 1
    prefill(m, 2, [1, 2, 3, 4, 5, 6])  # attaches it alone, writes 6 in place
    m.fork(2, 3)
    m.release(2)
    assert m.tree.nodes[1].tokens == (5, 6) and m.ref_counts[1] == 1
    step(m, 3, [7])  # the sibling now holds it alone and writes in place
    assert m.seqs[3].block_table == [0, 1]
    check_invariants(m)
    m.release(3)
    assert m.tree.nodes[1].tokens == (5, 6, 7)
    check_invariants(m)


# --- real storage ----------------------------------------------------------------------------


def real_storage(num_blocks: int = 4, num_layers: int = 3) -> KVStorage:
    return KVStorage(num_blocks=num_blocks, num_layers=num_layers, num_kv_heads=1, head_dim=2, block_size=B,
                     dtype=torch.float32, device="cpu")


def test_partial_match_copies_real_kv_tensors_in_every_layer():
    storage = real_storage()
    m = KVCacheManager(storage)
    m.allocate(1, [1, 2, 3])
    src = m.seqs[1].block_table[0]
    storage.k_pool[src, :, :3] = torch.arange(18.0).view(3, 3, 1, 2)  # different KV in each layer
    storage.v_pool[src, :, :3] = -storage.k_pool[src, :, :3]
    m.commit(1, 3)
    m.release(1)
    dst = m.allocate(2, [1, 2, 9, 9])[0]
    assert dst != src
    torch.testing.assert_close(storage.k_pool[dst, :, :2], storage.k_pool[src, :, :2])
    torch.testing.assert_close(storage.v_pool[dst, :, :2], storage.v_pool[src, :, :2])
    assert not storage.k_pool[dst, :, 2:].any(), "only the matched slots are copied"


def test_layer_views_write_through_to_the_pool():
    storage = real_storage()
    k, v = storage.layer(1)
    assert k.shape == (4, B, 1, 2) and k.data_ptr() == storage.k_pool[:, 1].data_ptr()
    assert not k.is_contiguous(), "a layer slice of a block-first pool is strided"
    k[2, 3] = 7.0
    v[0, 0] = -7.0
    assert storage.k_pool[2, 1, 3].tolist() == [[7.0, 7.0]] and storage.k_pool.sum() == 14.0
    assert storage.v_pool[0, 1, 0].tolist() == [[-7.0, -7.0]]


def test_bytes_per_block_matches_the_allocated_pool():
    storage = real_storage(num_blocks=5)
    pool_bytes = sum(t.numel() * t.element_size() for t in (storage.k_pool, storage.v_pool))
    assert KVStorage.bytes_per_block(3, 1, 2, B, torch.float32) * 5 == pool_bytes
    assert KVStorage.bytes_per_block(3, 1, 2, B, torch.bfloat16) * 2 == KVStorage.bytes_per_block(3, 1, 2, B, torch.float32)


# --- random workload -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_random_workload_keeps_every_invariant(seed):
    """Admissions, decode and speculative steps, forks, preemptions and releases over a tiny
    vocabulary, so prefixes collide, partial matches and dedupes happen often."""
    rng = random.Random(seed)
    m = make(10)
    prefixes = [[rng.randrange(4) for _ in range(rng.randrange(1, 10))] for _ in range(4)]
    next_id = 0
    for _ in range(300):
        op = rng.random()
        if op < 0.3 or not m.seqs:
            prompt = rng.choice(prefixes) + [rng.randrange(4) for _ in range(rng.randrange(1, 6))]
            cost = m.calculate_admission_cost(prompt)
            fits = cost.blocks_needed <= m.num_available_blocks()
            if m.allocate(next_id, prompt) is None:
                assert not fits
            else:
                assert fits
                s = m.seqs[next_id]
                assert s.reserved - s.committed == cost.prefill_tokens
                forward(m, next_id)
                m.commit(next_id, cost.prefill_tokens)
                next_id += 1
        elif op < 0.75:
            sid = rng.choice(list(m.seqs))
            tokens = [rng.randrange(4) for _ in range(rng.choice([1, 1, 1, 3]))]
            needed = m.blocks_needed_for_append(sid, len(tokens))
            if needed > m.num_available_blocks():
                m.release(sid)  # preempt
            else:
                before = m.num_available_blocks()
                m.append_slots(sid, tokens)
                assert before - m.num_available_blocks() == needed
                forward(m, sid)
                m.commit(sid, rng.randint(0, len(tokens)) if rng.random() < 0.3 else len(tokens))
        elif op < 0.85:
            m.fork(rng.choice(list(m.seqs)), next_id)
            next_id += 1
        else:
            m.release(rng.choice(list(m.seqs)))
        check_invariants(m)
