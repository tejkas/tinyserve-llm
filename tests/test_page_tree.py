from __future__ import annotations

import pytest

from tinyserve_llm.kvcache.page_tree import PageTree
from tinyserve_llm.kvcache.trie import TokenTrie

B = 8
A = list(range(100, 108))  # one full block of tokens


def test_trie_longest_match_returns_every_page_at_the_deepest_point():
    trie = TokenTrie()
    p, q = object(), object()
    trie.insert(p, [1, 2, 3])
    trie.insert(q, [1, 2, 9])
    assert trie.longest_match([1, 2, 3, 4]) == (3, {p})
    assert trie.longest_match([1, 2, 7]) == (2, {p, q})
    assert trie.longest_match([5]) == (0, trie.root.pages)


def test_trie_remove_keeps_a_path_another_page_still_uses():
    trie = TokenTrie()
    p, q = object(), object()
    trie.insert(p, [1, 2, 3])
    trie.insert(q, [1, 2, 9])
    trie.remove(p, [1, 2, 3])
    assert trie.longest_match([1, 2, 3]) == (2, {q})
    trie.remove(q, [1, 2, 9])
    assert not trie.root.next


def test_publish_returns_the_existing_page_on_a_duplicate():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    assert a.block_id == 1 and t.nodes == {1: a}
    assert t.publish(t.root, 2, A) is a
    assert 2 not in t.nodes


def test_publish_only_under_the_root_or_a_full_page():
    t = PageTree(B)
    partial = t.publish(t.root, 1, [7, 7])
    with pytest.raises(AssertionError):
        t.publish(partial, 2, [1])


def test_publish_rejects_a_block_already_in_the_tree():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    with pytest.raises(AssertionError):
        t.publish(a, 1, [1])


def test_match_walks_full_pages_then_attaches_an_exact_partial_page():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    c = t.publish(a, 2, [1, 2, 3])
    m = t.match(A + [1, 2, 3, 99])
    assert m.pages == [a, c] and m.partial_len == 0 and m.num_tokens == 11


def test_match_reports_a_partial_match_with_its_candidates():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    c = t.publish(a, 2, [1, 2, 3])
    m = t.match(A + [1, 2, 9, 9])
    assert m.pages == [a] and m.partial_len == 2 and m.partial_candidates == {c}
    assert m.num_tokens == 10


def test_match_leaves_the_last_prompt_token_unmatched():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    m = t.match(A)  # the whole prompt is cached, but its last token must still run
    assert m.pages == [] and m.partial_len == B - 1 and m.partial_candidates == {a}


def test_grow_extends_a_partial_page_in_place():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    c = t.publish(a, 2, [1, 2, 3])
    assert t.grow(c, [1, 2, 3, 4, 5]) is c
    assert c.tokens == (1, 2, 3, 4, 5)
    assert a.children == {(1, 2, 3, 4, 5): c}
    assert t.match(A + [1, 2, 3, 4, 5, 6]).pages == [a, c]


def test_grow_collision_returns_the_existing_page_and_leaves_ours_unchanged():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    c = t.publish(a, 2, [1, 2, 3])
    full = list(range(1, 9))
    d = t.publish(a, 3, full)
    assert t.grow(c, full) is d
    assert c.tokens == (1, 2, 3) and a.children[(1, 2, 3)] is c


def test_grow_must_strictly_extend():
    t = PageTree(B)
    c = t.publish(t.root, 1, [1, 2, 3])
    with pytest.raises(AssertionError):
        t.grow(c, [1, 2, 3])
    with pytest.raises(AssertionError):
        t.grow(c, [1, 2, 4, 5])


def test_evict_only_leaves_and_keeps_shared_trie_paths():
    t = PageTree(B)
    a = t.publish(t.root, 1, A)
    c = t.publish(a, 2, [1, 2, 3, 4, 5])
    d = t.publish(a, 3, list(range(1, 9)))
    with pytest.raises(AssertionError):
        t.evict(a)
    assert t.evict(c) == 2 and 2 not in t.nodes
    m = t.match(A + [1, 2, 3, 4, 5, 77])
    assert m.pages == [a] and m.partial_len == 5 and m.partial_candidates == {d}
    assert t.evict(d) == 3 and not a.children and not a.trie.root.next
    assert t.evict(a) == 1
    assert not t.root.children and not t.root.trie.root.next and not t.nodes
