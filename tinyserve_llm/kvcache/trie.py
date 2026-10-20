from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyserve_llm.kvcache.page_tree import PageNode

class TrieNode:
    __slots__ = ("next", "pages")

    def __init__(self) -> None:
        self.next: dict[int, TrieNode] = {} # token_id -> trie node for the following position
        self.pages: set[PageNode] = set() # Child pages whose tokens pass through this token.

class TokenTrie:
    """
    Indexes a page's child pages via their tokens.

    The TokenTrie is used per page when no child page exactly matches the next tokens of the query.
    In that case, the TokenTrie is walked to find the child with the longest match of tokens with the query.
    """
    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, page: PageNode, tokens: Sequence[int]) -> None:
        """
        Index a child page via its tokens. When a page grows, idempotently adds the new tokens
        along its existing tokens.
        """
        assert tokens, "a published page needs to hold at least one token"
        node = self.root
        for token in tokens:
            child = node.next.get(token)
            if child is None:
                child = node.next[token] = TrieNode()
            child.pages.add(page)
            node = child

    def remove(self, page: PageNode, tokens: Sequence[int]) -> None:
        """
        Drop a child page's path, without removing tokens other pages might be sharing.
        """
        node = self.root
        for token in tokens:
            child = node.next[token]
            child.pages.remove(page)
            if not child.pages:
                # No more pages holding this token, or anything below it
                del node.next[token]
                return
            node = child

    def longest_match(self, tokens: Sequence[int]) -> tuple[int, set[PageNode]]:
        """
        Returns the length of the longest prefix match tokens, plus the child pages that reached that length.
        The returned set is the trie's, so it should not be modified, only read.
        """
        node = self.root
        depth = 0
        for token in tokens:
            child = node.next.get(token)
            if child is None:
                break
            node = child
            depth += 1
        return depth, node.pages