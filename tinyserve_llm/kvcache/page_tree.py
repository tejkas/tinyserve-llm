from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tinyserve_llm.kvcache.trie import TokenTrie

@dataclass(eq=False, slots=True)
class PageNode:
    """
    A published block in the prefix tree.

    eq=False allows the node to be hashed (needed for TrieNode.pages set)
    """
    block_id: int
    tokens: tuple[int, ...] # tokens whose KV this block holds, cannot shrink
    parent: PageNode | None
    # Two level indexing over this page's child pages
    children: dict[tuple[int, ...], PageNode] = field(default_factory=dict) # Exact full-block size token tuple matching for the next child PageNode
    trie: TokenTrie = field(default_factory=TokenTrie) # This page's TokenTrie for longest shared run of leading tokens enabling partial block reuse
    last_used: int = 0 # For LRU

@dataclass(slots=True)
class Match:
    """
    Defines what a match in the tree is.
    """
    pages: list[PageNode] # full-block size token match, root-most first
    partial_len: int = 0  # tokens matched inside one more page, beyond `pages`. This is a partial block match.
    partial_candidates: set[PageNode] = field(default_factory=set) # The trie's set of candidate child pages: READ ONLY

    @property
    def num_tokens(self) -> int:
        return sum(len(p.tokens) for p in self.pages) + self.partial_len

class PageTree:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        # Root doesn't need a block
        self.root = PageNode(block_id=-1, tokens=(), parent=None)

    def match(self, token_ids: Sequence[int]) -> Match:
        """
        Takes a sequence of token_ids and returns the longest cached prefix of tokens.
        """
        limit = len(token_ids) - 1 # The last token goes through the model to start generation
        pages: list[PageNode] = []
        node = self.root
        pos = 0
        while pos < limit:
            chunk = tuple(token_ids[pos : min(pos + self.block_size, limit)])
            child = node.children.get(chunk)
            if child is not None:
                # full-block match found
                pages.append(child)
                pos += len(chunk)
                node = child
                continue
            # partial-block match search
            depth, candidates = node.trie.longest_match(chunk)
            if depth == 0:
                break
            # If there's a candidate whose recorded tokens exactly match depth, directly attach that block
            exact = next((p for p in candidates if len(p.tokens) == depth), None)
            if exact is not None:
                pages.append(exact)
                break
            return Match(pages, depth, candidates)
        return Match(pages)