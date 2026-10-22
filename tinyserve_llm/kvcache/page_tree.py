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
    children: dict[tuple[int, ...], PageNode] = field(default_factory=dict) # Exact whole token tuple matching for the next child PageNode
    trie: TokenTrie = field(default_factory=TokenTrie) # This page's TokenTrie for longest shared run of leading tokens enabling partial block reuse
    last_used: int = 0 # For LRU

@dataclass(slots=True)
class Match:
    """
    Defines what a match in the tree is.
    """
    pages: list[PageNode] # pages matched over all recorded tokens, root-most first
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
        self.nodes: dict[int, PageNode] = {} # block_id -> pageNode

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

    def publish(self, parent: PageNode, block_id: int, tokens: Sequence[int]) -> PageNode:
        """
        Adds `block_id` under a parent PageNode as a new child page holding `tokens`.

        Returns this child PageNode that holds `tokens`. If one already exists, it is returned
        unchanged (and its block_id differs), so the caller must dedupe.
        """
        tokens = tuple(tokens)
        assert 0 < len(tokens) <= self.block_size
        assert parent is self.root or len(parent.tokens) == self.block_size, "only full pages can have children"
        assert block_id not in self.nodes, "block is already in the tree. grow() instead"
        existing = parent.children.get(tokens)
        if existing is not None:
            return existing 

        node = PageNode(block_id=block_id, tokens=tokens, parent=parent)
        parent.children[tokens] = node
        parent.trie.insert(node, tokens)
        self.nodes[block_id] = node
        return node

    def grow(self, node: PageNode, tokens: Sequence[int]) -> PageNode:
        """
        Extends a partial page (a page that has fewer than `block_size` tokens) with `tokens`.
        To be used when the page has a single holder (ref == 1) and wrote past the recorded token count, growing it in place.

        Returns this PageNode that holds `tokens`. If a sibling page already holds them, it is returned and `node` is 
        unchanged, so the caller must dedupe.
        """
        tokens = tuple(tokens)
        old = node.tokens
        assert len(old) < len(tokens) <= self.block_size, "strictly requires more tokens than what already exists in this PageNode"
        assert tokens[:len(old)] == old, "tokens should share some exact prefix as what already exists in this PageNode"
        parent = node.parent
        assert parent is not None, "root PageNode cannot grow"
        existing = parent.children.get(tokens)
        if existing is not None:
            return existing # exact match sibling
        del parent.children[old] # old token-tuple entry needs to be replaced
        node.tokens = tokens
        parent.children[tokens] = node # replaces the old token-tuple entry
        parent.trie.insert(node, tokens) # re-walks and extends the old path
        return node

    def evict(self, node: PageNode) -> int:
        """
        Removes a leaf page from the tree and returns its block_id for reuse.
        The caller should check that the block is free.
        """
        assert not node.children, "only leaf nodes may be evicted"
        parent = node.parent
        assert parent is not None, "root PageNode cannot be evicted"
        removed = parent.children.pop(node.tokens)
        assert removed is node
        parent.trie.remove(node, node.tokens)
        del self.nodes[node.block_id]
        return node.block_id