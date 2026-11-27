"""
Prefix KV-Cache Reuse Engine for shared prompt prefix optimization.
Enables automatic detection and reuse of cached KV states from previous
requests that share the same prompt prefix, dramatically reducing
time-to-first-token (TTFT) for common system prompts and few-shot templates.

Features:
- Trie-based prefix matching for efficient longest-common-prefix lookups.
- Copy-on-write semantics: shared prefix cache is never mutated.
- Configurable minimum prefix length threshold.
- Integration with KVCacheManager for seamless cache population.
- Hit/miss statistics and memory savings tracking.

Maintained by the Distributed Memory Grid team.
"""
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from backend.inference.kv_cache import KVCacheManager

logger = logging.getLogger("DMG-PrefixCache")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class TrieNode:
    """Node in a prefix trie storing token IDs and optional cached KV states."""
    __slots__ = ("children", "kv_states", "depth", "ref_count", "last_accessed")

    def __init__(self, depth: int = 0):
        self.children: Dict[int, "TrieNode"] = {}
        self.kv_states: Optional[Dict[int, Any]] = None  # {layer_idx: (K, V)}
        self.depth: int = depth
        self.ref_count: int = 0
        self.last_accessed: float = time.time()


class PrefixCacheManager:
    """
    Trie-based prefix KV-cache reuse engine.

    When a new request arrives, the manager walks the trie using the prompt's
    token IDs to find the longest cached prefix. If a match meets the minimum
    length threshold, the cached KV states are cloned into the request's
    KVCacheManager, and the prefill forward pass starts from the matched
    position rather than token 0 — cutting TTFT proportionally.

    Usage:
        prefix_mgr = PrefixCacheManager(kv_cache_mgr)
        match = prefix_mgr.match_prefix(token_ids)
        if match["matched_length"] > 0:
            prefix_mgr.populate_kv_cache(request_id, match)
            # Start prefill from token position match["matched_length"]
        # After full prefill, store the new prefix:
        prefix_mgr.store_prefix(token_ids, kv_states_dict)
    """

    def __init__(
        self,
        kv_cache_mgr: Optional[KVCacheManager] = None,
        min_prefix_length: int = 4,
        max_entries: int = 256,
    ):
        """
        Args:
            kv_cache_mgr: The KVCacheManager to populate on cache hits.
            min_prefix_length: Minimum number of matching tokens to qualify as a hit.
            max_entries: Maximum number of stored prefix entries (LRU eviction).
        """
        self.kv_cache_mgr = kv_cache_mgr
        self.min_prefix_length = min_prefix_length
        self.max_entries = max_entries
        self._root = TrieNode(depth=0)
        self._entry_count = 0

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0  # Total prefix tokens reused across all hits

    def _tokenize_key(self, token_ids: Any) -> List[int]:
        """Normalize input token IDs into a plain list of ints."""
        if isinstance(token_ids, list):
            return [int(t) for t in token_ids]
        if HAS_NUMPY and isinstance(token_ids, np.ndarray):
            return token_ids.flatten().astype(int).tolist()
        try:
            import torch
            if isinstance(token_ids, torch.Tensor):
                return token_ids.flatten().tolist()
        except ImportError:
            pass
        if isinstance(token_ids, (tuple,)):
            return [int(t) for t in token_ids]
        return list(token_ids)

    def store_prefix(
        self,
        token_ids: Any,
        kv_states: Dict[int, Any],
        prefix_length: Optional[int] = None,
    ) -> int:
        """
        Store KV states for a token prefix in the trie.

        Args:
            token_ids: Full prompt token IDs.
            kv_states: Dict mapping layer_idx -> (key_tensor, value_tensor).
            prefix_length: Number of tokens to store (default: all).

        Returns:
            Number of tokens stored in the prefix entry.
        """
        tokens = self._tokenize_key(token_ids)
        length = prefix_length if prefix_length is not None else len(tokens)
        length = min(length, len(tokens))

        if length < self.min_prefix_length:
            return 0

        # Evict if at capacity
        if self._entry_count >= self.max_entries:
            self._evict_lru()

        # Walk / build trie path
        node = self._root
        for i in range(length):
            tid = tokens[i]
            if tid not in node.children:
                node.children[tid] = TrieNode(depth=i + 1)
            node = node.children[tid]

        # Store KV states at the terminal node
        node.kv_states = kv_states
        node.ref_count += 1
        node.last_accessed = time.time()
        self._entry_count += 1

        logger.debug(f"Stored prefix of length {length} in trie.")
        return length

    def match_prefix(self, token_ids: Any) -> Dict[str, Any]:
        """
        Find the longest cached prefix matching the given token IDs.

        Returns:
            Dict with keys:
            - matched_length (int): Number of prefix tokens matched (0 = miss).
            - kv_states (dict or None): Cached KV states if matched.
            - cache_hit (bool): Whether the match meets minimum threshold.
        """
        tokens = self._tokenize_key(token_ids)
        node = self._root
        best_match_node = None
        best_match_length = 0

        for i, tid in enumerate(tokens):
            if tid not in node.children:
                break
            node = node.children[tid]
            # Check if this node has cached KV states
            if node.kv_states is not None:
                best_match_node = node
                best_match_length = i + 1

        if best_match_length >= self.min_prefix_length and best_match_node is not None:
            self._hits += 1
            self._tokens_saved += best_match_length
            best_match_node.last_accessed = time.time()
            best_match_node.ref_count += 1
            logger.info(f"Prefix cache HIT: {best_match_length} tokens matched.")
            return {
                "matched_length": best_match_length,
                "kv_states": best_match_node.kv_states,
                "cache_hit": True,
            }

        self._misses += 1
        return {
            "matched_length": 0,
            "kv_states": None,
            "cache_hit": False,
        }

    def populate_kv_cache(
        self,
        request_id: str,
        match_result: Dict[str, Any],
    ) -> bool:
        """
        Copy matched prefix KV states into the request's KVCacheManager.

        Uses copy-on-write: numpy arrays are copied to avoid mutation of
        the shared prefix cache.

        Args:
            request_id: The inference request to populate cache for.
            match_result: Output from match_prefix().

        Returns:
            True if cache was populated, False otherwise.
        """
        if not match_result.get("cache_hit") or self.kv_cache_mgr is None:
            return False

        kv_states = match_result.get("kv_states")
        if kv_states is None:
            return False

        for layer_idx, kv_pair in kv_states.items():
            # Deep copy to maintain copy-on-write semantics
            copied = self._deep_copy_kv(kv_pair)
            self.kv_cache_mgr.update_cache(request_id, layer_idx, copied)

        logger.debug(
            f"Populated KV cache for request '{request_id}' from prefix "
            f"({len(kv_states)} layers)."
        )
        return True

    @staticmethod
    def _deep_copy_kv(kv_pair: Any) -> Any:
        """Deep-copy a (key, value) tensor pair for copy-on-write safety."""
        if isinstance(kv_pair, tuple) and len(kv_pair) == 2:
            k, v = kv_pair
            if HAS_NUMPY:
                if isinstance(k, np.ndarray):
                    return (k.copy(), v.copy())
            try:
                import torch
                if isinstance(k, torch.Tensor):
                    return (k.clone(), v.clone())
            except ImportError:
                pass
            # Fallback: return as-is (immutable types or unknown)
            return kv_pair
        return kv_pair

    def _evict_lru(self):
        """Evict the least-recently-accessed prefix entry from the trie."""
        # Collect all leaf nodes with KV states
        candidates: List[Tuple[float, List[int], TrieNode]] = []
        self._collect_leaves(self._root, [], candidates)

        if not candidates:
            return

        # Sort by last_accessed ascending (oldest first)
        candidates.sort(key=lambda c: c[0])
        _, path, node = candidates[0]

        # Clear KV states from this node
        node.kv_states = None
        self._entry_count = max(0, self._entry_count - 1)
        logger.debug(f"Evicted LRU prefix entry at depth {len(path)}.")

    def _collect_leaves(
        self,
        node: TrieNode,
        path: List[int],
        result: List[Tuple[float, List[int], TrieNode]],
    ):
        """Recursively collect all trie nodes that hold KV states."""
        if node.kv_states is not None:
            result.append((node.last_accessed, list(path), node))
        for tid, child in node.children.items():
            path.append(tid)
            self._collect_leaves(child, path, result)
            path.pop()

    def clear(self):
        """Clear all stored prefixes."""
        self._root = TrieNode(depth=0)
        self._entry_count = 0

    def compute_prefix_hash(self, token_ids: Any, length: Optional[int] = None) -> str:
        """Compute a deterministic hash for a token prefix (useful for dedup)."""
        tokens = self._tokenize_key(token_ids)
        if length is not None:
            tokens = tokens[:length]
        return hashlib.sha256(str(tokens).encode("utf-8")).hexdigest()[:16]

    def stats(self) -> Dict[str, Any]:
        """Return prefix cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = round(self._hits / total, 4) if total > 0 else 0.0
        return {
            "prefix_cache_hits": self._hits,
            "prefix_cache_misses": self._misses,
            "hit_rate": hit_rate,
            "total_tokens_saved": self._tokens_saved,
            "stored_entries": self._entry_count,
            "max_entries": self.max_entries,
            "min_prefix_length": self.min_prefix_length,
        }
