"""
Unit tests for Prefix KV-Cache Reuse Engine.
Tests cover trie-based prefix matching, copy-on-write population,
LRU eviction, minimum threshold enforcement, and statistics tracking.
"""
import numpy as np
from backend.inference.kv_cache import KVCacheManager
from backend.inference.prefix_cache import PrefixCacheManager


def _make_kv_states(num_layers: int = 4, seq_len: int = 8):
    """Helper: generate mock KV states dict for testing."""
    states = {}
    for layer in range(num_layers):
        k = np.random.randn(1, 4, seq_len, 16).astype(np.float32)
        v = np.random.randn(1, 4, seq_len, 16).astype(np.float32)
        states[layer] = (k, v)
    return states


def test_store_and_match_prefix():
    """Stored prefix can be matched by the same token IDs."""
    kv_mgr = KVCacheManager()
    prefix_mgr = PrefixCacheManager(kv_cache_mgr=kv_mgr, min_prefix_length=4)

    tokens = [10, 20, 30, 40, 50, 60]
    kv_states = _make_kv_states(num_layers=2, seq_len=6)
    prefix_mgr.store_prefix(tokens, kv_states)

    # Match with same tokens
    result = prefix_mgr.match_prefix(tokens)
    assert result["cache_hit"] is True
    assert result["matched_length"] == 6
    assert result["kv_states"] is not None


def test_partial_prefix_match():
    """Matching works with a longer sequence sharing the same prefix."""
    kv_mgr = KVCacheManager()
    prefix_mgr = PrefixCacheManager(kv_cache_mgr=kv_mgr, min_prefix_length=4)

    prefix_tokens = [1, 2, 3, 4, 5]
    full_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    kv_states = _make_kv_states(num_layers=2, seq_len=5)
    prefix_mgr.store_prefix(prefix_tokens, kv_states)

    result = prefix_mgr.match_prefix(full_tokens)
    assert result["cache_hit"] is True
    assert result["matched_length"] == 5


def test_minimum_prefix_length_enforced():
    """Matches shorter than min_prefix_length are treated as misses."""
    kv_mgr = KVCacheManager()
    prefix_mgr = PrefixCacheManager(kv_cache_mgr=kv_mgr, min_prefix_length=4)

    # Store 3-token prefix (below threshold)
    tokens_short = [1, 2, 3]
    kv_states = _make_kv_states(num_layers=1, seq_len=3)
    stored = prefix_mgr.store_prefix(tokens_short, kv_states)
    assert stored == 0  # Rejected

    result = prefix_mgr.match_prefix(tokens_short)
    assert result["cache_hit"] is False
    assert result["matched_length"] == 0


def test_no_match_on_different_tokens():
    """Completely different tokens produce a cache miss."""
    prefix_mgr = PrefixCacheManager(min_prefix_length=4)
    tokens_a = [10, 20, 30, 40]
    kv_states = _make_kv_states(num_layers=1, seq_len=4)
    prefix_mgr.store_prefix(tokens_a, kv_states)

    tokens_b = [99, 88, 77, 66]
    result = prefix_mgr.match_prefix(tokens_b)
    assert result["cache_hit"] is False


def test_populate_kv_cache_copy_on_write():
    """Populated KV cache entries are copies (not shared references)."""
    kv_mgr = KVCacheManager()
    prefix_mgr = PrefixCacheManager(kv_cache_mgr=kv_mgr, min_prefix_length=4)

    tokens = [1, 2, 3, 4]
    kv_states = _make_kv_states(num_layers=2, seq_len=4)
    original_k_layer0 = kv_states[0][0].copy()
    prefix_mgr.store_prefix(tokens, kv_states)

    match_result = prefix_mgr.match_prefix(tokens)
    populated = prefix_mgr.populate_kv_cache("req-001", match_result)
    assert populated is True

    # Verify the cached copy in kv_mgr is independent of the original
    cached = kv_mgr.get_cache("req-001", 0)
    assert cached is not None
    cached_k = cached[0]
    # Modify the cached copy
    cached_k[0, 0, 0, 0] = 999.0
    # Original should be unchanged
    assert kv_states[0][0][0, 0, 0, 0] == original_k_layer0[0, 0, 0, 0]


def test_lru_eviction():
    """Oldest prefix entries are evicted when max_entries is reached."""
    prefix_mgr = PrefixCacheManager(min_prefix_length=2, max_entries=2)

    # Store 2 entries
    kv1 = _make_kv_states(1, 2)
    kv2 = _make_kv_states(1, 2)
    prefix_mgr.store_prefix([1, 2], kv1)
    prefix_mgr.store_prefix([3, 4], kv2)
    assert prefix_mgr._entry_count == 2

    # Store 3rd — should evict the oldest
    kv3 = _make_kv_states(1, 2)
    prefix_mgr.store_prefix([5, 6], kv3)
    assert prefix_mgr._entry_count <= 2

    # Newest entry should still be matchable
    r = prefix_mgr.match_prefix([5, 6])
    assert r["cache_hit"] is True


def test_numpy_array_input():
    """PrefixCacheManager handles numpy array token IDs."""
    prefix_mgr = PrefixCacheManager(min_prefix_length=3)
    tokens_np = np.array([10, 20, 30, 40])
    kv_states = _make_kv_states(1, 4)
    prefix_mgr.store_prefix(tokens_np, kv_states)

    result = prefix_mgr.match_prefix(np.array([10, 20, 30, 40, 50]))
    assert result["cache_hit"] is True
    assert result["matched_length"] == 4


def test_prefix_hash_deterministic():
    """Prefix hash is deterministic for the same token sequence."""
    prefix_mgr = PrefixCacheManager()
    h1 = prefix_mgr.compute_prefix_hash([1, 2, 3, 4])
    h2 = prefix_mgr.compute_prefix_hash([1, 2, 3, 4])
    h3 = prefix_mgr.compute_prefix_hash([1, 2, 3, 5])
    assert h1 == h2
    assert h1 != h3


def test_prefix_cache_stats():
    """Statistics track hits, misses, and tokens saved."""
    prefix_mgr = PrefixCacheManager(min_prefix_length=3)
    kv = _make_kv_states(1, 4)
    prefix_mgr.store_prefix([1, 2, 3, 4], kv)

    # 1 hit
    prefix_mgr.match_prefix([1, 2, 3, 4, 5])
    # 1 miss
    prefix_mgr.match_prefix([9, 8, 7])

    stats = prefix_mgr.stats()
    assert stats["prefix_cache_hits"] == 1
    assert stats["prefix_cache_misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["total_tokens_saved"] == 4
    assert stats["stored_entries"] == 1


def test_clear_prefix_cache():
    """Clearing the cache removes all stored entries."""
    prefix_mgr = PrefixCacheManager(min_prefix_length=2)
    kv = _make_kv_states(1, 2)
    prefix_mgr.store_prefix([1, 2], kv)
    assert prefix_mgr._entry_count == 1

    prefix_mgr.clear()
    assert prefix_mgr._entry_count == 0
    result = prefix_mgr.match_prefix([1, 2])
    assert result["cache_hit"] is False
