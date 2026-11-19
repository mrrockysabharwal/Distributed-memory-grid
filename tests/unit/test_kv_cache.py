"""
Unit tests for KVCacheManager and PagedKVCache.
Maintained by Member 3 (Inference Optimization Engineer).
"""
import pytest
import numpy as np
from backend.inference.kv_cache import (
    KVCacheManager,
    PagedKVCache,
    BlockAllocator,
    _estimate_tensor_bytes,
)


def test_kv_cache_basic_operations():
    cache = KVCacheManager(max_cached_requests=5)

    # Insert cache
    k = np.zeros((1, 4, 10, 64))
    v = np.zeros((1, 4, 10, 64))
    cache.update_cache("req-1", layer_idx=0, kv_tensors=(k, v))

    # Retrieve cache
    res = cache.get_cache("req-1", layer_idx=0)
    assert res is not None
    assert np.array_equal(res[0], k)
    assert np.array_equal(res[1], v)
    assert cache.get_seq_len("req-1", layer_idx=0) == 10

    # Miss
    assert cache.get_cache("req-nonexistent", layer_idx=0) is None
    assert cache.get_seq_len("req-nonexistent", layer_idx=0) == 0


def test_kv_cache_incremental_append():
    cache = KVCacheManager(seq_dim=-2)

    # Prefill step: seq_len = 5
    k_init = np.ones((1, 2, 5, 32))
    v_init = np.ones((1, 2, 5, 32))
    cache.update_cache("req-append", layer_idx=1, kv_tensors=(k_init, v_init))

    # Incremental decode step: seq_len = 1
    k_step = np.ones((1, 2, 1, 32)) * 2
    v_step = np.ones((1, 2, 1, 32)) * 2
    updated_k, updated_v = cache.append_cache("req-append", layer_idx=1, new_k=k_step, new_v=v_step)

    assert updated_k.shape == (1, 2, 6, 32)
    assert updated_v.shape == (1, 2, 6, 32)
    assert cache.get_seq_len("req-append", layer_idx=1) == 6

    # Verify values at the last token position
    assert updated_k[0, 0, 5, 0] == 2.0


def test_kv_cache_lru_eviction():
    # Cache with capacity of 2
    cache = KVCacheManager(max_cached_requests=2)

    cache.update_cache("req-1", layer_idx=0, kv_tensors=np.ones((1, 1, 4, 4)))
    cache.update_cache("req-2", layer_idx=0, kv_tensors=np.ones((1, 1, 4, 4)))

    # Touch req-1 so req-2 becomes the oldest
    _ = cache.get_cache("req-1", layer_idx=0)

    # Adding req-3 should evict req-2
    cache.update_cache("req-3", layer_idx=0, kv_tensors=np.ones((1, 1, 4, 4)))

    assert cache.get_cache("req-1", layer_idx=0) is not None
    assert cache.get_cache("req-3", layer_idx=0) is not None
    assert cache.get_cache("req-2", layer_idx=0) is None


def test_kv_cache_memory_budget_eviction():
    # 1 element of float64 is 8 bytes. A (1, 1, 100, 10) array is 1000 elems = 8000 bytes.
    k = np.zeros((1, 1, 100, 10), dtype=np.float64)
    # Set cap at ~12000 bytes (enough for 1 request, not 2)
    cache = KVCacheManager(max_cached_requests=10, max_memory_bytes=10000)

    cache.update_cache("req-1", layer_idx=0, kv_tensors=(k, k))
    stats1 = cache.current_usage_stats()
    assert stats1["active_requests_in_cache"] == 1

    # Adding req-2 exceeds 10000 bytes, evicting oldest (req-1)
    cache.update_cache("req-2", layer_idx=0, kv_tensors=(k, k))
    assert cache.get_cache("req-1", layer_idx=0) is None
    assert cache.get_cache("req-2", layer_idx=0) is not None


def test_paged_kv_cache():
    paged = PagedKVCache(total_blocks=10, block_size=16)

    # Allocate for 35 tokens -> requires ceil(35/16) = 3 blocks
    success = paged.allocate_for_request("req-paged", layer_idx=0, num_tokens=35)
    assert success is True
    blocks = paged.get_blocks("req-paged", layer_idx=0)
    assert len(blocks) == 3

    stats = paged.usage_stats()
    assert stats["allocated_blocks"] == 3
    assert stats["free_blocks"] == 7

    # Free request
    paged.clear_request("req-paged")
    stats_after = paged.usage_stats()
    assert stats_after["allocated_blocks"] == 0
    assert stats_after["free_blocks"] == 10

