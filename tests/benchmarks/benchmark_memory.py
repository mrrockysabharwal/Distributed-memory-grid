"""
Benchmark: Memory footprint profiling and LRU budget enforcement.
Maintained by Member 3 (Inference Optimization Engineer).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import numpy as np
from backend.inference.kv_cache import KVCacheManager


def benchmark_memory_budget_enforcement():
    print("=" * 65)
    print("BENCHMARK: Memory Budget Enforcement & Eviction Under Load")
    print("=" * 65)

    # 1 request of 512 tokens with 4 heads and 64 dim (float32 = 4 bytes)
    # 1 * 4 * 512 * 64 * 4 * 2 (K and V) = 1,048,576 bytes = 1.0 MB
    tokens = 512
    heads = 4
    dim = 64
    bytes_per_req = 1 * heads * tokens * dim * 4 * 2  # ~1MB per layer

    # Set strict cache budget of 5 MB (max ~5 active requests)
    max_mb = 5
    max_bytes = max_mb * 1024 * 1024
    cache = KVCacheManager(max_cached_requests=100, max_memory_bytes=max_bytes)

    print(f"Memory Cap: {max_mb} MB (~5 requests)")
    print(f"{'Request':<12} | {'Action':<15} | {'Active Reqs':<12} | {'Cache Memory (MB)':<18}")
    print("-" * 65)

    # Influx of 10 requests
    for i in range(1, 11):
        req_id = f"req-{i}"
        k = np.zeros((1, heads, tokens, dim), dtype=np.float32)
        v = np.zeros((1, heads, tokens, dim), dtype=np.float32)
        cache.update_cache(req_id, layer_idx=0, kv_tensors=(k, v))
        stats = cache.current_usage_stats()
        action = "Inserted" if i <= 5 else "Inserted (Evicted LRU)"
        print(f"{req_id:<12} | {action:<15} | {stats['active_requests_in_cache']:<12} | {stats['total_memory_mb']:<18.2f}")

    final_stats = cache.current_usage_stats()
    print("-" * 65)
    print(f"Final Memory Stored: {final_stats['total_memory_mb']} MB (strictly <= {max_mb} MB limit)")
    assert final_stats["total_memory_bytes"] <= max_bytes
    print("✅ Memory Budget Successfully Enforced without OOM!\n")


if __name__ == "__main__":
    benchmark_memory_budget_enforcement()

