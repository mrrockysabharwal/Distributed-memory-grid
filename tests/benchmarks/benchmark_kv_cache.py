"""
Benchmark: KV Cache memory scaling, append latency, and paged allocation efficiency.
Maintained by Member 3 (Inference Optimization Engineer).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import numpy as np
from backend.inference.kv_cache import KVCacheManager, PagedKVCache


def benchmark_kv_cache_scaling():
    print("=" * 65)
    print("BENCHMARK 1: KV Cache Memory & Append Latency Scaling")
    print("=" * 65)
    print(f"{'Seq Length':<12} | {'Memory (KB)':<14} | {'Append Latency (ms)':<22} | {'Seq Len Verified'}")
    print("-" * 65)

    seq_lengths = [128, 256, 512, 1024, 2048]
    hidden_dim = 64
    num_heads = 4
    batch_size = 1

    cache = KVCacheManager(max_cached_requests=10)

    for seq_len in seq_lengths:
        req_id = f"bench-req-{seq_len}"
        # Prefill prompt tensor
        k_init = np.random.randn(batch_size, num_heads, seq_len, hidden_dim).astype(np.float32)
        v_init = np.random.randn(batch_size, num_heads, seq_len, hidden_dim).astype(np.float32)
        cache.update_cache(req_id, layer_idx=0, kv_tensors=(k_init, v_init))

        # Single-token decode append
        k_step = np.random.randn(batch_size, num_heads, 1, hidden_dim).astype(np.float32)
        v_step = np.random.randn(batch_size, num_heads, 1, hidden_dim).astype(np.float32)

        start = time.perf_counter()
        # Warmup and 50 iterations
        for _ in range(50):
            cache.append_cache(req_id, layer_idx=0, new_k=k_step, new_v=v_step)
        end = time.perf_counter()

        avg_append_ms = ((end - start) / 50) * 1000
        mem_kb = cache._request_meta[req_id]["memory_bytes"] / 1024
        final_seq = cache.get_seq_len(req_id, layer_idx=0)

        print(f"{seq_len:<12} | {mem_kb:<14.2f} | {avg_append_ms:<22.4f} | {final_seq}")

    stats = cache.current_usage_stats()
    print("-" * 65)
    print(f"Total Requests in Cache: {stats['active_requests_in_cache']}")
    print(f"Total Cache Memory: {stats['total_memory_mb']} MB")
    print(f"Cache Hit Rate: {stats['hit_rate'] * 100}%\n")


def benchmark_paged_cache():
    print("=" * 65)
    print("BENCHMARK 2: Paged KV Cache Block Allocation Throughput")
    print("=" * 65)

    paged = PagedKVCache(total_blocks=1024, block_size=16)
    num_requests = 100
    tokens_per_request = 120  # Ceil(120/16) = 8 blocks per request

    start = time.perf_counter()
    for i in range(num_requests):
        success = paged.allocate_for_request(f"req-{i}", layer_idx=0, num_tokens=tokens_per_request)
        assert success
    alloc_time = time.perf_counter() - start

    stats = paged.usage_stats()
    print(f"Allocated {stats['allocated_blocks']} physical blocks for {num_requests} requests in {alloc_time*1000:.2f} ms")
    print(f"Allocation Rate: {num_requests / alloc_time:.2f} requests/sec")
    print(f"Cache Utilization: {stats['utilization_pct']}% ({stats['allocated_blocks']}/{stats['total_blocks']} blocks)")

    # Deallocation benchmark
    start_free = time.perf_counter()
    for i in range(num_requests):
        paged.clear_request(f"req-{i}")
    free_time = time.perf_counter() - start_free

    stats_after = paged.usage_stats()
    print(f"Freed all requests in {free_time*1000:.2f} ms")
    print(f"Free blocks restored: {stats_after['free_blocks']}/{stats_after['total_blocks']}\n")


if __name__ == "__main__":
    benchmark_kv_cache_scaling()
    benchmark_paged_cache()
