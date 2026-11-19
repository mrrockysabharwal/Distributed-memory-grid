"""
Benchmark: Continuous Batching vs. Static Batching.
Demonstrates throughput gain, slot utilization, and latency reduction
achieved by iteration-level scheduling.

Maintained by Member 3 (Inference Optimization Engineer).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import random
from typing import List
from backend.master.scheduler import RequestScheduler, GenerationRequest
from backend.inference.kv_cache import KVCacheManager
from backend.inference.batching import ContinuousBatcher


def simulate_continuous_batching(
    requests_tokens: List[int], max_batch_size: int = 4, step_time_ms: float = 2.0
):
    scheduler = RequestScheduler(max_queue_size=200)
    kv_cache = KVCacheManager()
    batcher = ContinuousBatcher(scheduler, kv_cache, max_batch_size=max_batch_size)

    # Submit all requests
    for i, n_toks in enumerate(requests_tokens):
        req = GenerationRequest(
            request_id=f"req-{i}",
            max_new_tokens=n_toks,
            prompt=f"prompt {i}",
        )
        scheduler.submit_request_sync(req)

    total_requests = len(requests_tokens)
    total_tokens = sum(requests_tokens)

    start_time = time.perf_counter()
    iterations = 0
    active_slot_steps = 0

    def mock_forward(batch):
        time.sleep(step_time_ms / 1000.0)  # Simulate forward pass duration
        return [None for _ in batch]

    while len(scheduler.completed_requests) < total_requests:
        stat = batcher.step_sync(mock_forward)
        iterations += 1
        active_slot_steps += stat["active_count"]

    elapsed = time.perf_counter() - start_time
    throughput = total_tokens / elapsed
    slot_capacity_steps = iterations * max_batch_size
    utilization_pct = (active_slot_steps / slot_capacity_steps) * 100

    return {
        "elapsed_sec": elapsed,
        "throughput_tokens_per_sec": throughput,
        "iterations": iterations,
        "utilization_pct": utilization_pct,
    }


def simulate_static_batching(
    requests_tokens: List[int], max_batch_size: int = 4, step_time_ms: float = 2.0
):
    total_tokens = sum(requests_tokens)
    start_time = time.perf_counter()
    iterations = 0
    active_slot_steps = 0

    # Chunk requests into fixed static batches
    for i in range(0, len(requests_tokens), max_batch_size):
        chunk = requests_tokens[i : i + max_batch_size]
        batch_size = len(chunk)
        max_tokens_in_batch = max(chunk)

        # In static batching, every request in the batch must wait for max_tokens_in_batch steps
        for step in range(max_tokens_in_batch):
            time.sleep(step_time_ms / 1000.0)
            iterations += 1
            # Active non-padded slots at this step
            active_at_step = sum(1 for t in chunk if t > step)
            active_slot_steps += active_at_step

    elapsed = time.perf_counter() - start_time
    throughput = total_tokens / elapsed
    slot_capacity_steps = iterations * max_batch_size
    utilization_pct = (active_slot_steps / slot_capacity_steps) * 100

    return {
        "elapsed_sec": elapsed,
        "throughput_tokens_per_sec": throughput,
        "iterations": iterations,
        "utilization_pct": utilization_pct,
    }


def run_batching_benchmark():
    print("=" * 70)
    print("BENCHMARK: Continuous Batching vs. Static Batching Simulation")
    print("=" * 70)

    # 16 requests with varying sequence lengths (mimicking real-world traffic)
    random.seed(42)
    requests = [random.randint(5, 30) for _ in range(16)]
    max_batch_size = 4
    step_duration_ms = 1.0  # 1ms simulated forward pass

    print(f"Workload: 16 requests, token lengths: {requests}")
    print(f"Total tokens to generate: {sum(requests)} | Batch size limit: {max_batch_size}\n")

    print("Running Static Batching...")
    static_res = simulate_static_batching(requests, max_batch_size, step_duration_ms)

    print("Running Continuous Batching (Iteration-level scheduling)...")
    cont_res = simulate_continuous_batching(requests, max_batch_size, step_duration_ms)

    speedup = cont_res["throughput_tokens_per_sec"] / static_res["throughput_tokens_per_sec"]

    print("\n" + "-" * 70)
    print(f"{'Metric':<30} | {'Static Batching':<18} | {'Continuous Batching'}")
    print("-" * 70)
    print(f"{'Execution Time (s)':<30} | {static_res['elapsed_sec']:<18.3f} | {cont_res['elapsed_sec']:.3f}")
    print(f"{'Throughput (tokens/sec)':<30} | {static_res['throughput_tokens_per_sec']:<18.1f} | {cont_res['throughput_tokens_per_sec']:.1f}")
    print(f"{'Total Iterations':<30} | {static_res['iterations']:<18} | {cont_res['iterations']}")
    print(f"{'Slot Utilization (%)':<30} | {static_res['utilization_pct']:<18.1f} | {cont_res['utilization_pct']:.1f}")
    print("-" * 70)
    print(f"🚀 Speedup Factor: {speedup:.2f}x Throughput Improvement with Continuous Batching!\n")


if __name__ == "__main__":
    run_batching_benchmark()

