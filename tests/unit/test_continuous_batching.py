"""
Unit tests for continuous batching (iteration-level scheduling).
Maintained by Member 3 (Inference Optimization Engineer).
"""
import pytest
import asyncio
from backend.master.scheduler import RequestScheduler, GenerationRequest
from backend.inference.kv_cache import KVCacheManager
from backend.inference.batching import ContinuousBatcher


def test_continuous_batching_lifecycle():
    scheduler = RequestScheduler()
    kv_cache = KVCacheManager()
    batcher = ContinuousBatcher(scheduler, kv_cache, max_batch_size=2)

    # Submit 3 requests:
    # req-1 finishes after 2 tokens
    # req-2 finishes after 4 tokens
    # req-3 waits in queue until a slot opens
    req1 = GenerationRequest(request_id="req-1", max_new_tokens=2, prompt="p1")
    req2 = GenerationRequest(request_id="req-2", max_new_tokens=4, prompt="p2")
    req3 = GenerationRequest(request_id="req-3", max_new_tokens=3, prompt="p3")

    scheduler.submit_request_sync(req1)
    scheduler.submit_request_sync(req2)
    scheduler.submit_request_sync(req3)

    def mock_forward(batch):
        # Return mock logits for each request
        return [None for _ in batch]

    # Iteration 1: req-1 and req-2 admitted (batch size = 2)
    stat1 = batcher.step_sync(mock_forward)
    assert stat1["active_count"] == 2
    assert stat1["admitted_count"] == 2
    assert req1.generated_tokens == 1
    assert req2.generated_tokens == 1
    assert scheduler.queue_size() == 1  # req-3 still in queue

    # Iteration 2: req-1 reaches 2 tokens and will be finished at next check
    stat2 = batcher.step_sync(mock_forward)
    assert req1.generated_tokens == 2
    assert req2.generated_tokens == 2

    # Iteration 3: req-1 is pruned, releasing slot. req-3 is admitted dynamically!
    stat3 = batcher.step_sync(mock_forward)
    assert stat3["finished_count"] == 1  # req-1 finished
    assert stat3["admitted_count"] == 1  # req-3 admitted!
    assert "req-1" in scheduler.completed_requests
    assert req3.generated_tokens == 1
    assert req2.generated_tokens == 3
