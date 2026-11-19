"""
Unit tests for request scheduler and layer partitioning math.
Maintained by Member 3 (Inference Optimization Engineer) & Member 1.
"""
import pytest
from backend.master.scheduler import RequestScheduler, GenerationRequest, RequestPriority


def test_plan_partitions_even():
    assignments = RequestScheduler.plan_partitions(12, ["worker-1", "worker-2", "worker-3"])
    assert assignments["worker-1"] == (0, 3)
    assert assignments["worker-2"] == (4, 7)
    assert assignments["worker-3"] == (8, 11)


def test_plan_partitions_uneven():
    assignments = RequestScheduler.plan_partitions(6, ["worker-1", "worker-2", "worker-3", "worker-4"])
    # 6 layers across 4 workers: 2 + 2 + 1 + 1
    assert assignments["worker-1"] == (0, 1)
    assert assignments["worker-2"] == (2, 3)
    assert assignments["worker-3"] == (4, 4)
    assert assignments["worker-4"] == (5, 5)


def test_plan_partitions_weighted():
    # Worker 1 has 8GB, Worker 2 has 4GB -> 2:1 ratio for 12 layers = 8 layers and 4 layers
    workers = [
        {"worker_id": "worker-1", "available_memory_mb": 8192},
        {"worker_id": "worker-2", "available_memory_mb": 4096},
    ]
    assignments = RequestScheduler.plan_partitions_weighted(12, workers)
    assert assignments["worker-1"] == (0, 7)  # 8 layers (0 to 7)
    assert assignments["worker-2"] == (8, 11)  # 4 layers (8 to 11)


def test_scheduler_priority_queue():
    scheduler = RequestScheduler(max_queue_size=10)

    low_req = GenerationRequest(request_id="low", priority=RequestPriority.LOW, prompt="test low")
    high_req = GenerationRequest(request_id="high", priority=RequestPriority.HIGH, prompt="test high")
    normal_req = GenerationRequest(request_id="normal", priority=RequestPriority.NORMAL, prompt="test normal")

    # Submit in reverse order
    scheduler.submit_request_sync(low_req)
    scheduler.submit_request_sync(high_req)
    scheduler.submit_request_sync(normal_req)

    # First out must be HIGH priority
    first = scheduler.next_request_sync()
    assert first is not None
    assert first.request_id == "high"

    # Second out must be NORMAL priority
    second = scheduler.next_request_sync()
    assert second is not None
    assert second.request_id == "normal"

    # Third out must be LOW priority
    third = scheduler.next_request_sync()
    assert third is not None
    assert third.request_id == "low"
