"""
Unit tests for request scheduler and layer partitioning math.
"""
from backend.master.scheduler import RequestScheduler


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

