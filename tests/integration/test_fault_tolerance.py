"""
Integration tests for cluster registration, heartbeat telemetry,
worker failure detection, and automatic fault-tolerance re-partitioning.
"""
import time
from backend.master.controller import MasterController
from backend.worker.worker_node import WorkerNode


def test_cluster_registration_and_layer_partitioning():
    # 12-layer model distributed across 2 initial workers
    master = MasterController(total_model_layers=12, heartbeat_timeout_seconds=2.0)

    w1 = WorkerNode(worker_id="w1", port=50052)
    w1.connect_to_master(master)
    assert w1.register() is True

    # With only 1 worker, it must hold all layers [0, 11]
    assert w1.assigned_layers == (0, 11)

    w2 = WorkerNode(worker_id="w2", port=50053)
    w2.connect_to_master(master)
    assert w2.register() is True

    # With 2 workers, layers should be evenly split: [0, 5] and [6, 11]
    assert master._partition_map["w1"] == (0, 5)
    assert master._partition_map["w2"] == (6, 11)


def test_dynamic_worker_addition():
    master = MasterController(total_model_layers=12, heartbeat_timeout_seconds=2.0)

    w1 = WorkerNode(worker_id="w1", port=50052)
    w1.connect_to_master(master)
    w1.register()

    w2 = WorkerNode(worker_id="w2", port=50053)
    w2.connect_to_master(master)
    w2.register()

    # Add 3rd worker dynamically
    w3 = WorkerNode(worker_id="w3", port=50054)
    w3.connect_to_master(master)
    w3.register()

    # 12 layers across 3 workers -> 4 layers each (0-3, 4-7, 8-11)
    partitions = master._partition_map
    assert partitions["w1"] == (0, 3)
    assert partitions["w2"] == (4, 7)
    assert partitions["w3"] == (8, 11)


def test_automated_fault_detection_and_failover():
    master = MasterController(total_model_layers=12, heartbeat_timeout_seconds=0.5)

    w1 = WorkerNode(worker_id="w1", port=50052)
    w1.connect_to_master(master)
    w1.register()

    w2 = WorkerNode(worker_id="w2", port=50053)
    w2.connect_to_master(master)
    w2.register()

    w3 = WorkerNode(worker_id="w3", port=50054)
    w3.connect_to_master(master)
    w3.register()

    assert len(master.registry.get_active_workers()) == 3

    # Simulate delay past heartbeat timeout (0.5s)
    time.sleep(0.6)

    # w1 and w3 send pings; w2 fails / crashes (sends no ping)
    w1.send_heartbeat()
    w3.send_heartbeat()

    # Master runs health monitor scan
    failed = master.check_health_and_failover()
    assert "w2" in failed

    # Verify w2 is flagged inactive and remaining active workers are 2
    active = master.registry.get_active_workers()
    assert len(active) == 2
    active_ids = {w.worker_id for w in active}
    assert "w2" not in active_ids

    # Verify fault tolerance: all 12 layers have been re-partitioned across healthy workers w1 & w3
    new_partitions = master._partition_map
    assert "w2" not in new_partitions
    assert new_partitions["w1"] == (0, 5)
    assert new_partitions["w3"] == (6, 11)


def test_worker_recovery_and_reintegration():
    master = MasterController(total_model_layers=12, heartbeat_timeout_seconds=0.5)

    w1 = WorkerNode(worker_id="w1", port=50052)
    w1.connect_to_master(master)
    w1.register()

    w2 = WorkerNode(worker_id="w2", port=50053)
    w2.connect_to_master(master)
    w2.register()

    time.sleep(0.6)
    # w2 fails
    w1.send_heartbeat()
    master.check_health_and_failover()
    assert len(master.registry.get_active_workers()) == 1

    # w2 recovers and re-registers
    w2.register()
    active_after = master.registry.get_active_workers()
    assert len(active_after) == 2
    assert master._partition_map["w1"] == (0, 5)
    assert master._partition_map["w2"] == (6, 11)
