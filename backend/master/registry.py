"""
Worker registry and cluster membership management.
Maintained by Member 1 (Distributed Systems Lead).
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WorkerInfo:
    worker_id: str
    address: str
    device_type: str = "cpu"
    total_memory_bytes: int = 0
    available_memory_bytes: int = 0
    start_layer: int = -1
    end_layer: int = -1
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    active_requests: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_seconds: float = 10.0):
        self._workers: Dict[str, WorkerInfo] = {}
        self.heartbeat_timeout = heartbeat_timeout_seconds

    def register(self, worker_id: str, address: str, device_type: str, total_memory: int, available_memory: int) -> WorkerInfo:
        worker = WorkerInfo(
            worker_id=worker_id,
            address=address,
            device_type=device_type,
            total_memory_bytes=total_memory,
            available_memory_bytes=available_memory,
            last_heartbeat=time.time(),
            is_active=True,
        )
        self._workers[worker_id] = worker
        return worker

    def record_heartbeat(self, worker_id: str, available_memory: int, cpu_util: float, gpu_util: float, active_reqs: int) -> bool:
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        worker.last_heartbeat = time.time()
        worker.available_memory_bytes = available_memory
        worker.cpu_utilization = cpu_util
        worker.gpu_utilization = gpu_util
        worker.active_requests = active_reqs
        worker.is_active = True
        return True

    def get_active_workers(self) -> List[WorkerInfo]:
        now = time.time()
        active = []
        for worker in self._workers.values():
            if now - worker.last_heartbeat <= self.heartbeat_timeout:
                worker.is_active = True
                active.append(worker)
            else:
                worker.is_active = False
        return active

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        return self._workers.get(worker_id)

    def assign_partitions(self, assignments: Dict[str, tuple[int, int]]):
        for worker_id, (start_l, end_l) in assignments.items():
            if worker_id in self._workers:
                self._workers[worker_id].start_layer = start_l
                self._workers[worker_id].end_layer = end_l

