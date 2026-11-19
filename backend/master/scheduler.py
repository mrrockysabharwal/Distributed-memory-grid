"""
Inference request queue and partition placement scheduler.
Supports priority queuing, dynamic batch formation, and memory-aware layer partitioning.

Maintained by Member 3 (Inference Optimization Engineer) & Member 1.
"""
import time
import asyncio
import threading
import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Any


class RequestPriority(IntEnum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(order=True)
class GenerationRequest:
    # Priority and arrival time are used for priority queue ordering
    priority: int = RequestPriority.NORMAL
    arrival_time: float = field(default_factory=time.time)
    
    # Non-comparison fields
    request_id: str = field(compare=False, default="")
    model_id: str = field(compare=False, default="distilgpt2")
    prompt: str = field(compare=False, default="")
    max_new_tokens: int = field(compare=False, default=50)
    temperature: float = field(compare=False, default=0.7)
    status: str = field(compare=False, default="queued")
    generated_tokens: int = field(compare=False, default=0)
    output_tokens: List[str] = field(compare=False, default_factory=list)
    is_prefill: bool = field(compare=False, default=True)
    finished: bool = field(compare=False, default=False)


class RequestScheduler:
    """
    Scheduler handling incoming inference requests with priority queuing,
    concurrency controls, and dynamic layer partitioning.
    """
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self._priority_heap: List[GenerationRequest] = []
        self._lock = threading.Lock()
        self.active_requests: Dict[str, GenerationRequest] = {}
        self.completed_requests: Dict[str, GenerationRequest] = {}

    async def submit_request(self, request: GenerationRequest) -> bool:
        """Submit a new request into the priority queue."""
        with self._lock:
            if len(self._priority_heap) >= self.max_queue_size:
                return False
            request.status = "queued"
            heapq.heappush(self._priority_heap, request)
            return True

    def submit_request_sync(self, request: GenerationRequest) -> bool:
        """Synchronous submission helper for non-async contexts."""
        with self._lock:
            if len(self._priority_heap) >= self.max_queue_size:
                return False
            request.status = "queued"
            heapq.heappush(self._priority_heap, request)
            return True

    def next_request_sync(self) -> Optional[GenerationRequest]:
        """Fetch the highest priority pending request synchronously."""
        with self._lock:
            if not self._priority_heap:
                return None
            req = heapq.heappop(self._priority_heap)
            req.status = "running"
            self.active_requests[req.request_id] = req
            return req

    async def next_request(self) -> Optional[GenerationRequest]:
        """Fetch the highest priority pending request."""
        return self.next_request_sync()

    async def get_batch(self, max_batch_size: int) -> List[GenerationRequest]:
        """
        Pull up to max_batch_size requests from the priority queue
        for continuous batching admission.
        """
        batch: List[GenerationRequest] = []
        with self._lock:
            while self._priority_heap and len(batch) < max_batch_size:
                req = heapq.heappop(self._priority_heap)
                req.status = "running"
                self.active_requests[req.request_id] = req
                batch.append(req)
        return batch

    def complete_request(self, request_id: str):
        """Mark a request complete and move to finished history."""
        if request_id in self.active_requests:
            req = self.active_requests.pop(request_id)
            req.status = "completed"
            req.finished = True
            self.completed_requests[request_id] = req

    def queue_size(self) -> int:
        return len(self._priority_heap)

    def active_count(self) -> int:
        return len(self.active_requests)

    @staticmethod
    def plan_partitions(total_layers: int, worker_ids: List[str]) -> Dict[str, Tuple[int, int]]:
        """
        Evenly partition consecutive Transformer layers across available active workers.
        """
        n_workers = len(worker_ids)
        if n_workers == 0:
            return {}

        layers_per_worker = total_layers // n_workers
        remainder = total_layers % n_workers

        assignments: Dict[str, Tuple[int, int]] = {}
        curr = 0
        for i, wid in enumerate(worker_ids):
            count = layers_per_worker + (1 if i < remainder else 0)
            assignments[wid] = (curr, curr + count - 1)
            curr += count
        return assignments

    @staticmethod
    def plan_partitions_weighted(
        total_layers: int, workers_info: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Partition layers proportionally based on each worker's available memory/compute capacity.
        workers_info: list of dicts like [{"worker_id": "w1", "available_memory_mb": 8192}, ...]
        """
        if not workers_info:
            return {}

        total_weight = sum(w.get("available_memory_mb", 1024) for w in workers_info)
        if total_weight <= 0:
            return RequestScheduler.plan_partitions(
                total_layers, [w["worker_id"] for w in workers_info]
            )

        assignments: Dict[str, Tuple[int, int]] = {}
        curr = 0
        allocated_layers = 0

        for i, w in enumerate(workers_info):
            wid = w["worker_id"]
            if i == len(workers_info) - 1:
                # Last worker gets all remaining layers
                count = max(0, total_layers - allocated_layers)
            else:
                weight = w.get("available_memory_mb", 1024)
                count = max(1, round((weight / total_weight) * total_layers))
                # Ensure we don't exceed total layers before last worker
                if allocated_layers + count > total_layers:
                    count = total_layers - allocated_layers

            if count > 0:
                assignments[wid] = (curr, curr + count - 1)
                curr += count
                allocated_layers += count
            else:
                assignments[wid] = (-1, -1)

        return assignments
