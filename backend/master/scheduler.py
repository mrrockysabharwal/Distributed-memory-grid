"""
Inference request queue and partition placement scheduler.
Maintained by Member 3 (Inference Optimization Engineer) & Member 1.
"""
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class GenerationRequest:
    request_id: str
    model_id: str
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    status: str = "queued"


class RequestScheduler:
    def __init__(self, max_queue_size: int = 100):
        self.queue: asyncio.Queue[GenerationRequest] = asyncio.Queue(maxsize=max_queue_size)
        self.active_requests: Dict[str, GenerationRequest] = {}

    async def submit_request(self, request: GenerationRequest) -> bool:
        if self.queue.full():
            return False
        await self.queue.put(request)
        return True

    async def next_request(self) -> GenerationRequest:
        req = await self.queue.get()
        req.status = "running"
        self.active_requests[req.request_id] = req
        return req

    def complete_request(self, request_id: str):
        if request_id in self.active_requests:
            del self.active_requests[request_id]

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

