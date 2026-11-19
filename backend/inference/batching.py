"""
Continuous Batching (Iteration-Level Scheduling) Engine.
Enables dynamic admission of incoming requests and instantaneous retirement
of finished sequences at every token generation step.

Maintained by Member 3 (Inference Optimization Engineer).
"""
import time
import asyncio
import logging
from typing import List, Dict, Callable, Optional, Any, Tuple
from backend.master.scheduler import RequestScheduler, GenerationRequest
from backend.inference.kv_cache import KVCacheManager

logger = logging.getLogger("DMG-ContinuousBatcher")


class ContinuousBatcher:
    """
    Manages iteration-level continuous batching for the inference engine.
    Instead of waiting for an entire static batch of sequences to complete,
    the continuous batcher admits new requests on-the-fly and releases resources
    (including KV cache) the exact moment any sequence reaches EOS or length limit.
    """
    def __init__(
        self,
        scheduler: RequestScheduler,
        kv_cache_mgr: KVCacheManager,
        max_batch_size: int = 8,
        eos_token_id: int = 50256,
    ):
        self.scheduler = scheduler
        self.kv_cache_mgr = kv_cache_mgr
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id

        self.active_batch: List[GenerationRequest] = []
        self.total_tokens_generated: int = 0
        self.total_iterations: int = 0
        self._is_running: bool = False

    def admit_new_requests(self, available_slots: int) -> int:
        """Admit pending requests from the scheduler up to available slots."""
        admitted = 0
        while len(self.active_batch) < self.max_batch_size and available_slots > 0:
            if self.scheduler.queue_size() == 0:
                break
            # Pop next highest priority request synchronously from scheduler
            if hasattr(self.scheduler, "_priority_heap") and self.scheduler._priority_heap:
                import heapq
                req = heapq.heappop(self.scheduler._priority_heap)
                req.status = "running"
                req.is_prefill = True
                self.scheduler.active_requests[req.request_id] = req
                self.active_batch.append(req)
                admitted += 1
                available_slots -= 1
            else:
                break
        return admitted

    def prune_finished_requests(self) -> List[GenerationRequest]:
        """
        Identify requests that have reached max_new_tokens or hit EOS,
        clean up their KV cache, and notify the scheduler.
        """
        still_active: List[GenerationRequest] = []
        finished: List[GenerationRequest] = []

        for req in self.active_batch:
            if req.finished or req.generated_tokens >= req.max_new_tokens:
                req.status = "completed"
                req.finished = True
                self.kv_cache_mgr.clear_request(req.request_id)
                self.scheduler.complete_request(req.request_id)
                finished.append(req)
            else:
                still_active.append(req)

        self.active_batch = still_active
        return finished

    def step_sync(
        self,
        forward_fn: Callable[[List[GenerationRequest]], Any],
        sample_fn: Optional[Callable[[GenerationRequest, Any], Tuple[int, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one continuous batching iteration synchronously:
        1. Prune finished requests and free their KV cache.
        2. Dynamically admit new waiting requests from scheduler.
        3. Partition into prefill and decode groups.
        4. Execute model forward pass.
        5. Sample new tokens and append to sequences.
        """
        step_start = time.time()
        self.total_iterations += 1

        # 1. Prune finished requests
        finished = self.prune_finished_requests()

        # 2. Dynamically admit new requests
        slots_open = self.max_batch_size - len(self.active_batch)
        newly_admitted = self.admit_new_requests(slots_open)

        if not self.active_batch:
            return {
                "iteration": self.total_iterations,
                "active_count": 0,
                "prefill_count": 0,
                "decode_count": 0,
                "tokens_generated": 0,
                "step_latency_ms": round((time.time() - step_start) * 1000, 2),
                "finished_count": len(finished),
                "admitted_count": newly_admitted,
            }

        # 3. Categorize active batch
        prefill_reqs = [r for r in self.active_batch if r.is_prefill]
        decode_reqs = [r for r in self.active_batch if not r.is_prefill]

        # 4. Execute forward pass
        step_outputs = forward_fn(self.active_batch)

        # 5. Process and sample tokens
        step_tokens_generated = 0
        for i, req in enumerate(self.active_batch):
            output_for_req = step_outputs[i] if isinstance(step_outputs, (list, tuple)) else step_outputs
            
            if sample_fn is not None:
                token_id, token_str = sample_fn(req, output_for_req)
            else:
                # Default mock token generation for benchmark/testing
                token_id = 100 + self.total_tokens_generated
                token_str = f"tok_{req.generated_tokens} "

            req.generated_tokens += 1
            step_tokens_generated += 1
            self.total_tokens_generated += 1
            req.output_tokens.append(token_str)

            # Check EOS
            if token_id == self.eos_token_id:
                req.finished = True

            # Once prefill iteration executes, switch state to decode
            if req.is_prefill:
                req.is_prefill = False

        step_latency = time.time() - step_start

        return {
            "iteration": self.total_iterations,
            "active_count": len(self.active_batch),
            "prefill_count": len(prefill_reqs),
            "decode_count": len(decode_reqs),
            "tokens_generated": step_tokens_generated,
            "step_latency_ms": round(step_latency * 1000, 2),
            "finished_count": len(finished),
            "admitted_count": newly_admitted,
        }

    async def step(
        self,
        forward_fn: Callable[[List[GenerationRequest]], Any],
        sample_fn: Optional[Callable[[GenerationRequest, Any], Tuple[int, str]]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for continuous batching step."""
        if asyncio.iscoroutinefunction(forward_fn):
            async def _async_runner(batch):
                return await forward_fn(batch)
            # execute coroutine then process
            outputs = await forward_fn(self.active_batch)
            return self.step_sync(lambda b: outputs, sample_fn=sample_fn)
        return self.step_sync(forward_fn, sample_fn=sample_fn)

    def stats(self) -> Dict[str, Any]:
        """Summary of batcher performance and resource usage."""
        return {
            "total_iterations": self.total_iterations,
            "total_tokens_generated": self.total_tokens_generated,
            "current_active_batch_size": len(self.active_batch),
            "max_batch_size": self.max_batch_size,
            "kv_cache_stats": self.kv_cache_mgr.current_usage_stats(),
        }
