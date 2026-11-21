"""
Worker node client and server runner.
Handles registration with Master, periodic heartbeats, partition loading,
forward activation execution, and per-request KV cache management.

Maintained by Member 1 (Distributed Systems Lead) & Member 2 (AI / Model Engineer).
"""
import os
import argparse
import asyncio
import logging
from typing import Optional, Tuple, Any, Dict

from backend.monitoring.telemetry import get_system_telemetry
from backend.worker.executor import PartitionExecutor
from backend.inference.kv_cache import KVCacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DMG-Worker")


class WorkerNode:
    """
    Worker node managing local hardware execution:
    - Auto-registers with Master Controller on startup.
    - Sends periodic heartbeats with memory/CPU telemetry.
    - Hosts PartitionExecutor for its assigned contiguous Transformer layers.
    - Manages local KV-cache states for active requests.
    """
    def __init__(
        self,
        worker_id: str = "worker-1",
        port: int = 50052,
        master_address: str = "localhost:50051",
        device_type: str = "cpu",
    ):
        self.worker_id = worker_id
        self.port = port
        self.master_address = master_address
        self.device_type = device_type

        self.assigned_layers: Tuple[int, int] = (-1, -1)
        self.kv_cache_mgr = KVCacheManager(max_cached_requests=50)
        self.executor: Optional[PartitionExecutor] = None

        self._running = False
        self._master_ref: Optional[Any] = None

    def connect_to_master(self, master_controller: Any):
        """Set in-memory or direct reference to MasterController for local runs."""
        self._master_ref = master_controller

    def register(self) -> bool:
        """Register worker with the Master Controller and receive partition assignment."""
        telemetry = get_system_telemetry()
        total_mem = telemetry.get("total_memory_mb", 8192) * 1024 * 1024
        avail_mem = telemetry.get("available_memory_mb", 6144) * 1024 * 1024

        address = f"127.0.0.1:{self.port}"
        logger.info(f"[{self.worker_id}] Registering with Master at {self.master_address}...")

        if self._master_ref is not None:
            res = self._master_ref.register_worker(
                worker_id=self.worker_id,
                address=address,
                device_type=self.device_type,
                total_memory=total_mem,
                available_memory=avail_mem,
            )
            start_l = res.get("assigned_start_layer", -1)
            end_l = res.get("assigned_end_layer", -1)
            self.load_partition(start_l, end_l)
            logger.info(f"[{self.worker_id}] Successfully registered. Assigned layers: {self.assigned_layers}")
            return True

        # Network mock fallback
        self.load_partition(0, 3)
        return True

    def load_partition(self, start_layer: int, end_layer: int):
        """Configure local PartitionExecutor for assigned layer boundaries."""
        self.assigned_layers = (start_layer, end_layer)
        if start_layer >= 0 and end_layer >= start_layer:
            self.executor = PartitionExecutor(
                start_layer=start_layer,
                end_layer=end_layer,
                device=self.device_type,
                kv_cache_mgr=self.kv_cache_mgr,
            )
            # Load default identity/linear blocks
            num_blocks = end_layer - start_layer + 1
            self.executor.load_layers([lambda x, **kw: x for _ in range(num_blocks)])
            logger.info(f"[{self.worker_id}] Initialized PartitionExecutor for layers {start_layer}-{end_layer}")

    def execute_partition(
        self,
        hidden_states: Any,
        request_id: str = "",
        is_prefill: bool = True,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """Execute forward pass for incoming activation tensor."""
        if self.executor is None:
            raise RuntimeError(f"[{self.worker_id}] No model partition currently loaded.")
        return self.executor.forward(
            hidden_states,
            request_id=request_id,
            is_prefill=is_prefill,
            attention_mask=attention_mask,
        )

    def clear_request_cache(self, request_id: str):
        """Free KV cache associated with finished request on this worker."""
        if self.executor:
            self.executor.clear_cache(request_id)
        else:
            self.kv_cache_mgr.clear_request(request_id)

    def send_heartbeat(self) -> Dict[str, Any]:
        """Collect local telemetry and dispatch heartbeat ping to Master."""
        telemetry = get_system_telemetry()
        avail_mem = telemetry.get("available_memory_mb", 4096) * 1024 * 1024
        cpu_util = float(telemetry.get("cpu_percent", 0.0))
        active_cache = self.kv_cache_mgr.current_usage_stats()["active_requests_in_cache"]

        if self._master_ref is not None:
            return self._master_ref.record_heartbeat(
                worker_id=self.worker_id,
                available_memory=avail_mem,
                cpu_util=cpu_util,
                gpu_util=0.0,
                active_reqs=active_cache,
            )

        logger.debug(f"[{self.worker_id}] Heartbeat ping | CPU: {cpu_util}% | Avail RAM: {avail_mem//(1024*1024)}MB")
        return {"acknowledged": True, "should_re_register": False}

    async def _heartbeat_loop(self):
        """Periodic background heartbeat daemon."""
        while self._running:
            try:
                res = self.send_heartbeat()
                if res.get("should_re_register", False):
                    logger.warning(f"[{self.worker_id}] Master requested re-registration. Reconnecting...")
                    self.register()
            except Exception as e:
                logger.error(f"[{self.worker_id}] Heartbeat error: {e}")
            await asyncio.sleep(3.0)

    async def start(self):
        self._running = True
        logger.info(f"Worker {self.worker_id} starting on port {self.port}, master: {self.master_address}")
        self.register()
        asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        self._running = False
        logger.info(f"Worker {self.worker_id} stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DMG Worker node.")
    parser.add_argument("--worker-id", type=str, default=os.environ.get("WORKER_ID", "worker-1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", 50052)))
    parser.add_argument("--master-address", type=str, default=os.environ.get("MASTER_ADDRESS", "localhost:50051"))
    args = parser.parse_args()

    node = WorkerNode(args.worker_id, args.port, args.master_address)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(node.start())
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(node.stop())
