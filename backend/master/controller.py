"""
Master coordinator service managing cluster state, worker discovery, heartbeats,
dynamic layer partitioning, and automated fault tolerance.

Maintained by Member 1 (Distributed Systems Lead) & Member 3 (Inference Optimization Engineer).
"""
import os
import time
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any

from backend.master.registry import WorkerRegistry, WorkerInfo
from backend.master.scheduler import RequestScheduler
from backend.inference.kv_cache import KVCacheManager
from backend.monitoring.metrics import MetricsCollector
from backend.security.cluster_auth import ClusterAuthManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DMG-Master")


class MasterController:
    """
    Central coordinator orchestrating:
    - Worker registration, telemetry, and heartbeats.
    - Fault detection: timeout scanning and automatic failover/re-partitioning.
    - Model partition planning and dynamic re-balancing.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        grpc_port: int = 50051,
        heartbeat_timeout_seconds: float = 10.0,
        total_model_layers: int = 12,
        master_secret: str = "",
        require_auth: bool = False,
    ):
        self.host = host
        self.grpc_port = grpc_port
        self.heartbeat_timeout = heartbeat_timeout_seconds
        self.total_model_layers = total_model_layers

        self.registry = WorkerRegistry(heartbeat_timeout_seconds=heartbeat_timeout_seconds)
        self.scheduler = RequestScheduler()
        self.kv_cache_mgr = KVCacheManager()
        self.metrics_collector = MetricsCollector()
        self.auth_manager = ClusterAuthManager(
            master_secret=master_secret,
            require_auth=require_auth,
        )

        self._running = False
        self._partition_map: Dict[str, Tuple[int, int]] = {}

    def register_worker(
        self,
        worker_id: str,
        address: str,
        device_type: str = "cpu",
        total_memory: int = 8589934592,
        available_memory: int = 6442450944,
        auth_token: Optional[str] = None,
        auth_signature: Optional[str] = None,
        auth_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Register a new worker node and assign its initial layer partition.

        When cluster authentication is enabled, the worker must present a valid
        join token and HMAC signature. Without valid credentials the registration
        is rejected.
        """
        # --- Secure join verification ---
        if self.auth_manager.require_auth:
            if not auth_token or not auth_signature or auth_timestamp is None:
                logger.warning(f"Worker '{worker_id}' rejected: missing auth credentials.")
                return {
                    "success": False,
                    "worker_id": worker_id,
                    "reason": "missing_credentials",
                }
            verification = self.auth_manager.verify_worker_join(
                worker_id=worker_id,
                token=auth_token,
                signature=auth_signature,
                timestamp=auth_timestamp,
                source_address=address,
            )
            if not verification["authorized"]:
                return {
                    "success": False,
                    "worker_id": worker_id,
                    "reason": verification["reason"],
                }

        worker = self.registry.register(
            worker_id=worker_id,
            address=address,
            device_type=device_type,
            total_memory=total_memory,
            available_memory=available_memory,
        )
        logger.info(f"Worker registered: {worker_id} at {address} ({device_type})")

        # Automatically rebalance partitions to include new worker
        self.rebalance_partitions()
        assigned = self._partition_map.get(worker_id, (-1, -1))

        return {
            "success": True,
            "worker_id": worker_id,
            "assigned_start_layer": assigned[0],
            "assigned_end_layer": assigned[1],
            "total_layers": self.total_model_layers,
        }

    def record_heartbeat(
        self,
        worker_id: str,
        available_memory: int,
        cpu_util: float = 0.0,
        gpu_util: float = 0.0,
        active_reqs: int = 0,
    ) -> Dict[str, Any]:
        """Record a heartbeat ping from an active worker."""
        updated = self.registry.record_heartbeat(
            worker_id=worker_id,
            available_memory=available_memory,
            cpu_util=cpu_util,
            gpu_util=gpu_util,
            active_reqs=active_reqs,
        )
        if not updated:
            # Worker not found in registry -> instruct to re-register
            return {"acknowledged": False, "should_re_register": True}

        return {"acknowledged": True, "should_re_register": False}

    def rebalance_partitions(self) -> Dict[str, Tuple[int, int]]:
        """
        Dynamically plan and assign contiguous layer partitions across all active workers.
        """
        active_workers = self.registry.get_active_workers()
        if not active_workers:
            self._partition_map = {}
            return {}

        workers_info = [
            {
                "worker_id": w.worker_id,
                "available_memory_mb": max(1, w.available_memory_bytes // (1024 * 1024)),
            }
            for w in active_workers
        ]

        # Use memory-weighted partitioner from RequestScheduler
        self._partition_map = RequestScheduler.plan_partitions_weighted(
            total_layers=self.total_model_layers,
            workers_info=workers_info,
        )
        self.registry.assign_partitions(self._partition_map)
        logger.info(f"Rebalanced layer partitions across {len(active_workers)} workers: {self._partition_map}")
        return self._partition_map

    def check_health_and_failover(self) -> List[str]:
        """
        Scan workers for missing heartbeats.
        If a worker has timed out, evict it and automatically repartition layers to healthy workers.
        """
        now = time.time()
        failed_workers = []

        for wid, worker in list(self.registry._workers.items()):
            if worker.is_active and (now - worker.last_heartbeat > self.heartbeat_timeout):
                logger.warning(
                    f"FAULT DETECTED: Worker '{wid}' timed out (no heartbeat for {now - worker.last_heartbeat:.1f}s)."
                )
                worker.is_active = False
                failed_workers.append(wid)

        if failed_workers:
            logger.info(f"Triggering automated fault-tolerance repartitioning for failed workers: {failed_workers}")
            self.rebalance_partitions()

        return failed_workers

    async def _health_monitor_loop(self):
        """Background loop executing heartbeat timeout checks."""
        while self._running:
            self.check_health_and_failover()
            await asyncio.sleep(2.0)

    async def start(self):
        self._running = True
        logger.info(f"DMG Master Controller active on {self.host}:{self.grpc_port}")
        asyncio.create_task(self._health_monitor_loop())

    async def stop(self):
        self._running = False
        logger.info("DMG Master Controller stopped.")


if __name__ == "__main__":
    import uvicorn
    from backend.api.main import app

    api_port = int(os.environ.get("API_PORT", 8000))
    controller = MasterController(host="0.0.0.0", grpc_port=int(os.environ.get("MASTER_PORT", 50051)))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(controller.start())

    logger.info(f"Starting API Gateway on port {api_port} alongside Master Controller...")
    uvicorn.run(app, host="0.0.0.0", port=api_port)
