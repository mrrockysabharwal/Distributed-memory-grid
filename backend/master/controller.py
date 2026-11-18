"""
Master coordinator service managing cluster state, worker heartbeats, and model dispatch.
Maintained by Member 1 (Distributed Systems Lead).
"""
import asyncio
import logging
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import RequestScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DMG-Master")


class MasterController:
    def __init__(self, host: str = "0.0.0.0", grpc_port: int = 50051):
        self.host = host
        self.grpc_port = grpc_port
        self.registry = WorkerRegistry()
        self.scheduler = RequestScheduler()
        self._running = False

    async def start(self):
        self._running = True
        logger.info(f"DMG Master Controller initializing on {self.host}:{self.grpc_port}")
        # Background worker health monitor loop
        asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self):
        while self._running:
            active_workers = self.registry.get_active_workers()
            logger.debug(f"Heartbeat scan: {len(active_workers)} active workers.")
            await asyncio.sleep(5.0)

    async def stop(self):
        self._running = False
        logger.info("DMG Master Controller stopped.")


if __name__ == "__main__":
    controller = MasterController()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(controller.start())
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(controller.stop())

