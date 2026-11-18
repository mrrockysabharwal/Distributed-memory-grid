"""
Worker node client and server runner.
Maintained by Member 1 & Member 2.
"""
import argparse
import asyncio
import logging
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DMG-Worker")


class WorkerNode:
    def __init__(self, worker_id: str, port: int, master_address: str):
        self.worker_id = worker_id
        self.port = port
        self.master_address = master_address
        self.assigned_layers: tuple[int, int] = (-1, -1)
        self._running = False

    async def start(self):
        self._running = True
        logger.info(f"Worker {self.worker_id} listening on port {self.port}, master: {self.master_address}")
        asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        while self._running:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            logger.debug(f"[{self.worker_id}] Heartbeat ping | CPU: {cpu}% | Avail RAM: {mem.available // (1024*1024)}MB")
            await asyncio.sleep(3.0)

    async def stop(self):
        self._running = False
        logger.info(f"Worker {self.worker_id} stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DMG Worker node.")
    parser.add_argument("--worker-id", type=str, default="worker-1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--master-address", type=str, default="localhost:50051")
    args = parser.parse_args()

    node = WorkerNode(args.worker_id, args.port, args.master_address)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(node.start())
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(node.stop())

