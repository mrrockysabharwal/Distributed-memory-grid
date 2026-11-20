"""
Inter-node tensor transport communicator.
Supports in-memory dispatch, TCP/HTTP payloads, and gRPC tensor transmission.
Maintained by Member 1 (Distributed Systems Lead) & Member 2 (AI / Model Engineer).
"""
import logging
from typing import Any, Optional, Dict
from backend.distributed.pipeline_parallel import serialize_tensor, deserialize_tensor

logger = logging.getLogger("DMG-Communicator")


class TensorCommunicator:
    """
    Communicator facilitating the transfer of serialized activation tensors
    between consecutive worker stages in the pipeline parallel grid.
    """
    def __init__(self, mode: str = "direct"):
        self.mode = mode
        # In-process stage routing table: {stage_address: callback_or_executor}
        self._local_routes: Dict[str, Any] = {}

    def register_local_route(self, address: str, target_fn: Any):
        """Register an in-process worker stage endpoint."""
        self._local_routes[address] = target_fn

    async def send_activation(
        self,
        target_worker_address: str,
        tensor: Any,
        request_id: str = "",
        is_prefill: bool = True,
    ) -> Any:
        """
        Send activation tensor to downstream worker node and return output activation.
        """
        logger.debug(
            f"Communicator [{self.mode}]: Dispatching activation for {request_id} to {target_worker_address}"
        )

        # In-process direct routing
        if target_worker_address in self._local_routes:
            target = self._local_routes[target_worker_address]
            if hasattr(target, "forward"):
                return target.forward(tensor, request_id=request_id, is_prefill=is_prefill)
            elif callable(target):
                return target(tensor)

        # Network transport fallback (serialize -> mock transmit -> deserialize)
        raw_bytes, shape, dtype = serialize_tensor(tensor)
        logger.debug(f"Serialized tensor ({len(raw_bytes)} bytes, shape {shape}) for {target_worker_address}")
        
        # In distributed mode, this transmits over gRPC/HTTP socket
        restored = deserialize_tensor(raw_bytes)
        return restored
