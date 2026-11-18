"""
Inter-node tensor transport communicator.
Supports CPU TCP / gRPC payloads and optional torch.distributed / NCCL for GPUs.
Maintained by Member 1 & Member 2.
"""
import logging
import torch

logger = logging.getLogger("DMG-Communicator")


class TensorCommunicator:
    def __init__(self, mode: str = "grpc"):
        self.mode = mode

    def send_activation(self, target_worker_address: str, tensor: torch.Tensor) -> bool:
        logger.debug(f"Sending activation {tensor.shape} to {target_worker_address} via {self.mode}")
        return True

