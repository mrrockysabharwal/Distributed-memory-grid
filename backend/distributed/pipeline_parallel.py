"""
Pipeline-parallel activation serialization and stage coordinator.
Maintained by Member 2 (AI / Model Engineer) & Member 1.
"""
import io
import torch
from typing import Tuple


def serialize_tensor(tensor: torch.Tensor) -> Tuple[bytes, list[int], str]:
    """Serialize a PyTorch tensor into raw bytes, shape, and dtype for network transport."""
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue(), list(tensor.shape), str(tensor.dtype)


def deserialize_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """Deserialize raw bytes back into a PyTorch tensor on target device."""
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location=device)
    return tensor

