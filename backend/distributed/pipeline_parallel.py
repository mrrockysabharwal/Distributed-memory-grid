"""
Pipeline-parallel activation serialization, stage coordinator, and execution engine.
Maintained by Member 2 (AI / Model Engineer) & Member 1 (Distributed Systems Lead).
"""
import io
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any, Callable

logger = logging.getLogger("DMG-PipelineParallel")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def serialize_tensor(tensor: Any, compression: Optional[str] = None) -> Tuple[bytes, List[int], str]:
    """
    Serialize a PyTorch tensor or NumPy array into raw bytes, shape, and dtype for network transport.
    Supports optional compression ('fp16', 'int8') to reduce network payload size across worker stages.
    """
    buffer = io.BytesIO()

    # Apply optional compression
    if compression == "fp16":
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.float16)
        elif HAS_NUMPY and isinstance(tensor, np.ndarray):
            tensor = tensor.astype(np.float16)
    elif compression == "int8":
        if HAS_NUMPY and isinstance(tensor, np.ndarray):
            max_val = float(np.max(np.abs(tensor))) if tensor.size > 0 else 1.0
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q_arr = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
            np.save(buffer, {"quantized": q_arr, "scale": scale})
            return buffer.getvalue(), list(tensor.shape), "int8_scaled"
        elif HAS_TORCH and isinstance(tensor, torch.Tensor):
            max_val = float(torch.max(torch.abs(tensor)).item()) if tensor.nelement() > 0 else 1.0
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q_t = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
            torch.save({"quantized": q_t, "scale": scale}, buffer)
            return buffer.getvalue(), list(tensor.shape), "int8_scaled"

    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        torch.save(tensor.detach().cpu(), buffer)
        return buffer.getvalue(), list(tensor.shape), str(tensor.dtype)
    elif HAS_NUMPY and isinstance(tensor, np.ndarray):
        np.save(buffer, tensor)
        return buffer.getvalue(), list(tensor.shape), str(tensor.dtype)
    elif isinstance(tensor, (list, tuple)):
        if HAS_NUMPY:
            arr = np.array(tensor)
            np.save(buffer, arr)
            return buffer.getvalue(), list(arr.shape), str(arr.dtype)
    raise TypeError(f"Unsupported tensor type for serialization: {type(tensor)}")


def deserialize_tensor(data: bytes, device: str = "cpu", decompress_to_fp32: bool = True) -> Any:
    """
    Deserialize raw bytes back into a PyTorch tensor or NumPy array.
    Automatically handles decompressed restitution if stored as compressed INT8 or FP16.
    """
    buffer = io.BytesIO(data)
    # Check if the data begins with NumPy format header (\x93NUMPY)
    if data.startswith(b"\x93NUMPY"):
        if HAS_NUMPY:
            obj = np.load(buffer, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
                loaded_dict = obj.item()
                if isinstance(loaded_dict, dict) and "quantized" in loaded_dict and "scale" in loaded_dict:
                    q = loaded_dict["quantized"].astype(np.float32)
                    arr = q * float(loaded_dict["scale"])
                    return arr
            arr = obj
            if decompress_to_fp32 and arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            if HAS_TORCH and device != "numpy":
                try:
                    return torch.from_numpy(arr).to(device)
                except Exception:
                    return arr
            return arr
    # Otherwise attempt torch.load
    if HAS_TORCH:
        loaded = torch.load(buffer, map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "quantized" in loaded and "scale" in loaded:
            q = loaded["quantized"].to(torch.float32)
            return q * float(loaded["scale"])
        if decompress_to_fp32 and isinstance(loaded, torch.Tensor) and loaded.dtype == torch.float16:
            loaded = loaded.to(torch.float32)
        return loaded
    elif HAS_NUMPY:
        # Fallback to numpy if torch is unavailable
        try:
            return np.load(buffer, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Cannot deserialize without PyTorch: {e}")
    raise RuntimeError("Neither PyTorch nor NumPy available for deserialization.")


@dataclass
class PipelineStage:
    """Represents a contiguous partition of Transformer layers hosted on a worker."""
    stage_id: int
    worker_id: str
    start_layer: int
    end_layer: int
    executor: Optional[Any] = None
    address: Optional[str] = None

    @property
    def num_layers(self) -> int:
        if self.start_layer < 0 or self.end_layer < 0:
            return 0
        return self.end_layer - self.start_layer + 1


class PipelineParallelEngine:
    """
    Coordinates forward passes through a sequential chain of worker pipeline stages.
    Handles:
    - Input token embedding extraction
    - Intermediate activation handoffs between adjacent stages
    - Final layer normalization and LM head projection to output logits
    - Integration with KV-cache for incremental decoding
    """
    def __init__(
        self,
        stages: List[PipelineStage],
        embedding_fn: Optional[Callable[[Any], Any]] = None,
        final_head_fn: Optional[Callable[[Any], Any]] = None,
        activation_compression: Optional[str] = None,
    ):
        self.stages = sorted(stages, key=lambda s: s.stage_id)
        self.embedding_fn = embedding_fn or self._default_embedding_fn
        self.final_head_fn = final_head_fn or self._default_head_fn
        self.activation_compression = activation_compression

    @staticmethod
    def _default_embedding_fn(input_ids: Any) -> Any:
        """Default mock/identity embedding projection for testing."""
        hidden_dim = 64
        if HAS_TORCH and isinstance(input_ids, torch.Tensor):
            batch, seq_len = input_ids.shape[:2]
            return torch.zeros((batch, seq_len, hidden_dim), dtype=torch.float32, device=input_ids.device)
        elif HAS_NUMPY and isinstance(input_ids, np.ndarray):
            batch, seq_len = input_ids.shape[:2]
            return np.zeros((batch, seq_len, hidden_dim), dtype=np.float32)
        elif isinstance(input_ids, (list, tuple)):
            seq_len = len(input_ids) if not isinstance(input_ids[0], (list, tuple)) else len(input_ids[0])
            batch = 1 if not isinstance(input_ids[0], (list, tuple)) else len(input_ids)
            if HAS_NUMPY:
                return np.zeros((batch, seq_len, hidden_dim), dtype=np.float32)
            return [[[0.0] * hidden_dim for _ in range(seq_len)] for _ in range(batch)]
        return input_ids

    @staticmethod
    def _default_head_fn(hidden_states: Any, vocab_size: int = 50257) -> Any:
        """Default mock head projection computing logits from final hidden states."""
        if HAS_TORCH and isinstance(hidden_states, torch.Tensor):
            batch, seq_len, _ = hidden_states.shape
            logits = torch.zeros((batch, seq_len, vocab_size), dtype=torch.float32, device=hidden_states.device)
            logits[:, :, 100] = 5.0  # Dominant mock token
            return logits
        elif HAS_NUMPY and isinstance(hidden_states, np.ndarray):
            batch, seq_len, _ = hidden_states.shape
            logits = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)
            logits[:, :, 100] = 5.0
            return logits
        return hidden_states

    def forward_sync(
        self,
        input_ids: Any,
        request_id: str = "",
        is_prefill: bool = True,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """
        Execute forward pass synchronously through all stages in the pipeline.
        """
        x = self.embedding_fn(input_ids)
        for stage in self.stages:
            if stage.executor is not None:
                if hasattr(stage.executor, "forward"):
                    if "request_id" in stage.executor.forward.__code__.co_varnames:
                        x = stage.executor.forward(
                            x,
                            request_id=request_id,
                            is_prefill=is_prefill,
                            attention_mask=attention_mask,
                        )
                    else:
                        x = stage.executor.forward(x, attention_mask=attention_mask)
            elif stage.address:
                data, shape, dtype = serialize_tensor(x, compression=self.activation_compression)
                x = deserialize_tensor(data)

        logits = self.final_head_fn(x)
        return logits

    async def forward(
        self,
        input_ids: Any,
        request_id: str = "",
        is_prefill: bool = True,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """
        Execute forward pass sequentially through all stages in the pipeline.
        """
        return self.forward_sync(
            input_ids=input_ids,
            request_id=request_id,
            is_prefill=is_prefill,
            attention_mask=attention_mask,
        )
