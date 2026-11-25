"""
Weight quantization engine and numerical precision converter (FP32, FP16, INT8).
Supports dynamic symmetric integer quantization, memory savings tracking, and dequantized linear operations.
Compatible with both PyTorch tensors and NumPy arrays.

Maintained by Member 2 (AI / Model Engineer).
"""
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List, Union

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


@dataclass
class QuantizedWeight:
    """
    Encapsulates a quantized tensor with its scaling factor, zero-point,
    and memory footprint metadata.
    """
    quantized_data: Any
    scale: float
    zero_point: int
    original_shape: Tuple[int, ...]
    precision: str  # "fp32", "fp16", "int8"
    original_bytes: int

    @property
    def memory_bytes(self) -> int:
        """Calculate the actual in-memory bytes occupied by this quantized weight."""
        if self.precision == "fp32":
            return self.original_bytes

        if HAS_NUMPY and isinstance(self.quantized_data, np.ndarray):
            return int(self.quantized_data.nbytes) + 8  # data bytes + scale float
        elif HAS_TORCH and isinstance(self.quantized_data, torch.Tensor):
            return int(self.quantized_data.element_size() * self.quantized_data.nelement()) + 8
        elif isinstance(self.quantized_data, (list, tuple)):
            if self.precision == "int8":
                elem_bytes = 1
            elif self.precision == "fp16":
                elem_bytes = 2
            else:
                elem_bytes = 4
            num_elems = 1
            for dim in self.original_shape:
                num_elems *= dim
            return num_elems * elem_bytes + 8
        return self.original_bytes

    @property
    def compression_ratio(self) -> float:
        """Ratio of uncompressed FP32 bytes to quantized memory bytes."""
        mem = self.memory_bytes
        if mem <= 0:
            return 1.0
        return round(self.original_bytes / mem, 2)

    def dequantize(self) -> Any:
        """Reconstruct floating-point tensor/array from quantized data."""
        if self.precision == "fp32":
            return self.quantized_data

        if self.precision == "fp16":
            if HAS_NUMPY and isinstance(self.quantized_data, np.ndarray):
                return self.quantized_data.astype(np.float32)
            elif HAS_TORCH and isinstance(self.quantized_data, torch.Tensor):
                return self.quantized_data.to(torch.float32)
            elif isinstance(self.quantized_data, list):
                return self.quantized_data

        if self.precision == "int8":
            if HAS_NUMPY and isinstance(self.quantized_data, np.ndarray):
                # Dequantize: W_approx = (W_q - zero_point) * scale
                centered = self.quantized_data.astype(np.float32) - float(self.zero_point)
                return centered * float(self.scale)
            elif HAS_TORCH and isinstance(self.quantized_data, torch.Tensor):
                centered = self.quantized_data.to(torch.float32) - float(self.zero_point)
                return centered * float(self.scale)
            elif isinstance(self.quantized_data, list):
                return [float((x - self.zero_point) * self.scale) for x in self.quantized_data]

        return self.quantized_data


class Quantizer:
    """
    Dynamic quantization engine supporting FP32 (identity), FP16 (half precision),
    and INT8 (symmetric dynamic linear quantization).
    """

    @staticmethod
    def quantize_tensor(tensor: Any, target_precision: str = "int8") -> QuantizedWeight:
        """
        Quantize an arbitrary tensor or ndarray to target_precision ('fp32', 'fp16', 'int8').
        """
        target_precision = target_precision.lower()
        if target_precision not in ("fp32", "fp16", "int8"):
            raise ValueError(f"Unsupported target precision '{target_precision}'. Must be fp32, fp16, or int8.")

        # Determine original shape and element count
        if HAS_NUMPY and isinstance(tensor, np.ndarray):
            shape = tensor.shape
            orig_bytes = int(tensor.size * 4)  # baseline 4 bytes per FP32 element
        elif HAS_TORCH and isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
            orig_bytes = int(tensor.nelement() * 4)
        elif isinstance(tensor, (list, tuple)):
            arr = np.array(tensor, dtype=np.float32) if HAS_NUMPY else tensor
            shape = arr.shape if hasattr(arr, "shape") else (len(tensor),)
            orig_bytes = len(tensor) * 4
            tensor = arr
        else:
            raise TypeError(f"Unsupported tensor type for quantization: {type(tensor)}")

        if target_precision == "fp32":
            return QuantizedWeight(
                quantized_data=tensor,
                scale=1.0,
                zero_point=0,
                original_shape=shape,
                precision="fp32",
                original_bytes=orig_bytes,
            )

        if target_precision == "fp16":
            if HAS_NUMPY and isinstance(tensor, np.ndarray):
                q_data = tensor.astype(np.float16)
            elif HAS_TORCH and isinstance(tensor, torch.Tensor):
                q_data = tensor.to(torch.float16)
            else:
                q_data = tensor

            return QuantizedWeight(
                quantized_data=q_data,
                scale=1.0,
                zero_point=0,
                original_shape=shape,
                precision="fp16",
                original_bytes=orig_bytes,
            )

        if target_precision == "int8":
            if HAS_NUMPY and isinstance(tensor, np.ndarray):
                flat = tensor.flatten()
                max_abs = float(np.max(np.abs(flat))) if len(flat) > 0 else 1.0
                scale = max(max_abs / 127.0, 1e-8)
                zero_point = 0
                quantized = np.clip(np.round(flat / scale), -128, 127).astype(np.int8)
                q_data = quantized.reshape(shape)
            elif HAS_TORCH and isinstance(tensor, torch.Tensor):
                max_abs = float(torch.max(torch.abs(tensor)).item())
                scale = max(max_abs / 127.0, 1e-8)
                zero_point = 0
                quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
                q_data = quantized
            else:
                q_data = tensor
                scale = 1.0
                zero_point = 0

            return QuantizedWeight(
                quantized_data=q_data,
                scale=scale,
                zero_point=zero_point,
                original_shape=shape,
                precision="int8",
                original_bytes=orig_bytes,
            )

        return QuantizedWeight(tensor, 1.0, 0, shape, "fp32", orig_bytes)


class QuantizedLinear:
    """
    Linear projection layer storing weight parameters in INT8 or FP16 format
    and executing forward matrix multiplication.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Any,
        bias: Optional[Any] = None,
        precision: str = "int8",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision.lower()
        self.quantized_weight = Quantizer.quantize_tensor(weight, target_precision=self.precision)
        self.bias = bias

    def forward(self, x: Any) -> Any:
        """Execute linear transformation: y = x @ W.T + bias"""
        w = self.quantized_weight.dequantize()

        if HAS_NUMPY and isinstance(x, np.ndarray):
            out = np.matmul(x, w.T if w.shape == (self.out_features, self.in_features) else w)
            if self.bias is not None:
                out = out + self.bias
            return out
        elif HAS_TORCH and isinstance(x, torch.Tensor):
            w_tensor = w if isinstance(w, torch.Tensor) else torch.from_numpy(w).to(x.device)
            out = torch.matmul(x, w_tensor.t() if w_tensor.shape == (self.out_features, self.in_features) else w_tensor)
            if self.bias is not None:
                out = out + self.bias
            return out

        return x

    def __call__(self, x: Any) -> Any:
        return self.forward(x)


class QuantizedTransformerBlock:
    """
    Quantized transformer decoder block wrapping attention and MLP weights in INT8 or FP16.
    """
    def __init__(self, layer_idx: int, hidden_dim: int = 64, precision: str = "int8"):
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.precision = precision.lower()

        # Initialize mock weight matrices
        if HAS_NUMPY:
            np.random.seed(42 + layer_idx)
            w_attn = np.eye(hidden_dim, dtype=np.float32) + 0.05 * np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
            w_mlp = np.eye(hidden_dim, dtype=np.float32) + 0.05 * np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        else:
            w_attn = [[1.0 if i == j else 0.0 for j in range(hidden_dim)] for i in range(hidden_dim)]
            w_mlp = [[1.0 if i == j else 0.0 for j in range(hidden_dim)] for i in range(hidden_dim)]

        self.attn_proj = QuantizedLinear(hidden_dim, hidden_dim, w_attn, precision=self.precision)
        self.mlp_proj = QuantizedLinear(hidden_dim, hidden_dim, w_mlp, precision=self.precision)

    def __call__(
        self,
        x: Any,
        layer_past: Optional[Any] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Tuple[Any, Any]:
        # Self-attention projection with residual
        attn_out = self.attn_proj(x)
        h = x + attn_out if HAS_NUMPY and isinstance(x, np.ndarray) else x

        # MLP projection with residual
        mlp_out = self.mlp_proj(h)
        out = h + mlp_out if HAS_NUMPY and isinstance(h, np.ndarray) else h

        presents = (out, out) if use_cache else None
        return out, presents

    def memory_footprint_bytes(self) -> Dict[str, Any]:
        """Report memory footprint before and after quantization."""
        orig_bytes = self.attn_proj.quantized_weight.original_bytes + self.mlp_proj.quantized_weight.original_bytes
        quant_bytes = self.attn_proj.quantized_weight.memory_bytes + self.mlp_proj.quantized_weight.memory_bytes
        savings = (1.0 - (quant_bytes / orig_bytes)) * 100.0 if orig_bytes > 0 else 0.0

        return {
            "layer_idx": self.layer_idx,
            "precision": self.precision,
            "uncompressed_bytes": orig_bytes,
            "quantized_bytes": quant_bytes,
            "savings_percent": round(savings, 2),
            "compression_ratio": round(orig_bytes / quant_bytes, 2) if quant_bytes > 0 else 1.0,
        }
