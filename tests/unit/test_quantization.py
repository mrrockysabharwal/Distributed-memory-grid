"""
Unit tests for FP16 and INT8 weight quantization engine.
Maintained by Member 2 (AI / Model Engineer).
"""
import pytest
import numpy as np

from backend.inference.quantization import (
    Quantizer,
    QuantizedWeight,
    QuantizedLinear,
    QuantizedTransformerBlock,
)


def test_quantizer_fp32_identity():
    arr = np.array([1.0, 2.5, -3.0, 4.2], dtype=np.float32)
    q = Quantizer.quantize_tensor(arr, target_precision="fp32")
    assert q.precision == "fp32"
    assert q.compression_ratio == 1.0
    dequant = q.dequantize()
    np.testing.assert_allclose(arr, dequant)


def test_quantizer_fp16_compression():
    arr = np.random.randn(32, 64).astype(np.float32)
    q = Quantizer.quantize_tensor(arr, target_precision="fp16")
    assert q.precision == "fp16"
    assert q.quantized_data.dtype == np.float16
    # FP16 should occupy approximately half the memory of FP32
    assert q.compression_ratio >= 1.8
    dequant = q.dequantize()
    np.testing.assert_allclose(arr, dequant, atol=1e-3, rtol=1e-2)


def test_quantizer_int8_dynamic_quantization():
    arr = np.random.randn(64, 128).astype(np.float32)
    q = Quantizer.quantize_tensor(arr, target_precision="int8")
    assert q.precision == "int8"
    assert q.quantized_data.dtype == np.int8
    # INT8 should achieve ~3.5x to 4x compression over FP32
    assert q.compression_ratio >= 3.5
    dequant = q.dequantize()
    # Check max error is bounded by scale
    max_err = np.max(np.abs(arr - dequant))
    assert max_err <= q.scale * 1.5


def test_quantized_linear_forward():
    in_dim, out_dim = 16, 32
    w = np.random.randn(out_dim, in_dim).astype(np.float32)
    x = np.random.randn(4, in_dim).astype(np.float32)

    # Reference unquantized multiplication
    ref_out = np.matmul(x, w.T)

    # FP16 Quantized Linear
    linear_fp16 = QuantizedLinear(in_dim, out_dim, w, precision="fp16")
    out_fp16 = linear_fp16(x)
    np.testing.assert_allclose(ref_out, out_fp16, atol=1e-2, rtol=1e-2)

    # INT8 Quantized Linear
    linear_int8 = QuantizedLinear(in_dim, out_dim, w, precision="int8")
    out_int8 = linear_int8(x)
    # Cosine similarity between reference and INT8 output should be > 0.99
    cos_sim = np.dot(ref_out.flatten(), out_int8.flatten()) / (
        np.linalg.norm(ref_out.flatten()) * np.linalg.norm(out_int8.flatten())
    )
    assert cos_sim > 0.98


def test_quantized_transformer_block_memory_savings():
    block_fp32 = QuantizedTransformerBlock(layer_idx=0, hidden_dim=64, precision="fp32")
    block_int8 = QuantizedTransformerBlock(layer_idx=0, hidden_dim=64, precision="int8")

    mem_fp32 = block_fp32.memory_footprint_bytes()
    mem_int8 = block_int8.memory_footprint_bytes()

    assert mem_int8["quantized_bytes"] < mem_fp32["quantized_bytes"]
    assert mem_int8["savings_percent"] > 70.0  # >70% memory reduction for INT8
    assert mem_int8["compression_ratio"] >= 3.5

    x = np.random.randn(2, 8, 64).astype(np.float32)
    out, presents = block_int8(x, use_cache=True)
    assert out.shape == x.shape
    assert presents is not None

