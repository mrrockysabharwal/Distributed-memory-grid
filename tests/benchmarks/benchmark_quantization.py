"""
Benchmark: Model parameter memory footprint, activation transfer bandwidth,
and numerical logit fidelity across FP32, FP16, and INT8 precision modes.

Maintained by Member 2 (AI / Model Engineer).
"""
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from backend.inference.quantization import Quantizer, QuantizedTransformerBlock
from backend.inference.validation import compute_numerical_discrepancy
from backend.distributed.pipeline_parallel import serialize_tensor, deserialize_tensor


def benchmark_weight_memory_footprint():
    print("=" * 72)
    print("BENCHMARK 1: Model Weight Memory Footprint (FP32 vs FP16 vs INT8)")
    print("=" * 72)

    hidden_dim = 768  # Standard GPT-2 / DistilGPT-2 dimension
    num_layers = 12

    precisions = ["fp32", "fp16", "int8"]
    results = {}

    print(f"{'Precision':<12} | {'Total RAM':<15} | {'Compression':<15} | {'Memory Savings'}")
    print("-" * 72)

    for prec in precisions:
        total_bytes = 0
        orig_bytes = 0
        for layer_i in range(num_layers):
            block = QuantizedTransformerBlock(layer_idx=layer_i, hidden_dim=hidden_dim, precision=prec)
            stats = block.memory_footprint_bytes()
            total_bytes += stats["quantized_bytes"]
            orig_bytes += stats["uncompressed_bytes"]

        ram_mb = total_bytes / (1024 * 1024)
        comp_ratio = orig_bytes / total_bytes if total_bytes > 0 else 1.0
        savings = (1.0 - (total_bytes / orig_bytes)) * 100.0 if orig_bytes > 0 else 0.0
        results[prec] = {"mb": ram_mb, "ratio": comp_ratio, "savings": savings}

        print(f"{prec.upper():<12} | {ram_mb:<15.2f} MB | {comp_ratio:<15.2f}x | {savings:.1f}%")

    print("-" * 72 + "\n")
    return results


def benchmark_activation_transfer_latency():
    print("=" * 72)
    print("BENCHMARK 2: Inter-Node Activation Serialization & Payload Size")
    print("=" * 72)

    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    activation = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

    modes = [("Uncompressed (FP32)", None), ("Compressed (FP16)", "fp16"), ("Dynamic Quantized (INT8)", "int8")]

    print(f"{'Mode':<26} | {'Payload Size':<15} | {'Serialize':<12} | {'Deserialize'}")
    print("-" * 72)

    for label, comp_mode in modes:
        t0 = time.perf_counter()
        raw_bytes, shape, dtype = serialize_tensor(activation, compression=comp_mode)
        ser_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        reconstructed = deserialize_tensor(raw_bytes, decompress_to_fp32=True)
        deser_time_ms = (time.perf_counter() - t1) * 1000.0

        payload_kb = len(raw_bytes) / 1024.0
        print(f"{label:<26} | {payload_kb:<12.1f} KB | {ser_time_ms:<9.2f} ms | {deser_time_ms:<9.2f} ms")

    print("-" * 72 + "\n")


def benchmark_numerical_fidelity():
    print("=" * 72)
    print("BENCHMARK 3: Numerical Logit Fidelity vs FP32 Reference Baseline")
    print("=" * 72)

    # Simulate final layer logits across a vocabulary of 50,257 tokens
    np.random.seed(42)
    ref_logits = np.random.randn(1, 50257).astype(np.float32)
    ref_logits[0, 1024] = 8.5  # Distinct top token

    print(f"{'Precision Mode':<20} | {'Cosine Sim':<15} | {'Max Abs Error':<16} | {'Top-1 Match'}")
    print("-" * 72)

    for prec in ["fp16", "int8"]:
        q = Quantizer.quantize_tensor(ref_logits, target_precision=prec)
        dequant = q.dequantize()
        metrics = compute_numerical_discrepancy(ref_logits, dequant)

        top1_status = "MATCH (100%)" if metrics["top1_concordance"] else "MISMATCH"
        print(
            f"{prec.upper():<20} | {metrics['cosine_similarity']:<15.5f} | "
            f"{metrics['max_absolute_error']:<16.5f} | {top1_status}"
        )

    print("-" * 72 + "\n")


if __name__ == "__main__":
    benchmark_weight_memory_footprint()
    benchmark_activation_transfer_latency()
    benchmark_numerical_fidelity()
