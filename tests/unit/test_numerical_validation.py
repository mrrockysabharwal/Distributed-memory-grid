"""
Unit tests for numerical logit equivalence validation and compressed activation transfer.
Maintained by Member 2 (AI / Model Engineer).
"""
import pytest
import numpy as np

from backend.inference.validation import compute_numerical_discrepancy, ModelValidator
from backend.inference.quantization import Quantizer
from backend.distributed.pipeline_parallel import (
    serialize_tensor,
    deserialize_tensor,
    PipelineStage,
    PipelineParallelEngine,
)
from backend.worker.executor import PartitionExecutor


def test_compute_numerical_discrepancy_exact():
    logits = np.array([[1.0, 5.0, 2.0], [0.5, 3.2, 8.1]], dtype=np.float32)
    metrics = compute_numerical_discrepancy(logits, logits)

    assert metrics["max_absolute_error"] == 0.0
    assert metrics["mean_absolute_error"] == 0.0
    assert metrics["cosine_similarity"] == 1.0
    assert metrics["is_within_tolerance"] is True
    assert metrics["top1_concordance"] is True
    assert metrics["top5_concordance_rate"] == 1.0


def test_compute_numerical_discrepancy_quantized():
    ref = np.random.randn(1, 100).astype(np.float32)
    ref[0, 42] = 10.0  # Clear top token

    # Quantize to INT8 and dequantize
    q = Quantizer.quantize_tensor(ref, target_precision="int8")
    test_dequant = q.dequantize()

    metrics = compute_numerical_discrepancy(ref, test_dequant)
    assert metrics["cosine_similarity"] > 0.99
    assert metrics["top1_concordance"] is True
    assert metrics["ref_top1_token"] == 42
    assert metrics["test_top1_token"] == 42


def test_compressed_activation_serde_fp16():
    activation = np.random.randn(2, 16, 64).astype(np.float32)

    raw_bytes, shape, dtype = serialize_tensor(activation, compression=None)
    fp16_bytes, fp16_shape, fp16_dtype = serialize_tensor(activation, compression="fp16")

    # Compressed FP16 payload should be roughly half the size of FP32
    assert len(fp16_bytes) < len(raw_bytes) * 0.6
    assert shape == fp16_shape

    # Decompress back
    reconstructed = deserialize_tensor(fp16_bytes, decompress_to_fp32=True)
    np.testing.assert_allclose(activation, reconstructed, atol=1e-3, rtol=1e-2)


def test_compressed_activation_serde_int8():
    activation = np.random.randn(2, 16, 64).astype(np.float32)

    raw_bytes, shape, dtype = serialize_tensor(activation, compression=None)
    int8_bytes, int8_shape, int8_dtype = serialize_tensor(activation, compression="int8")

    # INT8 payload should achieve ~3-4x compression
    assert len(int8_bytes) < len(raw_bytes) * 0.35
    assert int8_dtype == "int8_scaled"

    # Decompress back
    reconstructed = deserialize_tensor(int8_bytes, decompress_to_fp32=True)
    cos_sim = np.dot(activation.flatten(), reconstructed.flatten()) / (
        np.linalg.norm(activation.flatten()) * np.linalg.norm(reconstructed.flatten())
    )
    assert cos_sim > 0.99


def test_model_validator_deterministic_equivalence():
    # Construct a 2-stage pipeline
    exec_1 = PartitionExecutor(start_layer=0, end_layer=2)
    exec_2 = PartitionExecutor(start_layer=3, end_layer=5)

    stages = [
        PipelineStage(stage_id=0, worker_id="w1", start_layer=0, end_layer=2, executor=exec_1),
        PipelineStage(stage_id=1, worker_id="w2", start_layer=3, end_layer=5, executor=exec_2),
    ]
    pipe_engine = PipelineParallelEngine(stages=stages)

    # Reference single-node forward function
    def single_node_forward(input_ids, step=0):
        return pipe_engine.forward_sync(input_ids, request_id="ref_job", is_prefill=(step == 0))

    # Distributed forward function
    def distributed_forward(input_ids, step=0):
        return pipe_engine.forward_sync(input_ids, request_id="dist_job", is_prefill=(step == 0))

    validator = ModelValidator(
        single_node_forward_fn=single_node_forward,
        distributed_forward_fn=distributed_forward,
    )

    input_ids = np.array([[10, 20, 30]])
    result = validator.validate_deterministic_equivalence(input_ids, steps=3, atol=1e-4)

    assert result["status"] == "PASSED"
    assert result["average_cosine_similarity"] == 1.0
    assert result["top1_concordance_rate"] == 1.0
    assert result["total_steps_verified"] == 3

