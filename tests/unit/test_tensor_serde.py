"""
Unit tests for tensor serialization and deserialization across nodes.
"""
import pytest

try:
    import torch
    from backend.distributed.pipeline_parallel import serialize_tensor, deserialize_tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in current environment")
def test_tensor_serde_roundtrip():
    original = torch.randn(2, 16, 768)
    data, shape, dtype = serialize_tensor(original)
    restored = deserialize_tensor(data, device="cpu")

    assert restored.shape == original.shape
    assert torch.allclose(original, restored)

