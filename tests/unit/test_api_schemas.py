"""
Unit tests for FastAPI Pydantic request/response schemas.
Maintained by Member 4 (Full-stack, Integration & DevOps Lead).
"""
import pytest
from pydantic import ValidationError
from backend.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    WorkerItem,
    ClusterStatusResponse,
    ModelLoadRequest,
    MetricsResponse,
)


def test_generate_request_validation():
    # Valid payload
    req = GenerateRequest(prompt="Hello", max_new_tokens=25, temperature=0.8, priority=0)
    assert req.prompt == "Hello"
    assert req.max_new_tokens == 25
    assert req.temperature == 0.8
    assert req.priority == 0
    assert req.model_id == "distilgpt2"

    # Empty prompt should fail
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")

    # Negative max_new_tokens should fail
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Valid", max_new_tokens=-5)

    # Temperature > 2.0 should fail
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Valid", temperature=3.5)

    # Priority outside 0-2 should fail
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Valid", priority=5)


def test_worker_item_schema():
    item = WorkerItem(
        worker_id="w-1",
        address="127.0.0.1:50052",
        device_type="cpu",
        available_memory_mb=4096.0,
        model_partition="layers 0-5",
        status="online",
    )
    assert item.worker_id == "w-1"
    assert item.status == "online"
    assert item.cpu_utilization == 0.0


def test_model_load_request():
    load_req = ModelLoadRequest(model_id="gpt2-small", total_layers=12)
    assert load_req.model_id == "gpt2-small"
    assert load_req.total_layers == 12

    with pytest.raises(ValidationError):
        ModelLoadRequest(total_layers=0)
