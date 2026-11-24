"""
Unit tests for FastAPI Pydantic request/response schemas.
Maintained by Member 4 (Full-stack, Integration & DevOps Lead).
"""
import pytest
from pydantic import ValidationError
from backend.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    CancelResponse,
    HistoryItem,
    HistoryResponse,
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


def test_cancel_response_schema():
    resp = CancelResponse(
        job_id="job-123",
        status="cancelled",
        message="Job cancelled successfully",
    )
    assert resp.job_id == "job-123"
    assert resp.status == "cancelled"
    assert "successfully" in resp.message


def test_history_schemas():
    item = HistoryItem(
        job_id="job-999",
        prompt="Test prompt",
        status="completed",
        model_id="distilgpt2",
        tokens_generated=25,
        ttft_seconds=0.035,
        tokens_per_second=42.5,
        total_latency_seconds=0.62,
        timestamp=1700000000.0,
    )
    assert item.job_id == "job-999"
    assert item.tokens_generated == 25

    hist_resp = HistoryResponse(history=[item], total_count=1)
    assert hist_resp.total_count == 1
    assert len(hist_resp.history) == 1
    assert hist_resp.history[0].status == "completed"


def test_metrics_response_schema():
    resp = MetricsResponse(
        ttft_seconds_avg=0.045,
        tokens_per_second_avg=38.2,
        total_processed_requests=10,
        kv_cache_memory_mb=128.5,
        kv_cache_hit_rate=0.85,
    )
    assert resp.total_processed_requests == 10
    assert resp.kv_cache_hit_rate == 0.85

