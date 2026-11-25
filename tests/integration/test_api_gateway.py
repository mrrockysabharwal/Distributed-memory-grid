"""
Integration tests for FastAPI gateway route handlers.
Tests REST endpoints directly.
"""
from backend.api.routes import (
    generate_inference,
    list_workers,
    cluster_status,
    load_model,
    get_metrics,
    GenerateRequest,
    ModelLoadRequest,
)


def test_api_cluster_status():
    data = cluster_status()
    assert "cluster_state" in data
    assert data["cluster_state"] == "healthy"
    assert data["active_workers"] >= 2
    assert "master_telemetry" in data


def test_api_list_workers():
    data = list_workers()
    assert "workers" in data
    assert len(data["workers"]) >= 2
    assert data["workers"][0]["status"] == "online"


def test_api_load_model():
    req = ModelLoadRequest(model_id="distilgpt2", total_layers=6, precision="int8")
    res = load_model(req)
    assert res["status"] == "success"
    assert "partition_assignments" in res
    assert res["precision"] == "int8"
    assert res["memory_savings_percent"] == 75.0
    assert res["compression_ratio"] == 4.0
    assignments = res["partition_assignments"]
    assert "worker-1" in assignments
    assert "worker-2" in assignments


def test_api_inference_generate_and_metrics():
    req = GenerateRequest(prompt="Distributed memory inference", max_new_tokens=5)
    res = generate_inference(req)
    assert res["status"] == "queued"
    assert "job_id" in res
    assert res["prompt"] == "Distributed memory inference"

    metrics = get_metrics()
    assert "ttft_seconds_avg" in metrics
    assert "tokens_per_second_avg" in metrics
    assert "kv_cache_memory_mb" in metrics
