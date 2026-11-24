"""
Integration tests for inference job cancellation, request history,
Prometheus metrics exporter, and cluster reset endpoints.
Corresponds to Section 9: API & Operational Reliability.
Maintained by Member 4 (Full-stack, Integration & DevOps Lead).
"""
import pytest
from backend.api.routes import (
    generate_inference,
    cancel_inference,
    get_inference_history,
    get_prometheus_metrics,
    reset_cluster,
    cluster,
    GenerateRequest,
)


def test_job_cancellation_flow():
    # Submit a job
    req = GenerateRequest(prompt="Test cancellation prompt", max_new_tokens=15)
    gen_res = generate_inference(req)
    job_id = gen_res["job_id"]
    assert gen_res["status"] == "queued"

    # Cancel the submitted job
    cancel_res = cancel_inference(job_id)
    assert cancel_res["status"] == "cancelled"
    assert cancel_res["job_id"] == job_id
    assert "cancelled and resources released" in cancel_res["message"]

    # Verify job is tracked in cancelled_jobs set
    assert job_id in cluster.cancelled_jobs


def test_inference_history_tracking():
    # Pre-populate dummy history item if empty
    cluster.request_history.append({
        "job_id": "test-hist-job-1",
        "prompt": "Explain distributed memory grids...",
        "status": "completed",
        "model_id": "distilgpt2",
        "tokens_generated": 20,
        "ttft_seconds": 0.042,
        "tokens_per_second": 35.8,
        "total_latency_seconds": 0.58,
        "timestamp": 1700000000.0,
    })

    res = get_inference_history(limit=10)
    assert "history" in res
    assert "total_count" in res
    assert res["total_count"] >= 1
    recent_jobs = [h["job_id"] for h in res["history"]]
    assert "test-hist-job-1" in recent_jobs


def test_prometheus_metrics_exposition():
    response = get_prometheus_metrics()
    assert response.media_type == "text/plain; version=0.0.4"
    content = response.body.decode("utf-8") if isinstance(response.body, bytes) else str(response.content)

    # Validate Prometheus exposition standard format
    assert "# HELP dmg_active_workers" in content
    assert "# TYPE dmg_active_workers gauge" in content
    assert "dmg_active_workers" in content

    assert "# HELP dmg_queue_size" in content
    assert "# TYPE dmg_queue_size gauge" in content
    assert "dmg_queue_size" in content

    assert "# HELP dmg_processed_requests_total" in content
    assert "# TYPE dmg_processed_requests_total counter" in content

    assert "# HELP dmg_avg_ttft_seconds" in content
    assert "# HELP dmg_avg_tokens_per_second" in content
    assert "# HELP dmg_kv_cache_memory_bytes" in content


def test_cluster_admin_reset():
    # Submit request so state exists
    req = GenerateRequest(prompt="Admin reset test", max_new_tokens=5)
    res = generate_inference(req)
    job_id = res["job_id"]

    reset_res = reset_cluster()
    assert reset_res["status"] == "ok"
    assert "successfully reset" in reset_res["message"]

    # Check requests_map was cleared
    assert len(cluster.requests_map) == 0
    assert len(cluster.cancelled_jobs) == 0
