"""
FastAPI route definitions for Distributed Memory Grid (DMG).
Handles REST client requests, real-time WebSocket token streaming,
job cancellation, history tracking, cluster telemetry, and Prometheus metrics.

Maintained by Member 4 (Full-stack & DevOps Lead) & Member 3 (Inference Optimization Engineer).
"""
import time
import uuid
import asyncio
from typing import Optional, List, Dict, Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Response
from pydantic import BaseModel, Field

from backend.master.registry import WorkerRegistry
from backend.master.scheduler import RequestScheduler, GenerationRequest, RequestPriority
from backend.inference.kv_cache import KVCacheManager
from backend.inference.generation import GenerationEngine, TokenSampler
from backend.distributed.pipeline_parallel import PipelineParallelEngine, PipelineStage
from backend.worker.executor import PartitionExecutor
from backend.monitoring.metrics import MetricsCollector, RequestBenchmark
from backend.monitoring.telemetry import get_system_telemetry

from backend.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    CancelResponse,
    HistoryItem,
    HistoryResponse,
    WorkerItem,
    WorkersListResponse,
    ClusterStatusResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    MetricsResponse,
)

router = APIRouter()


class ClusterState:
    """Singleton holding live master services and cluster runtime state."""
    def __init__(self):
        self.registry = WorkerRegistry()
        self.scheduler = RequestScheduler(max_queue_size=200)
        self.kv_cache_mgr = KVCacheManager(max_cached_requests=100)
        self.metrics_collector = MetricsCollector()
        self.cancelled_jobs: Set[str] = set()
        self.request_history: List[Dict[str, Any]] = []

        # Seed initial default local workers if empty
        self.registry.register(
            worker_id="worker-1",
            address="127.0.0.1:50052",
            device_type="cpu",
            total_memory=8 * 1024 * 1024 * 1024,
            available_memory=6 * 1024 * 1024 * 1024,
        )
        self.registry.register(
            worker_id="worker-2",
            address="127.0.0.1:50053",
            device_type="cpu",
            total_memory=8 * 1024 * 1024 * 1024,
            available_memory=6 * 1024 * 1024 * 1024,
        )

        # Set up default pipeline stages across workers (Layers 0-2 and 3-5)
        self.executor_1 = PartitionExecutor(start_layer=0, end_layer=2, kv_cache_mgr=self.kv_cache_mgr)
        self.executor_2 = PartitionExecutor(start_layer=3, end_layer=5, kv_cache_mgr=self.kv_cache_mgr)

        self.stages = [
            PipelineStage(stage_id=0, worker_id="worker-1", start_layer=0, end_layer=2, executor=self.executor_1),
            PipelineStage(stage_id=1, worker_id="worker-2", start_layer=3, end_layer=5, executor=self.executor_2),
        ]
        self.pipeline_engine = PipelineParallelEngine(stages=self.stages)

        # Mock tokenizer for streaming text generation
        class SimpleTokenizer:
            eos_token_id = 50256
            def encode(self, text: str, **kwargs):
                return [ord(c) % 1000 + 1 for c in text] if text else [1, 2]
            def decode(self, token_ids: List[int]) -> str:
                words = [" distributed", " memory", " grid", " tensor", " pipeline", " attention", " inference", " fast", " stream"]
                return words[token_ids[0] % len(words)]

        self.tokenizer = SimpleTokenizer()
        self.generation_engine = GenerationEngine(
            tokenizer=self.tokenizer,
            pipeline_forward_fn=self.pipeline_engine.forward,
            kv_cache_mgr=self.kv_cache_mgr,
        )
        self.requests_map: Dict[str, GenerationRequest] = {}


# Global cluster singleton
cluster = ClusterState()


@router.post("/inference/generate")
def generate_inference(req: GenerateRequest):
    job_id = req.request_id or str(uuid.uuid4())
    gen_req = GenerationRequest(
        request_id=job_id,
        model_id=req.model_id,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        priority=req.priority,
    )
    cluster.scheduler.submit_request_sync(gen_req)
    cluster.requests_map[job_id] = gen_req

    return {
        "job_id": job_id,
        "status": "queued",
        "model_id": req.model_id,
        "prompt": req.prompt,
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "queue_position": cluster.scheduler.queue_size(),
        "websocket_url": f"/ws/inference/{job_id}",
    }


@router.post("/inference/cancel/{job_id}")
def cancel_inference(job_id: str):
    """Cancel an active or queued generation job."""
    cluster.cancelled_jobs.add(job_id)
    cluster.kv_cache_mgr.clear_request(job_id)
    cluster.scheduler.complete_request(job_id)

    # Update history status if present
    for item in cluster.request_history:
        if item.get("job_id") == job_id:
            item["status"] = "cancelled"

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": f"Inference job {job_id} cancelled and resources released.",
    }


@router.get("/inference/history")
def get_inference_history(limit: int = 20):
    """Retrieve history of recently completed or cancelled inference jobs."""
    recent = list(reversed(cluster.request_history))[:limit]
    return {
        "history": recent,
        "total_count": len(cluster.request_history),
    }


@router.websocket("/ws/inference/{job_id}")
async def websocket_inference_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    benchmark = RequestBenchmark(request_id=job_id)
    tokens_streamed = 0
    status = "completed"

    try:
        req = cluster.requests_map.get(job_id)
        prompt = req.prompt if req else f"Demo prompt for {job_id}"
        max_tokens = req.max_new_tokens if req else 20
        temperature = req.temperature if req else 0.7
        model_id = req.model_id if req else "distilgpt2"

        await websocket.send_json({
            "event": "status",
            "job_id": job_id,
            "status": "processing",
            "prompt": prompt,
        })

        async for token_text in cluster.generation_engine.stream_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            request_id=job_id,
        ):
            # Check for cancellation signal
            if job_id in cluster.cancelled_jobs:
                status = "cancelled"
                await websocket.send_json({
                    "event": "cancelled",
                    "job_id": job_id,
                    "message": "Generation stopped by user.",
                })
                break

            if tokens_streamed == 0:
                benchmark.mark_first_token()
            tokens_streamed += 1

            await websocket.send_json({
                "event": "token",
                "job_id": job_id,
                "token": token_text,
                "step": tokens_streamed,
            })
            await asyncio.sleep(0.02)

        benchmark.mark_end(tokens_streamed)
        cluster.metrics_collector.record(benchmark)

        if status != "cancelled":
            await websocket.send_json({
                "event": "done",
                "job_id": job_id,
                "tokens_generated": tokens_streamed,
                "ttft_seconds": round(benchmark.ttft_seconds, 4),
                "tokens_per_second": round(benchmark.tokens_per_second, 2),
                "total_latency_seconds": round(benchmark.total_latency_seconds, 4),
            })

        # Append to in-memory request history
        cluster.request_history.append({
            "job_id": job_id,
            "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "status": status,
            "model_id": model_id,
            "tokens_generated": tokens_streamed,
            "ttft_seconds": round(benchmark.ttft_seconds, 4),
            "tokens_per_second": round(benchmark.tokens_per_second, 2),
            "total_latency_seconds": round(benchmark.total_latency_seconds, 4),
            "timestamp": time.time(),
        })
        # Keep recent 50
        if len(cluster.request_history) > 50:
            cluster.request_history.pop(0)

    except WebSocketDisconnect:
        pass
    finally:
        cluster.cancelled_jobs.discard(job_id)
        cluster.kv_cache_mgr.clear_request(job_id)
        cluster.scheduler.complete_request(job_id)


@router.get("/workers")
def list_workers():
    active_workers = cluster.registry.get_active_workers()
    workers_out = []
    for w in active_workers:
        partition_desc = f"layers {w.start_layer}-{w.end_layer}" if w.start_layer >= 0 else "unassigned"
        workers_out.append({
            "worker_id": w.worker_id,
            "address": w.address,
            "device_type": w.device_type,
            "available_memory_mb": round(w.available_memory_bytes / (1024 * 1024), 2),
            "model_partition": partition_desc,
            "status": "online" if w.is_active else "offline",
            "cpu_utilization": w.cpu_utilization,
        })
    return {"workers": workers_out, "count": len(workers_out)}


@router.get("/cluster/status")
def cluster_status():
    telemetry = get_system_telemetry()
    active_workers = cluster.registry.get_active_workers()
    cache_stats = cluster.kv_cache_mgr.current_usage_stats()

    return {
        "cluster_state": "healthy" if active_workers else "degraded",
        "active_workers": len(active_workers),
        "queue_size": cluster.scheduler.queue_size(),
        "total_active_requests": cluster.scheduler.active_count(),
        "kv_cache": cache_stats,
        "master_telemetry": telemetry,
    }


@router.post("/models/load")
def load_model(req: ModelLoadRequest):
    active_workers = cluster.registry.get_active_workers()
    if not active_workers:
        raise HTTPException(status_code=400, detail="No active workers available for partitioning.")

    worker_ids = [w.worker_id for w in active_workers]
    assignments = cluster.scheduler.plan_partitions(req.total_layers, worker_ids)
    cluster.registry.assign_partitions(assignments)

    # Configure precision and activation compression
    cluster.pipeline_engine.activation_compression = req.activation_compression
    savings = 75.0 if req.precision.lower() == "int8" else (50.0 if req.precision.lower() == "fp16" else 0.0)
    ratio = 4.0 if req.precision.lower() == "int8" else (2.0 if req.precision.lower() == "fp16" else 1.0)

    return {
        "status": "success",
        "message": f"Model '{req.model_id}' ({req.precision.upper()}) partitioned across {len(worker_ids)} workers.",
        "model_id": req.model_id,
        "partition_assignments": assignments,
        "precision": req.precision.lower(),
        "memory_savings_percent": savings,
        "compression_ratio": ratio,
    }


@router.get("/metrics")
def get_metrics():
    summary = cluster.metrics_collector.summary()
    cache_stats = cluster.kv_cache_mgr.current_usage_stats()

    return {
        "ttft_seconds_avg": summary.get("avg_ttft_seconds", 0.0),
        "tokens_per_second_avg": summary.get("avg_tokens_per_second", 0.0),
        "total_processed_requests": summary.get("total_requests", 0),
        "kv_cache_memory_mb": cache_stats.get("total_memory_mb", 0.0),
        "kv_cache_hit_rate": cache_stats.get("hit_rate", 0.0),
    }


@router.get("/metrics/prometheus")
def get_prometheus_metrics():
    """Prometheus exposition format exporter for cluster observability."""
    summary = cluster.metrics_collector.summary()
    cache_stats = cluster.kv_cache_mgr.current_usage_stats()
    active_workers = len(cluster.registry.get_active_workers())

    lines = [
        "# HELP dmg_active_workers Number of active distributed worker nodes.",
        "# TYPE dmg_active_workers gauge",
        f"dmg_active_workers {active_workers}",
        "# HELP dmg_queue_size Pending inference requests in scheduler.",
        "# TYPE dmg_queue_size gauge",
        f"dmg_queue_size {cluster.scheduler.queue_size()}",
        "# HELP dmg_processed_requests_total Total completed inference requests.",
        "# TYPE dmg_processed_requests_total counter",
        f"dmg_processed_requests_total {summary.get('total_requests', 0)}",
        "# HELP dmg_avg_ttft_seconds Average time to first token in seconds.",
        "# TYPE dmg_avg_ttft_seconds gauge",
        f"dmg_avg_ttft_seconds {summary.get('avg_ttft_seconds', 0.0)}",
        "# HELP dmg_avg_tokens_per_second Average generated tokens per second.",
        "# TYPE dmg_avg_tokens_per_second gauge",
        f"dmg_avg_tokens_per_second {summary.get('avg_tokens_per_second', 0.0)}",
        "# HELP dmg_kv_cache_memory_bytes Total memory occupied by KV cache.",
        "# TYPE dmg_kv_cache_memory_bytes gauge",
        f"dmg_kv_cache_memory_bytes {cache_stats.get('total_memory_bytes', 0)}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@router.post("/admin/cluster/reset")
def reset_cluster():
    """Admin endpoint to clear KV cache and reset metrics."""
    cluster.kv_cache_mgr.clear_all()
    cluster.cancelled_jobs.clear()
    cluster.requests_map.clear()
    return {"status": "ok", "message": "Cluster cache and state successfully reset."}
