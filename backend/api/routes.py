"""
FastAPI route definitions for DMG.
Corresponds to Section 9: API and Communication Contracts.
Maintained by Member 4 (Full-stack, Integration & DevOps Lead).
"""
import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.monitoring.telemetry import get_system_telemetry
from backend.services.job_service import create_job, get_job

router = APIRouter()


class GenerateRequest(BaseModel):
    model_id: str = "distilgpt2"
    prompt: str
    max_new_tokens: int = Field(default=50, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    request_id: Optional[str] = None


class ModelLoadRequest(BaseModel):
    model_id: str = "distilgpt2"
    strategy: str = "pipeline_layer"


@router.post("/inference/generate")
def generate_inference(req: GenerateRequest, db: Session = Depends(get_db)):
    job_id = req.request_id or str(uuid.uuid4())
    job = create_job(
        db,
        job_id=job_id,
        model_id=req.model_id,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "model_id": job.model_id,
        "prompt": job.prompt,
        "max_new_tokens": job.max_new_tokens,
        "temperature": job.temperature,
        "websocket_url": f"/ws/inference/{job.job_id}",
    }


@router.get("/inference/jobs/{job_id}")
def get_inference_job(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Inference job not found")

    return {
        "job_id": job.job_id,
        "model_id": job.model_id,
        "prompt": job.prompt,
        "max_new_tokens": job.max_new_tokens,
        "temperature": job.temperature,
        "status": job.status,
        "output": job.output,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@router.get("/workers")
async def list_workers():
    # Placeholder returning sample worker structure
    return {
        "workers": [
            {
                "worker_id": "worker-1",
                "address": "127.0.0.1:50052",
                "device_type": "cpu",
                "available_memory_mb": 4096,
                "model_partition": "layers 0-2",
                "status": "online",
            },
            {
                "worker_id": "worker-2",
                "address": "127.0.0.1:50053",
                "device_type": "cpu",
                "available_memory_mb": 4096,
                "model_partition": "layers 3-5",
                "status": "online",
            },
        ]
    }


@router.get("/cluster/status")
async def cluster_status():
    telemetry = get_system_telemetry()
    return {
        "cluster_state": "healthy",
        "active_workers": 2,
        "total_active_requests": 0,
        "master_telemetry": telemetry,
    }


@router.post("/models/load")
async def load_model(req: ModelLoadRequest):
    return {
        "status": "success",
        "message": f"Model '{req.model_id}' partitioned and loaded across active cluster workers.",
        "model_id": req.model_id,
    }


@router.get("/metrics")
async def get_metrics():
    return {
        "ttft_seconds_avg": 0.12,
        "tokens_per_second_avg": 24.5,
        "total_processed_requests": 0,
        "per_node_ram_usage_mb": {"worker-1": 850, "worker-2": 820},
    }


@router.websocket("/ws/inference/{job_id}")
async def websocket_inference_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        # Initial status
        await websocket.send_json({"event": "status", "job_id": job_id, "status": "processing"})
        # Stream mock / generated tokens
        demo_tokens = ["Distributed ", "Memory ", "Grid ", "inference ", "is ", "working!"]
        for token in demo_tokens:
            await asyncio.sleep(0.1)
            await websocket.send_json({"event": "token", "job_id": job_id, "token": token})
        await websocket.send_json({"event": "done", "job_id": job_id})
    except WebSocketDisconnect:
        pass

