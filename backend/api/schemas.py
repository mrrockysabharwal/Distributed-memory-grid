"""
Pydantic API request and response schemas for Distributed Memory Grid.
Corresponds to Section 9: API and Communication Contracts.

Maintained by Member 4 (Full-stack, Integration & DevOps Lead).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Inference job submission request."""
    model_id: str = Field(default="distilgpt2", description="Pretrained model identifier")
    prompt: str = Field(..., min_length=1, description="Input text prompt for inference")
    max_new_tokens: int = Field(default=20, ge=1, le=512, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=50, ge=0, description="Top-k filtering limit")
    priority: int = Field(default=1, ge=0, le=2, description="Request priority: 0=High, 1=Normal, 2=Low")
    request_id: Optional[str] = Field(default=None, description="Optional caller-supplied unique job ID")


class GenerateResponse(BaseModel):
    """Initial response returned upon queuing an inference job."""
    job_id: str
    status: str
    model_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    queue_position: int
    websocket_url: str


class WorkerItem(BaseModel):
    """Worker node status and assigned model partition details."""
    worker_id: str
    address: str
    device_type: str
    available_memory_mb: float
    model_partition: str
    status: str
    cpu_utilization: float = 0.0


class WorkersListResponse(BaseModel):
    """List of all registered worker nodes in the cluster."""
    workers: List[WorkerItem]
    count: int


class ClusterStatusResponse(BaseModel):
    """Cluster health summary and master node telemetry."""
    cluster_state: str
    active_workers: int
    queue_size: int
    total_active_requests: int
    kv_cache: Dict[str, Any]
    master_telemetry: Dict[str, Any]


class ModelLoadRequest(BaseModel):
    """Request to download and slice model layers across workers."""
    model_id: str = Field(default="distilgpt2", description="Target model ID")
    total_layers: int = Field(default=12, ge=1, le=128, description="Number of transformer layers to partition")
    strategy: str = Field(default="pipeline_layer", description="Partitioning strategy")


class ModelLoadResponse(BaseModel):
    """Result of partitioning model across active workers."""
    status: str
    message: str
    model_id: str
    partition_assignments: Dict[str, List[int]]


class MetricsResponse(BaseModel):
    """Aggregated latency, throughput, and memory performance metrics."""
    ttft_seconds_avg: float
    tokens_per_second_avg: float
    total_processed_requests: int
    kv_cache_memory_mb: float
    kv_cache_hit_rate: float
