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


class CancelResponse(BaseModel):
    """Response returned upon cancelling an inference job."""
    job_id: str
    status: str
    message: str


class HistoryItem(BaseModel):
    """Record of a processed inference request."""
    job_id: str
    prompt: str
    status: str
    model_id: str
    tokens_generated: int
    ttft_seconds: float
    tokens_per_second: float
    total_latency_seconds: float
    timestamp: float


class HistoryResponse(BaseModel):
    """List of recent inference requests."""
    history: List[HistoryItem]
    total_count: int


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
    precision: str = Field(default="fp32", description="Model weight precision: fp32, fp16, int8")
    activation_compression: Optional[str] = Field(default=None, description="Inter-node activation compression: fp16, int8, or None")


class ModelLoadResponse(BaseModel):
    """Result of partitioning model across active workers."""
    status: str
    message: str
    model_id: str
    partition_assignments: Dict[str, List[int]]
    precision: str = "fp32"
    memory_savings_percent: float = 0.0
    compression_ratio: float = 1.0


class MetricsResponse(BaseModel):
    """Aggregated latency, throughput, and memory performance metrics."""
    ttft_seconds_avg: float
    tokens_per_second_avg: float
    total_processed_requests: int
    kv_cache_memory_mb: float
    kv_cache_hit_rate: float


# --- Cluster Auth Schemas ---

class JoinTokenRequest(BaseModel):
    """Request to create a new worker join token."""
    ttl_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0, description="Token time-to-live in seconds")


class JoinTokenResponse(BaseModel):
    """Response containing the generated join token."""
    token: str
    expires_in_seconds: float
    message: str


class WorkerAuthRequest(BaseModel):
    """Worker authentication payload for secure cluster joining."""
    worker_id: str = Field(..., description="Unique worker identifier")
    token: str = Field(..., description="Join token issued by admin")
    signature: str = Field(..., description="HMAC-SHA256 signature")
    timestamp: float = Field(..., description="Unix timestamp of the request")
    address: str = Field(default="unknown", description="Worker network address")
    device_type: str = Field(default="cpu", description="Worker device type")
    total_memory: int = Field(default=8589934592, description="Total memory in bytes")
    available_memory: int = Field(default=6442450944, description="Available memory in bytes")


class AuthAuditResponse(BaseModel):
    """Audit log of authentication events."""
    events: List[Dict[str, Any]]
    total_count: int


# --- Retry / Checkpoint Schemas ---

class RetryStatsResponse(BaseModel):
    """Job retry system statistics."""
    total_tracked_jobs: int
    total_retries: int
    total_recoveries: int
    total_dead_lettered: int
    active_checkpoints: int
    dead_letter_queue_size: int


class DeadLetterItem(BaseModel):
    """A dead-lettered inference job."""
    request_id: str
    total_attempts: int
    last_reason: str
    last_error: str
    dead_lettered_at: float


# --- Prefix Cache Schemas ---

class PrefixCacheStatsResponse(BaseModel):
    """Prefix KV-cache reuse statistics."""
    prefix_cache_hits: int
    prefix_cache_misses: int
    hit_rate: float
    total_tokens_saved: int
    stored_entries: int
