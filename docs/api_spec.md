# Distributed Memory Grid (DMG) - API & Communication Specification

This document details the formal contracts, data models, endpoints, and streaming protocols for the Distributed Memory Grid inference platform.

---

## 1. REST API Gateway Endpoints

Base URL: `http://localhost:8000/api`

### 1.1 `POST /inference/generate`
Submit a prompt to be queued and scheduled across the distributed cluster.

**Request Body:**
```json
{
  "model_id": "distilgpt2",
  "prompt": "Distributed memory systems scale LLM inference",
  "max_new_tokens": 25,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "priority": 1,
  "request_id": "optional-custom-uuid"
}
```

**Response (`200 OK`):**
```json
{
  "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
  "status": "queued",
  "model_id": "distilgpt2",
  "prompt": "Distributed memory systems scale LLM inference",
  "max_new_tokens": 25,
  "temperature": 0.7,
  "queue_position": 1,
  "websocket_url": "/ws/inference/8fa8121f-8ec9-43d8-ba90-50bfa434a9ef"
}
```

---

### 1.2 `POST /inference/cancel/{job_id}`
Cancel an in-flight or queued inference job, terminating token generation and freeing associated KV-cache memory blocks.

**Response (`200 OK`):**
```json
{
  "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
  "status": "cancelled",
  "message": "Inference job 8fa8121f-8ec9-43d8-ba90-50bfa434a9ef cancelled and resources released."
}
```

---

### 1.3 `GET /inference/history`
Retrieve execution logs of recently completed, failed, or cancelled jobs.

**Query Parameters:**
- `limit` (int, default: 20): Number of recent entries to retrieve.

**Response (`200 OK`):**
```json
{
  "history": [
    {
      "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
      "prompt": "Distributed memory systems scale LLM inference...",
      "status": "completed",
      "model_id": "distilgpt2",
      "tokens_generated": 25,
      "ttft_seconds": 0.042,
      "tokens_per_second": 38.5,
      "total_latency_seconds": 0.65,
      "timestamp": 1700850000.0
    }
  ],
  "total_count": 1
}
```

---

### 1.4 `GET /workers`
List all registered worker processes, their advertised hardware specifications, assigned layer partitions, and live statuses.

**Response (`200 OK`):**
```json
{
  "workers": [
    {
      "worker_id": "worker-1",
      "address": "127.0.0.1:50052",
      "device_type": "cpu",
      "available_memory_mb": 4096.0,
      "model_partition": "layers 0-5",
      "status": "online",
      "cpu_utilization": 12.4
    },
    {
      "worker_id": "worker-2",
      "address": "127.0.0.1:50053",
      "device_type": "cpu",
      "available_memory_mb": 4096.0,
      "model_partition": "layers 6-11",
      "status": "online",
      "cpu_utilization": 14.1
    }
  ],
  "count": 2
}
```

---

### 1.5 `GET /cluster/status`
Aggregate health summary of master scheduler, queue sizes, active requests, and node resources.

**Response (`200 OK`):**
```json
{
  "cluster_state": "healthy",
  "active_workers": 2,
  "queue_size": 0,
  "total_active_requests": 0,
  "kv_cache": {
    "active_requests_in_cache": 0,
    "max_capacity": 100,
    "total_memory_bytes": 0,
    "total_memory_mb": 0.0,
    "cache_hits": 45,
    "cache_misses": 5,
    "hit_rate": 0.9000
  },
  "master_telemetry": {
    "cpu_percent": 15.2,
    "ram_used_mb": 245.8,
    "ram_total_mb": 16384.0,
    "ram_percent": 38.4
  }
}
```

---

### 1.6 `POST /models/load`
Trigger layer partitioning and partition distribution across available worker nodes.

**Request Body:**
```json
{
  "model_id": "distilgpt2",
  "total_layers": 12,
  "strategy": "pipeline_layer"
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "message": "Model 'distilgpt2' partitioned across 2 workers.",
  "model_id": "distilgpt2",
  "partition_assignments": {
    "worker-1": [0, 1, 2, 3, 4, 5],
    "worker-2": [6, 7, 8, 9, 10, 11]
  }
}
```

---

### 1.7 `GET /metrics`
Aggregated inference performance metrics across all completed queries.

**Response (`200 OK`):**
```json
{
  "ttft_seconds_avg": 0.0385,
  "tokens_per_second_avg": 42.18,
  "total_processed_requests": 50,
  "kv_cache_memory_mb": 4.12,
  "kv_cache_hit_rate": 0.88
}
```

---

### 1.8 `GET /metrics/prometheus`
Standard Prometheus exposition format (`text/plain; version=0.0.4`) exposing cluster gauges and counters for Grafana or Datadog ingestion.

**Sample Output:**
```text
# HELP dmg_active_workers Number of active distributed worker nodes.
# TYPE dmg_active_workers gauge
dmg_active_workers 2
# HELP dmg_queue_size Pending inference requests in scheduler.
# TYPE dmg_queue_size gauge
dmg_queue_size 0
# HELP dmg_processed_requests_total Total completed inference requests.
# TYPE dmg_processed_requests_total counter
dmg_processed_requests_total 50
# HELP dmg_avg_ttft_seconds Average time to first token in seconds.
# TYPE dmg_avg_ttft_seconds gauge
dmg_avg_ttft_seconds 0.0385
# HELP dmg_avg_tokens_per_second Average generated tokens per second.
# TYPE dmg_avg_tokens_per_second gauge
dmg_avg_tokens_per_second 42.18
# HELP dmg_kv_cache_memory_bytes Total memory occupied by KV cache.
# TYPE dmg_kv_cache_memory_bytes gauge
dmg_kv_cache_memory_bytes 4320144
```

---

### 1.9 `POST /admin/cluster/reset`
Administrator endpoint to evict all active KV-cache allocations, clear pending request queues, and reset internal metric counters.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "message": "Cluster cache and state successfully reset."
}
```

---

## 2. WebSocket Real-Time Streaming Channel

Endpoint: `ws://localhost:8000/ws/inference/{job_id}`

### Event Lifecycle

1. **Processing Initiation:**
```json
{
  "event": "status",
  "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
  "status": "processing",
  "prompt": "Distributed memory systems..."
}
```

2. **Incremental Token Yields:**
```json
{
  "event": "token",
  "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
  "token": " scale",
  "step": 1
}
```

3. **User Abort / Cancellation Event:**
```json
{
  "event": "cancelled",
  "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
  "message": "Generation stopped by user."
}
```

4. **Generation Completion:**
```json
{
  "event": "done",
  "job_id": "8fa8121f-8ec9-43d8-ba90-50bfa434a9ef",
  "tokens_generated": 25,
  "ttft_seconds": 0.0385,
  "tokens_per_second": 42.18,
  "total_latency_seconds": 0.5927
}
```

---

## 3. gRPC Control & Data Plane Definitions

Defined in `proto/cluster.proto`:

```protobuf
syntax = "proto3";
package cluster;

service ClusterManager {
  rpc RegisterWorker (RegisterRequest) returns (RegisterResponse);
  rpc SendHeartbeat (HeartbeatRequest) returns (HeartbeatResponse);
}

service WorkerService {
  rpc LoadPartition (LoadPartitionRequest) returns (LoadPartitionResponse);
  rpc ExecutePartition (ExecutePartitionRequest) returns (ExecutePartitionResponse);
  rpc ClearCache (ClearCacheRequest) returns (ClearCacheResponse);
}
```
