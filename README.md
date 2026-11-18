# Distributed Memory Grid for Real-Time LLM Inference

A lightweight, modular distributed inference engine built with **PyTorch**, **gRPC**, **FastAPI**, and **React**. The project partitions Large Language Models (LLMs) across multiple worker nodes to enable inference without requiring any single node to fit the full model weights into memory.

---

## 1. Core Architecture

The system decouples control coordination from heavy activation transfer:

```
[ User Browser (React + Vite) ]
             |
             v  (REST / WebSocket Streaming)
[ FastAPI API Gateway ]
             |
             v  (Async Queue & State)
[ Master Controller / Scheduler ]
     |              |              |  (gRPC Control Plane)
     v              v              v
[ Worker 1 ] ----> [ Worker 2 ] ----> [ Worker 3 ]
(Layers 0-3)       (Layers 4-7)       (Layers 8-11)
      \----------------- Data Plane (Tensor Activations)
```

- **Control Plane**: gRPC for worker registration, periodic heartbeats, memory/CPU telemetry, model partition assignment, and failure detection.
- **Data Plane**: Inter-node serialized tensor activations passed stage-by-stage across worker processes.
- **Inference Engine**: Autoregressive decoding loop, KV-cache tracking, token sampling, and WebSocket streaming.

---

## 2. Directory Layout

```text
distributed-memory-grid/
├── backend/
│   ├── master/          # Worker registry, scheduler, cluster coordinator
│   ├── worker/          # Worker node daemon and layer executor
│   ├── distributed/     # Tensor serialization and inter-node communicator
│   ├── inference/       # HF model slicing, autoregressive generation, KV cache
│   ├── api/             # FastAPI gateway and WebSocket streaming routes
│   └── monitoring/      # psutil/NVML telemetry, TTFT and tokens/sec metrics
├── frontend/            # React + Vite + Tailwind dashboard & chat console
├── proto/               # gRPC protobuf service definitions (cluster.proto)
├── docker/              # Dockerfile definitions for master and worker containers
├── docs/                # Architecture specifications and benchmark reports
├── tests/               # Unit, integration, and benchmark tests
├── docker-compose.yml   # Multi-worker local cluster orchestration
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 3. Four-Member Team Ownership

| Member | Role | Key Ownership | Concrete Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Distributed Systems Lead** | Master controller, gRPC, worker discovery, heartbeats, lifecycle | `backend/master/`, `proto/`, worker registry & health tests |
| **Member 2** | **AI / Model Engineer** | PyTorch model partitioning, stage execution, forward pass correctness | `backend/distributed/`, `backend/worker/executor.py`, model tests |
| **Member 3** | **Inference Optimization Engineer** | KV cache, request queue, continuous batching, memory benchmarks | `backend/inference/kv_cache.py`, `scheduler.py`, benchmarks |
| **Member 4** | **Full-Stack & DevOps Lead** | FastAPI gateway, React chat/dashboard, Docker Compose, metrics | `backend/api/`, `frontend/`, `docker/`, telemetry & reports |

---

## 4. Four-Week Milestone Roadmap

- **Week 1 (Foundation)**: Master-worker gRPC registration and heartbeats. Single-node model baseline and layer slicing. FastAPI & React shells.
  - *Acceptance Gate*: Two workers register with master and report health; model runs on single node.
- **Week 2 (Cross-Node Execution)**: Worker dispatch, sequential cross-node layer forward passes. Basic request queue and memory monitoring.
  - *Acceptance Gate*: Model stages execute across $\ge 2$ worker processes with matching deterministic output.
- **Week 3 (Streaming & Scheduling)**: Autoregressive token generation loop, streaming to frontend over WebSockets, continuous batching experiments.
  - *Acceptance Gate*: Prompts submitted from web UI, tokens stream in real time, requests scheduled.
- **Week 4 (Benchmarks & Delivery)**: Stress testing, worker failure handling/retry, TTFT & tokens/sec benchmarks, Docker reproducibility, final demo.
  - *Acceptance Gate*: Reproducible cluster deployment, benchmark graphs, tested fault-tolerance, and final demonstration.

---

## 5. API Contracts

### REST & WebSocket Gateway
- `POST /api/inference/generate` - Submit prompt for generation
- `GET /api/workers` - List registered workers, assigned partitions, and health
- `GET /api/cluster/status` - Cluster health and master node telemetry
- `POST /api/models/load` - Trigger model weight download and partition loading
- `GET /api/metrics` - TTFT, tokens/sec, and per-worker memory metrics
- `WS /ws/inference/{job_id}` - Real-time token streaming channel

### gRPC Control & Data RPCs
- `RegisterWorker` / `Heartbeat` - Worker discovery, capability advertisement, liveness
- `LoadPartition` - Dynamic layer assignment (e.g. layers 0-3 on Worker A)
- `ExecutePartition` - Forward pass of tensor activations through assigned layers
- `ClearCache` - Eviction of per-request KV cache

---

## 6. Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional, for containerized run)

### 2. Python Environment Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Compile Protocol Buffers
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/cluster.proto
```

### 4. Run the API Gateway & Master
```bash
python -m backend.api.main
```

### 5. Launch Worker Nodes
```bash
# Terminal 1: Worker 1
python -m backend.worker.worker_node --worker-id worker-1 --port 50052

# Terminal 2: Worker 2
python -m backend.worker.worker_node --worker-id worker-2 --port 50053
```

### 6. Run via Docker Compose
```bash
docker compose up --build
```

