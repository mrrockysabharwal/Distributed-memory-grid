# Distributed Memory Grid (DMG) - Deployment & Demonstration Guide

This guide provides instructions for deploying the Distributed Memory Grid locally, running in Docker Compose, and executing the live demonstration workflow specified in Section 14 of the project blueprint.

---

## 1. Prerequisites

- **Python**: 3.11+
- **Node.js**: 18+ (for frontend web dashboard)
- **Docker & Docker Compose**: (Optional, for containerized multi-node orchestration)

---

## 2. Local Multi-Process Deployment

### Terminal 1: Launch Master Controller & API Gateway
```bash
# Activate your environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start FastAPI Gateway and Master Controller on port 8000
python -m backend.api.main
```

### Terminal 2: Launch Worker Node 1
```bash
python -m backend.worker.worker_node --worker-id worker-1 --port 50052 --master-address localhost:50051
```

### Terminal 3: Launch Worker Node 2
```bash
python -m backend.worker.worker_node --worker-id worker-2 --port 50053 --master-address localhost:50051
```

### Terminal 4: Launch React Web Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open your browser to `http://localhost:3000`.

---

## 3. Docker Compose Containerized Deployment

To start the entire cluster (Master, 2 Workers, and Frontend Dashboard) in isolated containers with bridge networking:

```bash
docker compose up --build
```

To shut down:
```bash
docker compose down
```

---

## 4. Live Demonstration Script (Step-by-Step)

Follow this 6-stage walkthrough during the final project evaluation:

### Stage 1: Cluster Health & Worker Discovery
1. Open the dashboard at `http://localhost:3000`.
2. Inspect the **Navbar** and **Worker Topology** panel.
3. Verify that `worker-1` and `worker-2` are shown with status `ONLINE`, along with their available memory (e.g. 4096 MB) and initial layer assignments.

### Stage 2: Dynamic Layer Partitioning
1. Click the **"Rebalance Partitions"** button in the Worker Topology panel.
2. The UI triggers `POST /api/models/load` to partition the 12 transformer layers of `distilgpt2`:
   - `worker-1`: Layers 0-5
   - `worker-2`: Layers 6-11
3. Confirm that each node reflects its updated partition range.

### Stage 3: Real-Time Streaming Inference
1. In the **Inference Console**, enter a prompt:
   ```text
   Distributed memory grids partition large language models across consumer hardware to
   ```
2. Select parameters: `Max Tokens: 25`, `Temperature: 0.7`, `Priority: High`.
3. Click **"Generate"**.
4. Observe the tokens streaming into the console via WebSocket in real time.
5. Review the post-generation telemetry bar:
   - **TTFT (Time to First Token)**: ~30-50 ms
   - **Throughput**: ~35-50 tokens/sec
   - **Total Latency**: ~0.5-0.7 s

### Stage 4: Job Cancellation Mid-Stream
1. Enter a long-generation prompt with `Max Tokens: 100`.
2. Click **"Generate"**.
3. While tokens are streaming, click the red **"Stop"** button.
4. Verify that:
   - Generation terminates immediately.
   - The WebSocket receives a `cancelled` event.
   - The job is recorded in the **Recent Request History** table with status `cancelled`.
   - The KV-cache resources for this job are released.

### Stage 5: Observability & Prometheus Exporter
1. Check the **Telemetry Bar** on the dashboard for real-time memory and hit rate.
2. Open `http://localhost:8000/api/metrics/prometheus` in a separate tab or run:
   ```bash
   curl http://localhost:8000/api/metrics/prometheus
   ```
3. Verify standard Prometheus exposition metrics:
   - `dmg_active_workers`
   - `dmg_queue_size`
   - `dmg_processed_requests_total`
   - `dmg_avg_ttft_seconds`
   - `dmg_avg_tokens_per_second`
   - `dmg_kv_cache_memory_bytes`

### Stage 6: Fault-Tolerance & Automated Failover
1. Terminate Worker 2 (press `Ctrl+C` in Terminal 3, or run `docker stop dmg-worker-2`).
2. Within the heartbeat timeout period, observe the Master Controller logs:
   ```text
   WARNING:DMG-Master: FAULT DETECTED: Worker 'worker-2' timed out.
   INFO:DMG-Master: Triggering automated fault-tolerance repartitioning.
   ```
3. Refresh the dashboard:
   - `worker-2` is marked offline.
   - `worker-1` automatically inherits the remaining layers (Layers 0-11) to preserve full model availability without dropping incoming requests.

---

## 5. Automated Demonstration Script

You can also run the complete automated test script that programmatically executes all 6 stages and verifies output:

```bash
python scripts/demo_cluster.py
```
Expected output:
```text
==============================================================================
   DISTRIBUTED MEMORY GRID (DMG) - END-TO-END DEMONSTRATION SCRIPT
==============================================================================
[Stage 1/6] Initializing Master Controller and Worker Cluster...
-> Successfully registered 3 worker nodes in cluster.

[Stage 2/6] Planning & Assigning Model Partition Layers...
   * Node [worker-1] @ 127.0.0.1:50052 | Layers: (0, 2) | Available RAM: 4096 MB
   * Node [worker-2] @ 127.0.0.1:50053 | Layers: (3, 5) | Available RAM: 4096 MB
   * Node [worker-3] @ 127.0.0.1:50054 | Layers: (6, 11) | Available RAM: 6144 MB
...
==============================================================================
   DEMONSTRATION COMPLETE: ALL 6 PHASES PASSED WITH ZERO DATA LOSS
==============================================================================
```
