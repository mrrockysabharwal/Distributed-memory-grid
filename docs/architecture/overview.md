# DMG Architecture Overview

## Motivation & Problem Statement
Running modern LLMs requires significant memory (VRAM/RAM) to hold model parameters, KV cache, and runtime activation tensors. Ordinary consumer machines or edge devices often fail to fit entire models. 

**Distributed Memory Grid (DMG)** provides a pipeline/model-layer parallel execution framework where:
1. Layers are sliced and distributed across multiple worker nodes.
2. Intermediate activations flow sequentially across nodes.
3. A centralized Master Controller coordinates cluster health, partition placement, and request queuing.
4. An API Gateway connects clients with streaming token responses.

## Key Design Principles
1. **Pipeline Parallelism First**: Model-layer partitioning (e.g., Worker 1 executes layers 0-3, Worker 2 executes 4-7) is prioritized over intra-layer tensor parallelism, which requires low-latency inter-GPU networks (NVLink/NCCL) and performs poorly on commodity networks.
2. **Explicit Separation of Concerns**:
   - **Control Plane**: gRPC for fast, typed RPCs (heartbeats, registration, lifecycle).
   - **Data Plane**: Direct tensor activation transfers between adjacent stages.
3. **Observability by Design**: Every worker reports memory utilization, CPU/GPU load, and communication latency to inform scheduling decisions.

