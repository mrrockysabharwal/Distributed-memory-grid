"""
Full end-to-end integration test validating the entire distributed inference loop:
Master Controller -> Worker Nodes -> Scheduler -> Continuous Batcher -> Pipeline Parallel -> Metrics.
"""
from backend.master.controller import MasterController
from backend.worker.worker_node import WorkerNode
from backend.master.scheduler import GenerationRequest, RequestPriority
from backend.inference.batching import ContinuousBatcher
from backend.distributed.pipeline_parallel import PipelineParallelEngine, PipelineStage


def test_full_cluster_end_to_end_pipeline():
    # 1. Initialize Master
    master = MasterController(total_model_layers=6)

    # 2. Spawn and register 2 worker nodes
    worker1 = WorkerNode(worker_id="w1", port=50052)
    worker1.connect_to_master(master)
    assert worker1.register() is True

    worker2 = WorkerNode(worker_id="w2", port=50053)
    worker2.connect_to_master(master)
    assert worker2.register() is True

    assert len(master.registry.get_active_workers()) == 2
    assert master._partition_map["w1"] == (0, 2)  # layers 0-2
    assert master._partition_map["w2"] == (3, 5)  # layers 3-5

    # 3. Assemble PipelineParallelEngine linking worker executors
    stages = [
        PipelineStage(stage_id=0, worker_id="w1", start_layer=0, end_layer=2, executor=worker1.executor),
        PipelineStage(stage_id=1, worker_id="w2", start_layer=3, end_layer=5, executor=worker2.executor),
    ]
    pipeline = PipelineParallelEngine(stages=stages)

    # 4. Continuous Batcher with Master Scheduler
    batcher = ContinuousBatcher(scheduler=master.scheduler, kv_cache_mgr=master.kv_cache_mgr, max_batch_size=4)

    # 5. Submit 2 requests with different priorities
    req_high = GenerationRequest(
        request_id="req-high",
        prompt="Explain distributed pipelines",
        max_new_tokens=3,
        priority=RequestPriority.HIGH,
    )
    req_norm = GenerationRequest(
        request_id="req-norm",
        prompt="Hello world",
        max_new_tokens=4,
        priority=RequestPriority.NORMAL,
    )
    master.scheduler.submit_request_sync(req_high)
    master.scheduler.submit_request_sync(req_norm)

    # 6. Execute continuous batching steps
    def forward_fn(batch):
        # Passes activations through multi-stage pipeline
        outputs = []
        for r in batch:
            logits = pipeline.forward_sync(
                input_ids=[[1, 2]],
                request_id=r.request_id,
                is_prefill=r.is_prefill,
            )
            outputs.append(logits)
        return outputs

    # Step 1: Both requests admitted
    step1 = batcher.step_sync(forward_fn)
    assert step1["active_count"] == 2
    assert req_high.generated_tokens == 1
    assert req_norm.generated_tokens == 1

    # Step 2: Decode step
    step2 = batcher.step_sync(forward_fn)
    assert req_high.generated_tokens == 2
    assert req_norm.generated_tokens == 2

    # Step 3: req_high reaches max_new_tokens (3)
    step3 = batcher.step_sync(forward_fn)
    assert req_high.generated_tokens == 3

    # Step 4: req_high is pruned, freeing slot, and marked complete
    step4 = batcher.step_sync(forward_fn)
    assert step4["finished_count"] >= 1
    assert "req-high" in master.scheduler.completed_requests
    assert master.scheduler.completed_requests["req-high"].finished is True
