"""
Integration test for cross-partition pipeline parallel forward execution and KV-caching.
"""
import pytest
import numpy as np
from backend.inference.kv_cache import KVCacheManager
from backend.worker.executor import PartitionExecutor
from backend.distributed.pipeline_parallel import PipelineParallelEngine, PipelineStage
from backend.inference.generation import GenerationEngine, TokenSampler


def test_pipeline_parallel_multi_stage_execution():
    kv_cache = KVCacheManager()

    # Create mock blocks for Stage 1 (Layers 0-1) and Stage 2 (Layers 2-3)
    def mock_block_stage1(x, **kwargs):
        # Adds 1.0 to activation
        return x + 1.0

    def mock_block_stage2(x, **kwargs):
        # Multiplies activation by 2.0
        return x * 2.0

    executor_1 = PartitionExecutor(start_layer=0, end_layer=1, kv_cache_mgr=kv_cache)
    executor_1.load_layers([mock_block_stage1, mock_block_stage1])

    executor_2 = PartitionExecutor(start_layer=2, end_layer=3, kv_cache_mgr=kv_cache)
    executor_2.load_layers([mock_block_stage2, mock_block_stage2])

    stages = [
        PipelineStage(stage_id=0, worker_id="worker-1", start_layer=0, end_layer=1, executor=executor_1),
        PipelineStage(stage_id=1, worker_id="worker-2", start_layer=2, end_layer=3, executor=executor_2),
    ]

    pipeline = PipelineParallelEngine(stages=stages)

    # Input: batch=1, seq=3
    input_ids = np.array([[10, 20, 30]])

    # Embedding produces zeros (1, 3, 64)
    # Stage 1 adds 1.0 twice -> 2.0
    # Stage 2 multiplies by 2.0 twice -> 2.0 * 2.0 * 2.0 = 8.0
    # Head produces logits
    logits = pipeline.forward_sync(input_ids, request_id="req-integ-1", is_prefill=True)

    assert logits.shape[0] == 1  # batch
    assert logits.shape[1] == 3  # seq_len
    assert logits.shape[2] == 50257  # vocab


def test_pipeline_streaming_generation():
    class DummyTokenizer:
        eos_token_id = 9999
        def encode(self, text, **kwargs):
            return np.array([[1, 2]])
        def decode(self, token_ids):
            return f"w{token_ids[0]} "

    def mock_forward(x, **kwargs):
        # Return logits where token 7 is highest
        logits = np.zeros((1, 1, 100))
        logits[0, 0, 7] = 10.0
        return logits

    engine = GenerationEngine(tokenizer=DummyTokenizer(), pipeline_forward_fn=mock_forward)
    tokens = engine.generate("test prompt", max_new_tokens=4)

    assert len(tokens) == 4
    assert tokens[0] == "w7 "
