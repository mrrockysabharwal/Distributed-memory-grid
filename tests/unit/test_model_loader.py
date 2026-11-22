"""
Unit tests for model partition loader and layer extraction.
"""
from backend.inference.model_loader import ModelPartitionLoader, SyntheticBlock


def test_synthetic_block_execution():
    block = SyntheticBlock(layer_idx=0)
    x = [1.0, 2.0]
    out, presents = block(x, use_cache=True)
    assert out == x
    assert presents == (x, x)


def test_model_partition_loader_extraction():
    loader = ModelPartitionLoader(model_id="mock-gpt")
    tokenizer = loader.load_tokenizer()
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")

    embeddings, blocks, final_head = loader.extract_layers(total_fallback_layers=12)
    assert len(blocks) == 12
    assert final_head is not None


def test_get_worker_partition_blocks():
    loader = ModelPartitionLoader(model_id="mock-gpt")
    # Slice layers 0-3 for worker-1
    worker1_blocks = loader.get_worker_partition_blocks(0, 3)
    assert len(worker1_blocks) == 4

    # Slice layers 4-7 for worker-2
    worker2_blocks = loader.get_worker_partition_blocks(4, 7)
    assert len(worker2_blocks) == 4

