"""
Unit tests for TokenSampler and GenerationEngine.
Maintained by Member 3 (Inference Optimization Engineer).
"""
import pytest
import numpy as np
from backend.inference.generation import TokenSampler, GenerationEngine
from backend.inference.kv_cache import KVCacheManager


def test_sampler_greedy():
    # Logits: token 3 has the highest score (10.0)
    logits = np.array([[1.0, 2.0, 3.0, 10.0, 0.5]])
    sampled = TokenSampler.sample(logits, temperature=0.0)
    assert sampled == 3


def test_sampler_repetition_penalty():
    # Token 2 has the highest score, but is in generated_token_ids
    logits = np.array([[1.0, 2.0, 10.0, 9.5]])
    penalized = TokenSampler.apply_repetition_penalty(
        np.copy(logits), generated_token_ids=[2], penalty=2.0
    )
    # Logit for token 2 should be divided by 2 (10 / 2 = 5.0), making token 3 (9.5) the winner
    sampled = TokenSampler.sample(penalized, temperature=0.0)
    assert sampled == 3


def test_sampler_top_k_top_p():
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered = TokenSampler.top_k_top_p_filtering(logits, top_k=2)
    # Only top 2 tokens (indices 3 and 4) should remain valid
    assert filtered[3] == 4.0
    assert filtered[4] == 5.0
    assert filtered[0] == -float("inf")


def test_generation_engine_generation():
    class MockTokenizer:
        eos_token_id = 999

        def encode(self, text, **kwargs):
            return np.array([[1, 2]])

        def decode(self, token_ids):
            return f"word_{token_ids[0]} "

    def mock_forward(curr_input, **kwargs):
        return np.array([[0.1, 0.2, 0.3, 0.4, 5.0]])

    engine = GenerationEngine(tokenizer=MockTokenizer(), pipeline_forward_fn=mock_forward)
    tokens = engine.generate("Hello", max_new_tokens=3)

    assert len(tokens) == 3
    assert tokens[0] == "word_4 "
