"""
Autoregressive token generation, KV-cache-aware decoding, advanced sampling, and streaming.
Maintained by Member 2 (AI / Model Engineer) & Member 3 (Inference Optimization Engineer).
"""
import time
import math
import uuid
import asyncio
from typing import AsyncGenerator, Optional, List, Tuple, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from backend.inference.kv_cache import KVCacheManager
from backend.monitoring.metrics import RequestBenchmark


class TokenSampler:
    """
    Advanced token sampling engine supporting Greedy, Temperature, Top-K,
    Top-P (Nucleus), and Repetition Penalty. Works with both PyTorch tensors and NumPy arrays.
    """
    @staticmethod
    def apply_repetition_penalty(
        logits: Any,
        generated_token_ids: List[int],
        penalty: float = 1.1,
    ) -> Any:
        """Penalize logits of already generated tokens to prevent repetition loops."""
        if penalty == 1.0 or not generated_token_ids:
            return logits

        if HAS_TORCH and isinstance(logits, torch.Tensor):
            last_logits = logits.view(-1, logits.size(-1))[-1, :]
            for token_id in set(generated_token_ids):
                if token_id < last_logits.shape[-1]:
                    if last_logits[token_id] < 0:
                        last_logits[token_id] = last_logits[token_id] * penalty
                    else:
                        last_logits[token_id] = last_logits[token_id] / penalty
            return logits
        elif HAS_NUMPY and isinstance(logits, np.ndarray):
            reshaped = logits.reshape(-1, logits.shape[-1])
            last_logits = reshaped[-1, :]
            for token_id in set(generated_token_ids):
                if token_id < last_logits.shape[-1]:
                    if last_logits[token_id] < 0:
                        last_logits[token_id] = last_logits[token_id] * penalty
                    else:
                        last_logits[token_id] = last_logits[token_id] / penalty
            return logits
        return logits

    @staticmethod
    def top_k_top_p_filtering(
        logits: Any,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
    ) -> Any:
        """Filter distribution using top-k and/or top-p (nucleus) thresholding."""
        if HAS_TORCH and isinstance(logits, torch.Tensor):
            last_logits = logits.view(-1, logits.size(-1))[-1, :].clone()
            if top_k > 0:
                indices_to_remove = last_logits < torch.topk(last_logits, min(top_k, last_logits.size(-1)))[0][..., -1, None]
                last_logits[indices_to_remove] = filter_value

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Shift indices right to keep first token above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_logits[indices_to_remove] = filter_value
            return last_logits
        elif HAS_NUMPY and isinstance(logits, np.ndarray):
            last_logits = np.copy(logits.reshape(-1, logits.shape[-1])[-1, :])
            vocab_size = last_logits.shape[-1]
            if top_k > 0 and top_k < vocab_size:
                top_k_idx = np.argpartition(last_logits, -top_k)[-top_k:]
                min_top_k = np.min(last_logits[top_k_idx])
                last_logits[last_logits < min_top_k] = filter_value

            if 0.0 < top_p < 1.0:
                sorted_idx = np.argsort(last_logits)[::-1]
                sorted_logits = last_logits[sorted_idx]
                exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
                probs = exp_logits / np.sum(exp_logits)
                cum_probs = np.cumsum(probs)
                remove_idx = cum_probs > top_p
                remove_idx[1:] = remove_idx[:-1].copy()
                remove_idx[0] = False
                last_logits[sorted_idx[remove_idx]] = filter_value
            return last_logits
        return logits

    @classmethod
    def sample(
        cls,
        logits: Any,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        generated_token_ids: Optional[List[int]] = None,
    ) -> int:
        """Sample next token index from model output logits."""
        # Repetition penalty
        if repetition_penalty != 1.0 and generated_token_ids:
            logits = cls.apply_repetition_penalty(logits, generated_token_ids, repetition_penalty)

        # Greedy decoding
        if temperature <= 0.0:
            if HAS_TORCH and isinstance(logits, torch.Tensor):
                return int(torch.argmax(logits[-1, :]).item())
            elif HAS_NUMPY and isinstance(logits, np.ndarray):
                return int(np.argmax(logits[-1, :]))
            elif isinstance(logits, (list, tuple)):
                flat = logits[-1] if isinstance(logits[-1], (list, tuple)) else logits
                return int(flat.index(max(flat)))
            return 0

        # Filter top-k and top-p
        filtered_logits = cls.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # Scale by temperature and sample
        if HAS_TORCH and isinstance(filtered_logits, torch.Tensor):
            scaled = filtered_logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            # Guard against NaN/zero sum
            if torch.isnan(probs).any() or probs.sum() == 0:
                return int(torch.argmax(filtered_logits).item())
            next_token = torch.multinomial(probs, num_samples=1)
            return int(next_token.item())
        elif HAS_NUMPY and isinstance(filtered_logits, np.ndarray):
            scaled = filtered_logits / temperature
            shifted = scaled - np.max(scaled)
            exp_probs = np.exp(shifted)
            prob_sum = np.sum(exp_probs)
            if np.isnan(prob_sum) or prob_sum == 0:
                return int(np.argmax(filtered_logits))
            probs = exp_probs / prob_sum
            return int(np.random.choice(len(probs), p=probs))

        return 0


class GenerationEngine:
    """
    Autoregressive generation loop with KV-cache integration, TTFT tracking,
    and streaming token yields.
    """
    def __init__(
        self,
        tokenizer: Any,
        pipeline_forward_fn: Any,
        kv_cache_mgr: Optional[KVCacheManager] = None,
    ):
        self.tokenizer = tokenizer
        self.pipeline_forward_fn = pipeline_forward_fn
        self.kv_cache_mgr = kv_cache_mgr or KVCacheManager()

    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated tokens with KV-cache optimization and TTFT measurement.
        """
        req_id = request_id or str(uuid.uuid4())
        benchmark = RequestBenchmark(request_id=req_id)

        # Encode input prompt
        if hasattr(self.tokenizer, "encode"):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt" if HAS_TORCH else None)
        else:
            # Fallback mock encoding
            input_ids = [1, 2, 3]

        prompt_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids)
        benchmark.prompt_tokens = prompt_len

        generated_ids: List[int] = []
        curr_input = input_ids

        try:
            for step in range(max_new_tokens):
                is_prefill = (step == 0)

                # Forward pass: pass is_prefill flag and request_id so workers can retrieve/update KV cache
                if hasattr(self.pipeline_forward_fn, "__code__") and "is_prefill" in self.pipeline_forward_fn.__code__.co_varnames:
                    logits = await self.pipeline_forward_fn(curr_input, request_id=req_id, is_prefill=is_prefill)
                else:
                    logits = await self.pipeline_forward_fn(curr_input)

                # Sample next token
                next_token = TokenSampler.sample(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_token_ids=generated_ids,
                )

                # Check EOS
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is not None and next_token == eos_id:
                    break

                generated_ids.append(next_token)

                # Mark TTFT on first generated token
                if step == 0:
                    benchmark.mark_first_token()

                # Decode token to string and stream
                if hasattr(self.tokenizer, "decode"):
                    token_str = self.tokenizer.decode([next_token])
                else:
                    token_str = f"tok_{next_token} "

                yield token_str

                # For subsequent decode steps:
                # With KV cache, only the single newly generated token is passed into next forward pass!
                if HAS_TORCH and isinstance(input_ids, torch.Tensor):
                    curr_input = torch.tensor([[next_token]], device=input_ids.device)
                elif HAS_NUMPY and isinstance(input_ids, np.ndarray):
                    curr_input = np.array([[next_token]])
                else:
                    curr_input = [next_token]

        finally:
            benchmark.mark_end(len(generated_ids))
            # Clean up KV cache for this completed request
            self.kv_cache_mgr.clear_request(req_id)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        request_id: Optional[str] = None,
    ) -> List[str]:
        """
        Synchronous token generation using pipeline forward function.
        """
        import inspect
        req_id = request_id or str(uuid.uuid4())
        benchmark = RequestBenchmark(request_id=req_id)

        if hasattr(self.tokenizer, "encode"):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt" if HAS_TORCH else None)
        else:
            input_ids = [1, 2, 3]

        prompt_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else len(input_ids)
        benchmark.prompt_tokens = prompt_len

        generated_ids: List[int] = []
        tokens_out: List[str] = []
        curr_input = input_ids

        try:
            for step in range(max_new_tokens):
                is_prefill = (step == 0)

                # Forward pass
                if inspect.iscoroutinefunction(self.pipeline_forward_fn):
                    # For coroutine functions in sync mode
                    loop = asyncio.new_event_loop()
                    try:
                        logits = loop.run_until_complete(self.pipeline_forward_fn(curr_input))
                    finally:
                        loop.close()
                else:
                    logits = self.pipeline_forward_fn(curr_input)

                next_token = TokenSampler.sample(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_token_ids=generated_ids,
                )

                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is not None and next_token == eos_id:
                    break

                generated_ids.append(next_token)
                if step == 0:
                    benchmark.mark_first_token()

                if hasattr(self.tokenizer, "decode"):
                    token_str = self.tokenizer.decode([next_token])
                else:
                    token_str = f"tok_{next_token} "

                tokens_out.append(token_str)

                if HAS_TORCH and isinstance(input_ids, torch.Tensor):
                    curr_input = torch.tensor([[next_token]], device=input_ids.device)
                elif HAS_NUMPY and isinstance(input_ids, np.ndarray):
                    curr_input = np.array([[next_token]])
                else:
                    curr_input = [next_token]

            return tokens_out
        finally:
            benchmark.mark_end(len(generated_ids))
            self.kv_cache_mgr.clear_request(req_id)
