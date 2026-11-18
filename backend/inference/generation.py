"""
Autoregressive token generation, greedy/temperature sampling, and streaming.
Maintained by Member 2 (AI / Model Engineer) & Member 3.
"""
from typing import AsyncGenerator
import torch


class TokenSampler:
    @staticmethod
    def sample(logits: torch.Tensor, temperature: float = 0.7) -> int:
        if temperature <= 0.0:
            return int(torch.argmax(logits[-1, :]).item())
        scaled = logits[-1, :] / temperature
        probs = torch.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())


class GenerationEngine:
    def __init__(self, tokenizer, pipeline_forward_fn):
        self.tokenizer = tokenizer
        self.pipeline_forward_fn = pipeline_forward_fn

    async def stream_generate(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        curr_ids = input_ids

        for _ in range(max_new_tokens):
            logits = await self.pipeline_forward_fn(curr_ids)
            next_token = TokenSampler.sample(logits, temperature=temperature)
            if next_token == self.tokenizer.eos_token_id:
                break
            token_str = self.tokenizer.decode([next_token])
            yield token_str
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]])], dim=1)

