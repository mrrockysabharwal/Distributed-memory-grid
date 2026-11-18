"""
Latency, throughput, TTFT, and tokens/sec measurement collectors.
Maintained by Member 3 & Member 4.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RequestBenchmark:
    request_id: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    first_token_time: float = 0.0
    end_time: float = 0.0

    def mark_first_token(self):
        if self.first_token_time == 0.0:
            self.first_token_time = time.time()

    def mark_end(self, total_tokens: int):
        self.end_time = time.time()
        self.generated_tokens = total_tokens

    @property
    def ttft_seconds(self) -> float:
        if self.first_token_time == 0.0:
            return 0.0
        return self.first_token_time - self.start_time

    @property
    def total_latency_seconds(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        duration = self.total_latency_seconds
        if duration <= 0:
            return 0.0
        return self.generated_tokens / duration


class MetricsCollector:
    def __init__(self):
        self.history: List[RequestBenchmark] = []

    def record(self, bench: RequestBenchmark):
        self.history.append(bench)

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"total_requests": 0, "avg_ttft": 0.0, "avg_tokens_per_sec": 0.0}
        avg_ttft = sum(b.ttft_seconds for b in self.history) / len(self.history)
        avg_tps = sum(b.tokens_per_second for b in self.history) / len(self.history)
        return {
            "total_requests": len(self.history),
            "avg_ttft_seconds": round(avg_ttft, 4),
            "avg_tokens_per_second": round(avg_tps, 2),
        }

