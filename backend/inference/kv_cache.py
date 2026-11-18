"""
Key-Value (KV) cache manager for attention states per request.
Maintained by Member 3 (Inference Optimization Engineer).
"""
from typing import Dict, Any, Optional
import torch


class KVCacheManager:
    def __init__(self, max_cached_requests: int = 50):
        self.max_cached_requests = max_cached_requests
        self._cache_store: Dict[str, Dict[int, Any]] = {}

    def get_cache(self, request_id: str, layer_idx: int) -> Optional[Any]:
        if request_id in self._cache_store:
            return self._cache_store[request_id].get(layer_idx)
        return None

    def update_cache(self, request_id: str, layer_idx: int, kv_tensors: Any):
        if request_id not in self._cache_store:
            self._cache_store[request_id] = {}
        self._cache_store[request_id][layer_idx] = kv_tensors

    def clear_request(self, request_id: str):
        if request_id in self._cache_store:
            del self._cache_store[request_id]

    def current_usage_stats(self) -> Dict[str, int]:
        return {
            "active_requests_in_cache": len(self._cache_store),
            "max_capacity": self.max_cached_requests,
        }

