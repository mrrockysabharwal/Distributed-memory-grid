"""
Key-Value (KV) cache manager for attention states per request.
Includes contiguous tensor caching, incremental decode appends, LRU eviction,
memory budget tracking, and Paged KV Cache block allocation.

Maintained by Member 3 (Inference Optimization Engineer).
"""
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger("DMG-KVCache")

# Graceful PyTorch / NumPy handling
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


def _estimate_tensor_bytes(obj: Any) -> int:
    """Helper to calculate byte size of tensors, arrays, or nested tuples/lists."""
    if obj is None:
        return 0
    if HAS_TORCH and isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (tuple, list)):
        return sum(_estimate_tensor_bytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_estimate_tensor_bytes(v) for v in obj.values())
    return 0


class KVCacheManager:
    """
    High-performance, thread-safe KV Cache Manager for contiguous layer states.
    Supports:
    - Incremental key-value appends during autoregressive decoding.
    - LRU eviction when request capacity or memory budget is exceeded.
    - Real-time memory consumption telemetry and cache hit/miss statistics.
    """
    def __init__(
        self,
        max_cached_requests: int = 50,
        max_memory_bytes: Optional[int] = None,
        seq_dim: int = -2,
    ):
        """
        Args:
            max_cached_requests: Maximum number of concurrent request sequences.
            max_memory_bytes: Optional memory cap in bytes (e.g. 2 * 1024**3 for 2GB).
            seq_dim: Dimension index representing the sequence length (default -2 for [batch, heads, seq, dim]).
        """
        self.max_cached_requests = max_cached_requests
        self.max_memory_bytes = max_memory_bytes
        self.seq_dim = seq_dim

        # Structure: {request_id: {layer_idx: (key_tensor, value_tensor)}}
        self._cache_store: Dict[str, Dict[int, Any]] = {}
        # Metadata tracking: {request_id: {"last_accessed": float, "memory_bytes": int}}
        self._request_meta: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get_cache(self, request_id: str, layer_idx: int) -> Optional[Any]:
        """Retrieve cached KV states for a given request and layer."""
        with self._lock:
            if request_id in self._cache_store:
                layer_cache = self._cache_store[request_id].get(layer_idx)
                if layer_cache is not None:
                    self._hits += 1
                    self._request_meta[request_id]["last_accessed"] = time.time()
                    return layer_cache
            self._misses += 1
            return None

    def update_cache(self, request_id: str, layer_idx: int, kv_tensors: Any):
        """
        Insert or overwrite KV states for a given request and layer.
        Triggers LRU eviction if limits are exceeded.
        """
        with self._lock:
            # Check if this is a new request and we are at capacity
            if request_id not in self._cache_store:
                if len(self._cache_store) >= self.max_cached_requests:
                    self._evict_lru(exclude_id=request_id)

            if request_id not in self._cache_store:
                self._cache_store[request_id] = {}
                self._request_meta[request_id] = {
                    "last_accessed": time.time(),
                    "memory_bytes": 0,
                    "created_at": time.time(),
                }

            self._cache_store[request_id][layer_idx] = kv_tensors
            self._request_meta[request_id]["last_accessed"] = time.time()
            self._recalculate_request_memory(request_id)

            # Check memory budget
            if self.max_memory_bytes and self.total_memory_bytes > self.max_memory_bytes:
                self._evict_to_budget(exclude_id=request_id)

    def append_cache(
        self,
        request_id: str,
        layer_idx: int,
        new_k: Any,
        new_v: Any,
        seq_dim: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """
        Incrementally append new single-token key and value states to existing cache.
        If no cache exists, initializes it with (new_k, new_v).
        """
        dim = seq_dim if seq_dim is not None else self.seq_dim
        with self._lock:
            existing = self.get_cache(request_id, layer_idx)
            if existing is None:
                updated = (new_k, new_v)
            else:
                old_k, old_v = existing
                if HAS_TORCH and isinstance(old_k, torch.Tensor) and isinstance(new_k, torch.Tensor):
                    updated_k = torch.cat([old_k, new_k], dim=dim)
                    updated_v = torch.cat([old_v, new_v], dim=dim)
                elif HAS_NUMPY and isinstance(old_k, np.ndarray) and isinstance(new_k, np.ndarray):
                    updated_k = np.concatenate([old_k, new_k], axis=dim)
                    updated_v = np.concatenate([old_v, new_v], axis=dim)
                else:
                    # Fallback for generic lists
                    if isinstance(old_k, list):
                        updated_k = old_k + (new_k if isinstance(new_k, list) else [new_k])
                        updated_v = old_v + (new_v if isinstance(new_v, list) else [new_v])
                    else:
                        updated_k = (old_k, new_k)
                        updated_v = (old_v, new_v)
                updated = (updated_k, updated_v)

            self.update_cache(request_id, layer_idx, updated)
            return updated

    def get_seq_len(self, request_id: str, layer_idx: int) -> int:
        """Return the current sequence length cached for a given layer."""
        with self._lock:
            cache = self.get_cache(request_id, layer_idx)
            if cache is None:
                return 0
            k = cache[0] if isinstance(cache, (tuple, list)) else cache
            if HAS_TORCH and isinstance(k, torch.Tensor):
                return k.shape[self.seq_dim]
            if HAS_NUMPY and isinstance(k, np.ndarray):
                return k.shape[self.seq_dim]
            if isinstance(k, list):
                return len(k)
            return 1

    def clear_request(self, request_id: str):
        """Evict all cached layers for a finished or aborted request."""
        with self._lock:
            if request_id in self._cache_store:
                del self._cache_store[request_id]
            if request_id in self._request_meta:
                del self._request_meta[request_id]

    def clear_all(self):
        """Clear the entire cache across all requests."""
        with self._lock:
            self._cache_store.clear()
            self._request_meta.clear()
            self._hits = 0
            self._misses = 0

    def _recalculate_request_memory(self, request_id: str):
        """Calculate total bytes stored in tensors for a single request."""
        if request_id not in self._cache_store:
            return
        total = 0
        for layer_tensors in self._cache_store[request_id].values():
            total += _estimate_tensor_bytes(layer_tensors)
        self._request_meta[request_id]["memory_bytes"] = total

    @property
    def total_memory_bytes(self) -> int:
        """Sum of memory in bytes occupied by all active cached requests."""
        with self._lock:
            return sum(meta.get("memory_bytes", 0) for meta in self._request_meta.values())

    def _evict_lru(self, exclude_id: Optional[str] = None):
        """Evict the least recently accessed request."""
        candidates = [
            (rid, meta["last_accessed"])
            for rid, meta in self._request_meta.items()
            if rid != exclude_id
        ]
        if not candidates:
            return
        # Find oldest accessed request
        oldest_id = min(candidates, key=lambda x: x[1])[0]
        logger.info(f"KVCache: Capacity reached. Evicting LRU request: {oldest_id}")
        self.clear_request(oldest_id)

    def _evict_to_budget(self, exclude_id: Optional[str] = None):
        """Evict LRU requests until memory usage drops below max_memory_bytes."""
        if self.max_memory_bytes is None:
            return
        while self.total_memory_bytes > self.max_memory_bytes and len(self._cache_store) > 1:
            candidates = [
                (rid, meta["last_accessed"])
                for rid, meta in self._request_meta.items()
                if rid != exclude_id
            ]
            if not candidates:
                break
            oldest_id = min(candidates, key=lambda x: x[1])[0]
            logger.info(f"KVCache: Memory budget exceeded. Evicting request: {oldest_id}")
            self.clear_request(oldest_id)

    def current_usage_stats(self) -> Dict[str, Any]:
        """Return real-time usage metrics and telemetry."""
        with self._lock:
            total_bytes = self.total_memory_bytes
            total_requests = len(self._cache_store)
            total_accesses = self._hits + self._misses
            hit_rate = round(self._hits / total_accesses, 4) if total_accesses > 0 else 0.0

            return {
                "active_requests_in_cache": total_requests,
                "max_capacity": self.max_cached_requests,
                "total_memory_bytes": total_bytes,
                "total_memory_mb": round(total_bytes / (1024 * 1024), 2),
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "hit_rate": hit_rate,
            }


# ---------------------------------------------------------------------------
# Paged KV Cache Architecture (Block Allocation & Virtual-to-Physical Paging)
# ---------------------------------------------------------------------------

class PhysicalBlock:
    """Represents a fixed-capacity physical memory chunk holding key and value states."""
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size  # Number of tokens per block
        self.num_tokens = 0
        self.keys: Any = None
        self.values: Any = None

    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    def free(self):
        self.num_tokens = 0
        self.keys = None
        self.values = None


class BlockAllocator:
    """Manages pool of physical blocks and handles allocation/deallocation."""
    def __init__(self, total_blocks: int = 128, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._pool: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size) for i in range(total_blocks)
        ]
        self._free_indices: List[int] = list(range(total_blocks))
        self._lock = threading.Lock()

    def allocate(self) -> Optional[PhysicalBlock]:
        """Allocate a free block from the pool."""
        with self._lock:
            if not self._free_indices:
                return None
            idx = self._free_indices.pop(0)
            return self._pool[idx]

    def free(self, block_id: int):
        """Return a block to the free pool."""
        with self._lock:
            if 0 <= block_id < self.total_blocks and block_id not in self._free_indices:
                self._pool[block_id].free()
                self._free_indices.append(block_id)

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free_indices)

    @property
    def allocated_blocks(self) -> int:
        with self._lock:
            return self.total_blocks - len(self._free_indices)


class PagedKVCache:
    """
    Paged KV Cache manager inspired by vLLM.
    Maps logical sequence tokens into non-contiguous physical blocks via a block table,
    eliminating external memory fragmentation and allowing dynamic token growth.
    """
    def __init__(self, total_blocks: int = 256, block_size: int = 16):
        self.block_size = block_size
        self.allocator = BlockAllocator(total_blocks=total_blocks, block_size=block_size)
        # Mapping: {request_id: {layer_idx: [PhysicalBlock, ...]}}
        self._block_tables: Dict[str, Dict[int, List[PhysicalBlock]]] = {}
        self._lock = threading.RLock()

    def allocate_for_request(self, request_id: str, layer_idx: int, num_tokens: int) -> bool:
        """Pre-allocate sufficient physical blocks for an incoming sequence."""
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        with self._lock:
            if self.allocator.free_blocks < needed_blocks:
                return False

            if request_id not in self._block_tables:
                self._block_tables[request_id] = {}
            if layer_idx not in self._block_tables[request_id]:
                self._block_tables[request_id][layer_idx] = []

            for _ in range(needed_blocks):
                block = self.allocator.allocate()
                if block is None:
                    return False
                self._block_tables[request_id][layer_idx].append(block)
            return True

    def get_blocks(self, request_id: str, layer_idx: int) -> List[PhysicalBlock]:
        """Return the physical blocks assigned to a request layer."""
        with self._lock:
            if request_id in self._block_tables:
                return self._block_tables[request_id].get(layer_idx, [])
            return []

    def clear_request(self, request_id: str):
        """Free all physical blocks allocated to the request."""
        with self._lock:
            if request_id in self._block_tables:
                for layer_blocks in self._block_tables[request_id].values():
                    for block in layer_blocks:
                        self.allocator.free(block.block_id)
                del self._block_tables[request_id]

    def usage_stats(self) -> Dict[str, Any]:
        """Report paged cache memory stats."""
        with self._lock:
            total = self.allocator.total_blocks
            allocated = self.allocator.allocated_blocks
            free = self.allocator.free_blocks
            return {
                "total_blocks": total,
                "allocated_blocks": allocated,
                "free_blocks": free,
                "block_size_tokens": self.block_size,
                "utilization_pct": round((allocated / total) * 100, 2) if total > 0 else 0.0,
                "active_requests": len(self._block_tables),
            }
