"""
Partition layer forward pass execution engine with KV-cache integration.
Maintained by Member 2 (AI / Model Engineer) & Member 3 (Inference Optimization Engineer).
"""
import logging
from typing import List, Optional, Any, Tuple
from backend.inference.kv_cache import KVCacheManager

logger = logging.getLogger("DMG-PartitionExecutor")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


class PartitionExecutor:
    """
    Executes forward passes for an assigned contiguous range of Transformer layers [start_layer, end_layer].
    Integrates directly with KVCacheManager to support incremental key-value caching during decoding.
    """
    def __init__(
        self,
        start_layer: int,
        end_layer: int,
        device: str = "cpu",
        kv_cache_mgr: Optional[KVCacheManager] = None,
        precision: str = "fp32",
    ):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device_str = device
        self.precision = precision.lower()
        self.kv_cache_mgr = kv_cache_mgr or KVCacheManager()

        if HAS_TORCH:
            self.device = torch.device(device)
            self.layers = nn.ModuleList()
        else:
            self.device = device
            self.layers = []

    def load_layers(self, blocks: List[Any]):
        """Load assigned Transformer blocks onto this worker device."""
        if HAS_TORCH and nn is not None:
            self.layers = nn.ModuleList(
                [block.to(self.device) if hasattr(block, "to") else block for block in blocks]
            )
        else:
            self.layers = list(blocks)
        logger.info(
            f"Loaded {len(self.layers)} blocks for layers {self.start_layer} to {self.end_layer}"
        )

    def forward(
        self,
        hidden_states: Any,
        request_id: Optional[str] = None,
        is_prefill: bool = True,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """
        Execute forward pass through assigned contiguous layers.
        Retrieves and updates KV cache states when request_id is provided.
        """
        if HAS_TORCH and isinstance(hidden_states, torch.Tensor) and hasattr(self, "device"):
            x = hidden_states.to(self.device)
        else:
            x = hidden_states

        for local_idx, layer in enumerate(self.layers):
            global_layer_idx = self.start_layer + local_idx

            # Retrieve cached past key/values if in decoding phase
            layer_past = None
            if self.kv_cache_mgr and request_id and not is_prefill:
                layer_past = self.kv_cache_mgr.get_cache(request_id, global_layer_idx)

            # Check if block supports layer_past / use_cache
            try:
                if layer_past is not None:
                    output = layer(x, layer_past=layer_past, use_cache=True, attention_mask=attention_mask)
                else:
                    # Prefill or standard pass
                    output = layer(x, attention_mask=attention_mask)
            except TypeError:
                # Fallback for simple/mock modules that only accept input tensor
                output = layer(x)

            # Handle HF block output format: (hidden_states, presents, ...)
            if isinstance(output, tuple):
                x = output[0]
                if len(output) > 1 and self.kv_cache_mgr and request_id:
                    presents = output[1]
                    if presents is not None:
                        self.kv_cache_mgr.update_cache(request_id, global_layer_idx, presents)
            else:
                x = output

        return x

    def clear_cache(self, request_id: str):
        """Free KV cache associated with a finished request on this partition."""
        if self.kv_cache_mgr:
            self.kv_cache_mgr.clear_request(request_id)
