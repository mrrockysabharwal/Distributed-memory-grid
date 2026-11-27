"""
Partition layer forward pass execution engine.
Maintained by Member 2 (AI / Model Engineer).
Preserves full backwards compatibility with original PartitionExecutor API.
"""
from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from backend.inference.model_loader import EmbeddingStage, BlockPartitionStage, HeadStage


class PartitionExecutor:
    def __init__(
        self,
        start_layer: int,
        end_layer: int,
        device: str = "cpu",
        total_layers: Optional[int] = None,
        embedding_stage: Optional[EmbeddingStage] = None,
        block_stage: Optional[BlockPartitionStage] = None,
        head_stage: Optional[HeadStage] = None,
    ):
        # Original fields
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = torch.device(device)
        self.layers: nn.ModuleList = nn.ModuleList()

        # New stage-aware fields
        self.total_layers = total_layers if total_layers is not None else (end_layer + 1)
        self.is_first_stage = (start_layer == 0)
        self.is_final_stage = (end_layer >= self.total_layers - 1)

        self.embedding_stage = embedding_stage
        self.block_stage = block_stage
        self.head_stage = head_stage

    def load_layers(self, blocks: List[nn.Module]):
        """Original method: Load assigned Transformer blocks onto this worker device."""
        self.layers = nn.ModuleList([block.to(self.device) for block in blocks])
        self.block_stage = BlockPartitionStage(list(self.layers), self.start_layer, self.end_layer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Dict[int, Any]] = None,
        use_cache: bool = False,
        past_length: int = 0,
    ) -> torch.Tensor:
        """
        Original forward() contract preserved:
        Takes an input tensor and returns the transformed tensor.
        Also handles token IDs if passed on stage 0, and head projection if on final stage.
        """
        x = hidden_states.to(self.device)

        # 1. Embed tokens if first stage and input is token IDs
        if self.is_first_stage and x.dtype in (torch.int64, torch.int32):
            if self.embedding_stage is not None:
                x = self.embedding_stage(x, past_length=past_length)

        # 2. Forward through blocks (uses block_stage or fallback to self.layers)
        if self.block_stage is not None:
            x, _ = self.block_stage(
                x,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
        else:
            for layer in self.layers:
                output = layer(x, attention_mask=attention_mask)
                x = output[0] if isinstance(output, tuple) else output

        # 3. Final projection to logits if final stage
        if self.is_final_stage and self.head_stage is not None:
            x = self.head_stage(x)

        return x

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Dict[int, Any]] = None,
        use_cache: bool = True,
        past_length: int = 0,
    ) -> Tuple[torch.Tensor, Dict[int, Any], bool]:
        """
        Extended method for Member 3's KV Cache integration.
        Returns (output_tensor, present_kvs_dict, is_final_stage).
        """
        x = hidden_states.to(self.device)

        if self.is_first_stage and x.dtype in (torch.int64, torch.int32):
            if self.embedding_stage is not None:
                x = self.embedding_stage(x, past_length=past_length)

        present_kvs = {}
        if self.block_stage is not None:
            x, present_kvs = self.block_stage(
                x,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
        else:
            for layer in self.layers:
                output = layer(x, attention_mask=attention_mask)
                x = output[0] if isinstance(output, tuple) else output

        if self.is_final_stage and self.head_stage is not None:
            x = self.head_stage(x)

        return x, present_kvs, self.is_final_stage