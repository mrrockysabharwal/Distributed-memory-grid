"""
Partition layer forward pass execution engine.
Maintained by Member 2 (AI / Model Engineer).
"""
import torch
import torch.nn as nn
from typing import List, Optional


class PartitionExecutor:
    def __init__(self, start_layer: int, end_layer: int, device: str = "cpu"):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = torch.device(device)
        self.layers: nn.ModuleList = nn.ModuleList()

    def load_layers(self, blocks: List[nn.Module]):
        """Load assigned Transformer blocks onto this worker device."""
        self.layers = nn.ModuleList([block.to(self.device) for block in blocks])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute forward pass through the assigned contiguous layers.
        """
        x = hidden_states.to(self.device)
        for layer in self.layers:
            output = layer(x, attention_mask=attention_mask)
            # HF Transformer blocks typically return a tuple (hidden_states, ...)
            x = output[0] if isinstance(output, tuple) else output
        return x

