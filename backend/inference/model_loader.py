"""
Model weight partitioning and loader for Transformer architectures (GPT-2, DistilGPT-2, LLaMA).
Extracts embedding projections, slices contiguous layer subsets for workers, and handles LM head.

Maintained by Member 2 (AI / Model Engineer).
"""
import logging
from typing import Tuple, List, Optional, Any

logger = logging.getLogger("DMG-ModelLoader")

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    HAS_TORCH = False


class SyntheticBlock:
    """Lightweight synthetic transformer block for testing and CPU fallback environments."""
    def __init__(self, layer_idx: int, hidden_dim: int = 64):
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim

    def __call__(self, x: Any, layer_past: Optional[Any] = None, use_cache: bool = True, **kwargs) -> Tuple[Any, Any]:
        # Return (transformed_activations, presents)
        presents = (x, x) if use_cache else None
        return x, presents


class ModelPartitionLoader:
    """
    Loads pretrained Transformer checkpoints and extracts contiguous subsets
    of decoder blocks for distributed worker nodes.
    """
    def __init__(self, model_id: str = "distilgpt2", device: str = "cpu"):
        self.model_id = model_id
        self.device_str = device
        self.device = torch.device(device) if HAS_TORCH else "cpu"
        self.tokenizer = None
        self.full_model = None

    def load_tokenizer(self) -> Any:
        """Load HuggingFace AutoTokenizer or fallback tokenizer."""
        if HAS_TORCH and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                return self.tokenizer
            except Exception as e:
                logger.warning(f"Could not load HuggingFace tokenizer '{self.model_id}': {e}. Using fallback.")

        # Fallback Mock Tokenizer
        class FallbackTokenizer:
            eos_token_id = 50256
            pad_token_id = 50256
            def encode(self, text: str, **kwargs):
                return [ord(c) % 50000 for c in text] if text else [1, 2]
            def decode(self, token_ids: List[int], **kwargs) -> str:
                return "".join(chr(t % 128) if 32 <= (t % 128) <= 126 else " " for t in token_ids)

        self.tokenizer = FallbackTokenizer()
        return self.tokenizer

    def extract_layers(self, total_fallback_layers: int = 12) -> Tuple[Any, List[Any], Any]:
        """
        Extract embeddings, list of transformer decoder blocks, and final lm_head.
        """
        if HAS_TORCH and AutoModelForCausalLM is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_id)
                # GPT-2 / DistilGPT-2 style
                if hasattr(model, "transformer"):
                    embeddings = (model.transformer.wte, model.transformer.wpe, model.transformer.drop)
                    blocks = list(model.transformer.h)
                    final_norm = model.transformer.ln_f
                    lm_head = model.lm_head
                    return embeddings, blocks, (final_norm, lm_head)
                # LLaMA / Mistral style
                elif hasattr(model, "model") and hasattr(model.model, "layers"):
                    embeddings = model.model.embed_tokens
                    blocks = list(model.model.layers)
                    final_norm = model.model.norm
                    lm_head = model.lm_head
                    return embeddings, blocks, (final_norm, lm_head)
            except Exception as e:
                logger.warning(f"Could not load checkpoint '{self.model_id}': {e}. Creating synthetic blocks.")

        # Synthetic fallback for CPU testing environments
        blocks = [SyntheticBlock(layer_idx=i) for i in range(total_fallback_layers)]
        return ("synthetic_embeddings", blocks, ("synthetic_norm", "synthetic_head"))

    def get_worker_partition_blocks(self, start_layer: int, end_layer: int) -> List[Any]:
        """
        Extract only the subset of Transformer blocks assigned to a specific worker.
        """
        _, all_blocks, _ = self.extract_layers()
        if start_layer < 0 or end_layer >= len(all_blocks) or start_layer > end_layer:
            raise ValueError(
                f"Invalid layer range [{start_layer}, {end_layer}] for model with {len(all_blocks)} layers."
            )
        sliced = all_blocks[start_layer : end_layer + 1]
        logger.info(f"Extracted {len(sliced)} blocks for partition [{start_layer}, {end_layer}]")
        return sliced
