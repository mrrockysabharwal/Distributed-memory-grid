"""
Model weight partitioning and loader for HuggingFace Transformers (e.g., GPT-2, DistilGPT-2).
Maintained by Member 2 (AI / Model Engineer).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any


class ModelPartitionLoader:
    def __init__(self, model_id: str = "distilgpt2", device: str = "cpu"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.tokenizer = None
        self.full_model = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def extract_layers(self) -> Tuple[Any, list, Any]:
        """
        Extract embeddings, list of transformer decoder blocks, and final lm_head.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        # Adaptable for GPT-2 / DistilGPT-2 architecture
        if hasattr(model, "transformer"):
            embeddings = (model.transformer.wte, model.transformer.wpe, model.transformer.drop)
            blocks = list(model.transformer.h)
            final_norm = model.transformer.ln_f
            lm_head = model.lm_head
            return embeddings, blocks, (final_norm, lm_head)
        raise NotImplementedError(f"Layer extraction for {type(model)} not yet implemented.")

