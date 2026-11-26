"""
Model weight partitioning and loader for HuggingFace Transformers (e.g., GPT-2, DistilGPT-2).
Maintained by Member 2 (AI / Model Engineer).
Preserves full backwards compatibility with original extract_layers() API.
"""
from typing import Tuple, List, Optional, Any, Dict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class EmbeddingStage(nn.Module):
    """
    Computes initial hidden states from token IDs and position IDs.
    Executed on Worker 1 (Stage 0).
    """
    def __init__(self, wte: nn.Module, wpe: nn.Module, drop: nn.Module):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.drop = drop

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_length: int = 0
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        seq_len = input_ids.shape[-1]

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_length, seq_len + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states


class BlockPartitionStage(nn.Module):
    """
    Executes a contiguous subset of Transformer decoder blocks (e.g. layers 0-2 or 3-5).
    """
    def __init__(self, blocks: List[nn.Module], start_layer: int, end_layer: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.start_layer = start_layer
        self.end_layer = end_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Dict[int, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, Any]]:
        present_kvs = {}
        for local_idx, layer in enumerate(self.blocks):
            global_layer_idx = self.start_layer + local_idx
            past_kv = layer_past.get(global_layer_idx) if layer_past else None

            layer_kwargs = {"use_cache": use_cache}
            if past_kv is not None:
                layer_kwargs["layer_past"] = past_kv
            if attention_mask is not None:
                layer_kwargs["attention_mask"] = attention_mask

            outputs = layer(hidden_states, **layer_kwargs)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            if use_cache and isinstance(outputs, tuple) and len(outputs) > 1:
                present_kvs[global_layer_idx] = outputs[1]

        return hidden_states, present_kvs


class HeadStage(nn.Module):
    """
    Computes final layer normalization and language modeling head projection to logits.
    Executed on the final worker in the pipeline.
    """
    def __init__(self, ln_f: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.ln_f = ln_f
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


class ModelPartitionLoader:
    def __init__(self, model_id: str = "distilgpt2", device: str = "cpu"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.tokenizer = None
        self.full_model = None

    def load_tokenizer(self):
        """Preserves original signature and behavior."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def extract_layers(self) -> Tuple[Any, list, Any]:
        """
        Original method kept 100% intact for backwards compatibility.
        Returns: (embeddings, blocks, (final_norm, lm_head))
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if hasattr(model, "transformer"):
            embeddings = (model.transformer.wte, model.transformer.wpe, model.transformer.drop)
            blocks = list(model.transformer.h)
            final_norm = model.transformer.ln_f
            lm_head = model.lm_head
            return embeddings, blocks, (final_norm, lm_head)
        raise NotImplementedError(f"Layer extraction for {type(model)} not yet implemented.")

    def extract_components(self, model: Optional[nn.Module] = None):
        """
        New modular extraction: returns typed modules ready for pipeline stages.
        """
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(self.model_id)

        model.eval()

        if hasattr(model, "transformer"):
            embeddings = EmbeddingStage(
                wte=model.transformer.wte,
                wpe=model.transformer.wpe,
                drop=model.transformer.drop,
            ).to(self.device)

            blocks = [block.to(self.device) for block in model.transformer.h]

            head = HeadStage(
                ln_f=model.transformer.ln_f,
                lm_head=model.lm_head,
            ).to(self.device)

            total_layers = len(blocks)
            return embeddings, blocks, head, total_layers

        raise NotImplementedError(f"Layer extraction for architecture {type(model)} not yet implemented.")