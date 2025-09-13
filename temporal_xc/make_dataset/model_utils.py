"""Utilities for working with TransformerLens and HuggingFace models."""

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any


def load_model_with_tl(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_transformer_lens: bool = True,
    hf_model=None  # Allow passing existing HF model
) -> tuple:
    """
    Load a model either as TransformerLens HookedTransformer or regular HuggingFace.

    Args:
        model_id: HuggingFace model ID
        device: Device to load on
        dtype: Data type for model
        use_transformer_lens: If True, load as HookedTransformer

    Returns:
        (model, tokenizer) tuple
    """
    if use_transformer_lens:
        # Try to load with TransformerLens
        try:
            # Try native TransformerLens loading first
            model = HookedTransformer.from_pretrained(
                model_id,
                device=device,
                dtype=dtype,
                default_padding_side="left"  # Important for causal models
            )
            tokenizer = model.tokenizer
            print(f"✓ Loaded {model_id} as TransformerLens HookedTransformer")
            print(f"  Model config: {model.cfg.n_layers} layers, {model.cfg.d_model} hidden dim")
            return model, tokenizer

        except Exception as e:
            print(f"TransformerLens doesn't support {model_id} natively.")
            print("Loading HuggingFace model and converting to TransformerLens format...")

            # Use existing HF model if provided, otherwise load it
            if hf_model is None:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True
                )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            # Use TransformerLens's HookedTransformer.from_pretrained_no_processing
            # This works with arbitrary HuggingFace models
            try:
                model = HookedTransformer.from_pretrained_no_processing(
                    model_name=model_id,
                    hf_model=hf_model,
                    tokenizer=tokenizer,
                    device=device,
                    dtype=dtype,
                    default_padding_side="left"
                )
                print(f"✓ Converted {model_id} to TransformerLens HookedTransformer")
                print(f"  Model config: {model.cfg.n_layers} layers, {model.cfg.d_model} hidden dim")
                return model, tokenizer
            except:
                # If that doesn't work, use our wrapper
                print("Using custom wrapper for TransformerLens-style hooks...")
                model = create_hooked_transformer_wrapper(hf_model, tokenizer, device)
                return model, tokenizer
    else:
        # Regular HuggingFace loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer


def create_hooked_transformer_wrapper(hf_model, tokenizer, device="cuda"):
    """
    Create a minimal wrapper to use HuggingFace models with TransformerLens-style hooks.

    This is a simplified wrapper - for full functionality, the model should be
    properly converted to TransformerLens format.
    """
    class HookedTransformerWrapper:
        def __init__(self, hf_model, tokenizer, device):
            self.model = hf_model
            self.tokenizer = tokenizer
            self.device = device
            self.cfg = self._create_cfg()

        def _create_cfg(self):
            """Create a config object compatible with TransformerLens."""
            class Config:
                pass

            cfg = Config()
            cfg.device = self.device
            cfg.n_layers = self.model.config.num_hidden_layers
            cfg.d_model = self.model.config.hidden_size
            cfg.n_heads = self.model.config.num_attention_heads
            cfg.d_head = cfg.d_model // cfg.n_heads
            cfg.d_vocab = self.model.config.vocab_size
            return cfg

        def run_with_cache(self, tokens, names_filter=None, device=None):
            """
            Run model and cache activations at specified hook points.

            This is a simplified version - real implementation would need
            proper hook registration and management.
            """
            if device is None:
                device = self.device

            cache = {}
            handles = []

            # Register hooks for caching
            def create_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    cache[name] = output.detach()
                return hook_fn

            # Map TransformerLens names to HuggingFace module paths
            for tl_name in names_filter or []:
                hf_path = self._tl_to_hf_path(tl_name)
                if hf_path:
                    try:
                        module = self.model
                        for part in hf_path.split('.'):
                            module = getattr(module, part) if hasattr(module, part) else module[int(part)]
                        handle = module.register_forward_hook(create_hook(tl_name))
                        handles.append(handle)
                    except:
                        pass

            # Forward pass
            with torch.no_grad():
                outputs = self.model(tokens.to(device))

            # Remove hooks
            for handle in handles:
                handle.remove()

            return outputs, cache

        def _tl_to_hf_path(self, tl_name):
            """Convert TransformerLens hook name to HuggingFace module path."""
            # Example mappings for Llama-style models
            # This needs to be adjusted based on the specific model architecture
            mappings = {
                # Pattern: blocks.{i}.hook_resid_mid -> model.layers.{i}.post_attention_layernorm
                # Pattern: blocks.{i}.hook_mlp_in -> model.layers.{i}.mlp
            }

            if "blocks." in tl_name:
                parts = tl_name.split('.')
                layer_idx = parts[1]
                hook_type = parts[2]

                if hook_type == "hook_resid_mid":
                    return f"model.layers.{layer_idx}.post_attention_layernorm"
                elif hook_type == "hook_mlp_in":
                    return f"model.layers.{layer_idx}.mlp"
                elif hook_type == "hook_resid_post":
                    return f"model.layers.{layer_idx}"
                elif hook_type == "hook_attn_out":
                    return f"model.layers.{layer_idx}.self_attn"

            return None

    return HookedTransformerWrapper(hf_model, tokenizer, device)