"""Harvest activations from transformer models for probe training."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ActivationPair:
    """Store activation pairs for src->tgt token prediction."""
    problem_id: str
    src_idx: int
    tgt_idx: int
    src_token_id: int
    tgt_token_id: int
    src_activations: Dict[str, torch.Tensor]  # hook_name -> activation
    tgt_activations: Dict[str, torch.Tensor]  # hook_name -> activation
    metadata: Dict[str, Any]  # anchor scores, tags, etc.


def get_hook_points(layer: int, hook_types: List[str]) -> List[str]:
    """Generate hook point names for a given layer."""
    hook_points = []
    for hook_type in hook_types:
        if hook_type == "resid_mid":
            hook_points.append(f"model.layers.{layer}.post_attention_layernorm")
        elif hook_type == "mlp_pre":
            hook_points.append(f"model.layers.{layer}.mlp")
        elif hook_type == "attn_out":
            hook_points.append(f"model.layers.{layer}.self_attn")
        elif hook_type == "resid_post":
            if layer < 31:  # Assuming 32 layers total
                hook_points.append(f"model.layers.{layer + 1}")
            else:
                hook_points.append("model.norm")  # Final layer norm
    return hook_points


def harvest_activations_for_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    hook_points: List[str],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Harvest activations at specified hook points for a sequence."""
    activations = {}
    handles = []

    def create_hook(name):
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().cpu()
        return hook_fn

    # Register hooks
    for hook_point in hook_points:
        try:
            # Navigate to the module
            module = model
            for part in hook_point.split('.'):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)

            handle = module.register_forward_hook(create_hook(hook_point))
            handles.append(handle)
        except (AttributeError, IndexError) as e:
            print(f"Warning: Could not register hook at {hook_point}: {e}")

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids.to(device))

    # Remove hooks
    for handle in handles:
        handle.remove()

    return activations


def create_activation_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: List[Any],  # List of ProblemRecord objects
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    verbosity: int = 1
) -> List[ActivationPair]:
    """
    Create dataset of activation pairs for probe training.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        problems: List of ProblemRecord objects
        config: Configuration dict with dataset settings
        output_dir: Directory to save the dataset
        verbosity: Logging verbosity level

    Returns:
        List of ActivationPair objects
    """
    # Get configuration
    dataset_config = config.get('dataset', {})
    harvest_config = dataset_config.get('activation_harvest', {})

    # Default configuration
    layers = harvest_config.get('layers', [15, 16, 17, 18, 19])  # Middle layers
    hook_types = harvest_config.get('hook_types', ['resid_mid', 'mlp_pre'])
    max_problems = harvest_config.get('max_problems', 2)
    max_pairs_per_problem = harvest_config.get('max_pairs_per_problem', 10)
    min_anchor_score = harvest_config.get('min_anchor_score', 0.1)

    if verbosity > 0:
        print("\n" + "="*80)
        print("Harvesting Activations for Probe Training")
        print("="*80)
        print(f"Layers: {layers}")
        print(f"Hook types: {hook_types}")
        print(f"Max problems: {max_problems}")
        print(f"Max pairs per problem: {max_pairs_per_problem}")
        print(f"Min anchor score: {min_anchor_score}")

    activation_pairs = []
    device = str(model.device) if hasattr(model, 'device') else 'cuda'

    # Process each problem
    for prob_idx, problem in enumerate(problems[:max_problems]):
        if verbosity > 0:
            print(f"\n[{prob_idx+1}/{min(len(problems), max_problems)}] Processing {problem.meta.problem_id}...")

        # Get all chunks for this problem
        chunks = problem.chunks

        # Find high-importance chunk pairs (src -> tgt)
        pairs_collected = 0
        for i in range(len(chunks) - 1):
            if pairs_collected >= max_pairs_per_problem:
                break

            src_chunk = chunks[i]
            tgt_chunk = chunks[i + 1]

            # Check if this is an important transition
            anchor_score = src_chunk.importance_metrics.get('resampling_importance_accuracy', 0.0)
            if anchor_score < min_anchor_score:
                continue

            # Tokenize source and target chunks
            src_tokens = tokenizer(src_chunk.text, return_tensors="pt", truncation=True, max_length=512)
            tgt_tokens = tokenizer(tgt_chunk.text, return_tensors="pt", truncation=True, max_length=512)

            # For each layer, harvest activations
            for layer in layers:
                hook_points = get_hook_points(layer, hook_types)

                if verbosity > 1:
                    print(f"  Layer {layer}: Harvesting from {src_chunk.chunk_idx}->{tgt_chunk.chunk_idx}")

                # Get activations for source
                src_activations = harvest_activations_for_sequence(
                    model, tokenizer, src_tokens['input_ids'], hook_points, device
                )

                # Get activations for target
                tgt_activations = harvest_activations_for_sequence(
                    model, tokenizer, tgt_tokens['input_ids'], hook_points, device
                )

                # For token-level pairs, we'll take the last token of src and first token of tgt
                # (In future, we can do sentence averaging here)
                for hook_name in src_activations.keys():
                    src_act = src_activations[hook_name]
                    tgt_act = tgt_activations[hook_name]

                    # Get last token of source, first token of target
                    src_last_token_act = src_act[0, -1, :]  # [hidden_dim]
                    tgt_first_token_act = tgt_act[0, 0, :]  # [hidden_dim]

                    activation_pair = ActivationPair(
                        problem_id=problem.meta.problem_id,
                        src_idx=src_chunk.chunk_idx,
                        tgt_idx=tgt_chunk.chunk_idx,
                        src_token_id=src_tokens['input_ids'][0, -1].item(),
                        tgt_token_id=tgt_tokens['input_ids'][0, 0].item(),
                        src_activations={f"layer_{layer}_{hook_name}": src_last_token_act},
                        tgt_activations={f"layer_{layer}_{hook_name}": tgt_first_token_act},
                        metadata={
                            'anchor_score': anchor_score,
                            'layer': layer,
                            'hook_type': hook_name,
                            'src_tags': src_chunk.function_tags,
                            'tgt_tags': tgt_chunk.function_tags,
                        }
                    )
                    activation_pairs.append(activation_pair)

            pairs_collected += 1

    if verbosity > 0:
        print(f"\n✓ Collected {len(activation_pairs)} activation pairs")

        # Show statistics
        layers_seen = set()
        hook_types_seen = set()
        problems_seen = set()
        for pair in activation_pairs:
            layers_seen.add(pair.metadata['layer'])
            hook_types_seen.add(pair.metadata['hook_type'])
            problems_seen.add(pair.problem_id)

        print(f"  Problems: {len(problems_seen)}")
        print(f"  Layers: {sorted(layers_seen)}")
        print(f"  Hook types: {len(hook_types_seen)}")

    # Save dataset if output directory is specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as pickle for full data
        pickle_path = output_path / "activation_dataset.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(activation_pairs, f)

        # Save metadata as JSON for inspection
        metadata = {
            'num_pairs': len(activation_pairs),
            'problems': list(set(p.problem_id for p in activation_pairs)),
            'layers': sorted(list(layers_seen)),
            'hook_types': sorted(list(hook_types_seen)),
            'config': harvest_config
        }
        json_path = output_path / "activation_dataset_meta.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if verbosity > 0:
            print(f"\n📁 Dataset saved to {output_dir}/")
            print(f"  - activation_dataset.pkl ({pickle_path.stat().st_size / 1024:.1f} KB)")
            print(f"  - activation_dataset_meta.json")

    return activation_pairs