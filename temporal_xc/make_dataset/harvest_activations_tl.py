"""Harvest activations using TransformerLens for cleaner hook management."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ActivationPair:
    """Store activation pairs for src->tgt token prediction."""
    problem_id: str
    src_idx: int
    tgt_idx: int
    src_token_ids: torch.Tensor  # Full sequence of tokens
    tgt_token_ids: torch.Tensor  # Full sequence of tokens
    src_activations: Dict[str, torch.Tensor]  # hook_name -> activation
    tgt_activations: Dict[str, torch.Tensor]  # hook_name -> activation
    metadata: Dict[str, Any]  # anchor scores, tags, etc.


def get_tl_hook_points(layers: List[int], hook_types: List[str]) -> List[str]:
    """
    Generate TransformerLens hook point names for given layers and types.

    TransformerLens hook naming convention:
    - blocks.{layer}.hook_resid_pre: Residual stream before attention
    - blocks.{layer}.hook_resid_mid: Residual stream after attention, before MLP
    - blocks.{layer}.hook_resid_post: Residual stream after MLP
    - blocks.{layer}.hook_mlp_in: Input to MLP (after LayerNorm)
    - blocks.{layer}.hook_mlp_out: Output of MLP
    - blocks.{layer}.hook_attn_out: Output of attention
    """
    hook_points = []
    for layer in layers:
        for hook_type in hook_types:
            if hook_type == "resid_mid":
                hook_points.append(f"blocks.{layer}.hook_resid_mid")
            elif hook_type == "resid_post":
                hook_points.append(f"blocks.{layer}.hook_resid_post")
            elif hook_type == "resid_pre":
                hook_points.append(f"blocks.{layer}.hook_resid_pre")
            elif hook_type == "mlp_in":
                hook_points.append(f"blocks.{layer}.hook_mlp_in")
            elif hook_type == "mlp_out":
                hook_points.append(f"blocks.{layer}.hook_mlp_out")
            elif hook_type == "attn_out":
                hook_points.append(f"blocks.{layer}.hook_attn_out")
    return hook_points


def harvest_activations_with_cache(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_points: List[str],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Harvest activations at specified hook points using TransformerLens caching.

    Returns:
        Dict mapping hook names to activation tensors of shape [batch, seq_len, hidden_dim]
    """
    # Run model with cache
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens.to(device),
            names_filter=hook_points,  # Only cache specified hooks
            device=device
        )

    # Extract activations from cache
    activations = {}
    for hook_name in hook_points:
        if hook_name in cache:
            activations[hook_name] = cache[hook_name].cpu()

    return activations


def create_activation_dataset_tl(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    problems: List[Any],  # List of ProblemRecord objects
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    verbosity: int = 1
) -> List[ActivationPair]:
    """
    Create dataset of activation pairs using TransformerLens.

    Args:
        model: TransformerLens HookedTransformer model
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
    hook_types = harvest_config.get('hook_types', ['resid_mid', 'mlp_in'])
    max_problems = harvest_config.get('max_problems', 2)
    max_pairs_per_problem = harvest_config.get('max_pairs_per_problem', 10)
    min_anchor_score = harvest_config.get('min_anchor_score', 0.1)
    sentence_pooling = harvest_config.get('sentence_pooling', 'last')  # 'mean', 'last', or 'max'

    if verbosity > 0:
        print("\n" + "="*80)
        print("Harvesting Activations with TransformerLens")
        print("="*80)
        print(f"Layers: {layers}")
        print(f"Hook types: {hook_types}")
        print(f"Max problems: {max_problems}")
        print(f"Max pairs per problem: {max_pairs_per_problem}")
        print(f"Min anchor score: {min_anchor_score}")
        print(f"Sentence pooling: {sentence_pooling}")

    # Get all hook points
    hook_points = get_tl_hook_points(layers, hook_types)
    if verbosity > 1:
        print(f"Hook points to cache: {hook_points[:3]}... ({len(hook_points)} total)")

    activation_pairs = []
    device = str(model.cfg.device) if hasattr(model.cfg, 'device') else 'cuda'

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
            anchor_score = src_chunk.resampling_importance_accuracy or 0.0
            if anchor_score < min_anchor_score:
                continue

            # Tokenize source and target chunks
            src_tokens = tokenizer(
                src_chunk.text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False
            )['input_ids']

            tgt_tokens = tokenizer(
                tgt_chunk.text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False
            )['input_ids']

            if verbosity > 1:
                print(f"  Pair {i}->{i+1}: anchor_score={anchor_score:.3f}, src_len={src_tokens.shape[1]}, tgt_len={tgt_tokens.shape[1]}")

            # Harvest activations for source and target
            src_activations = harvest_activations_with_cache(model, src_tokens, hook_points, device)
            tgt_activations = harvest_activations_with_cache(model, tgt_tokens, hook_points, device)

            # Apply sentence pooling
            pooled_src_activations = {}
            pooled_tgt_activations = {}

            for hook_name in src_activations.keys():
                src_act = src_activations[hook_name]  # [1, seq_len, hidden_dim]
                tgt_act = tgt_activations[hook_name]  # [1, seq_len, hidden_dim]

                # Pool over sequence dimension
                if sentence_pooling == 'mean':
                    pooled_src = src_act.mean(dim=1)  # [1, hidden_dim]
                    pooled_tgt = tgt_act.mean(dim=1)
                elif sentence_pooling == 'last':
                    pooled_src = src_act[:, -1, :]  # [1, hidden_dim]
                    pooled_tgt = tgt_act[:, 0, :]  # First token of target
                elif sentence_pooling == 'max':
                    pooled_src = src_act.max(dim=1)[0]  # [1, hidden_dim]
                    pooled_tgt = tgt_act.max(dim=1)[0]
                else:
                    # No pooling - keep full sequence
                    pooled_src = src_act
                    pooled_tgt = tgt_act

                pooled_src_activations[hook_name] = pooled_src.squeeze(0)  # Remove batch dim
                pooled_tgt_activations[hook_name] = pooled_tgt.squeeze(0)

            activation_pair = ActivationPair(
                problem_id=problem.meta.problem_id,
                src_idx=src_chunk.chunk_idx,
                tgt_idx=tgt_chunk.chunk_idx,
                src_token_ids=src_tokens.squeeze(0),  # Remove batch dim
                tgt_token_ids=tgt_tokens.squeeze(0),
                src_activations=pooled_src_activations,
                tgt_activations=pooled_tgt_activations,
                metadata={
                    'anchor_score': anchor_score,
                    'src_tags': src_chunk.function_tags,
                    'tgt_tags': tgt_chunk.function_tags,
                    'src_text': src_chunk.text[:100],  # Store snippet for debugging
                    'tgt_text': tgt_chunk.text[:100],
                    'pooling': sentence_pooling,
                }
            )
            activation_pairs.append(activation_pair)
            pairs_collected += 1

    if verbosity > 0:
        print(f"\n✓ Collected {len(activation_pairs)} activation pairs")

        # Show statistics
        hook_names_seen = set()
        problems_seen = set()
        for pair in activation_pairs:
            hook_names_seen.update(pair.src_activations.keys())
            problems_seen.add(pair.problem_id)

        print(f"  Problems: {len(problems_seen)}")
        print(f"  Unique hooks: {len(hook_names_seen)}")

        # Show activation shapes
        if activation_pairs:
            sample_pair = activation_pairs[0]
            for hook_name in list(sample_pair.src_activations.keys())[:2]:
                act_shape = sample_pair.src_activations[hook_name].shape
                print(f"  {hook_name}: shape={act_shape}")

    # Save dataset if output directory is specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as pickle for full data
        pickle_path = output_path / "activation_dataset_tl.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(activation_pairs, f)

        # Save metadata as JSON for inspection
        metadata = {
            'num_pairs': len(activation_pairs),
            'problems': list(problems_seen),
            'hook_points': sorted(list(hook_names_seen)),
            'config': harvest_config,
            'pooling': sentence_pooling,
        }
        json_path = output_path / "activation_dataset_tl_meta.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if verbosity > 0:
            print(f"\n📁 Dataset saved to {output_dir}/")
            print(f"  - activation_dataset_tl.pkl ({pickle_path.stat().st_size / 1024:.1f} KB)")
            print(f"  - activation_dataset_tl_meta.json")

    return activation_pairs