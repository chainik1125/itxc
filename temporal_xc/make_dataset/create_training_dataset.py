"""Create training dataset for temporal probes with k-token-ahead prediction within chunks."""

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
class TrainingExample:
    """Single training example for probe training."""
    problem_id: str
    chunk_idx: int
    src_token_idx: int  # Position within chunk
    tgt_token_idx: int  # Position within chunk (src + k)
    src_activation: torch.Tensor  # [hidden_dim]
    tgt_activation: torch.Tensor  # [hidden_dim]
    metadata: Dict[str, Any]


@dataclass
class ProbeTrainingDataset:
    """Dataset for training temporal probes."""
    examples: List[TrainingExample]
    config: Dict[str, Any]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Return (src_activation, tgt_activation) pair."""
        ex = self.examples[idx]
        return ex.src_activation, ex.tgt_activation

    def get_splits(self, train_ratio: float = 0.8, seed: int = 42):
        """Split into train/val sets."""
        np.random.seed(seed)
        n = len(self.examples)
        indices = np.random.permutation(n)
        train_size = int(n * train_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_examples = [self.examples[i] for i in train_indices]
        val_examples = [self.examples[i] for i in val_indices]

        return (ProbeTrainingDataset(train_examples, self.config),
                ProbeTrainingDataset(val_examples, self.config))

    def save(self, path: str):
        """Save dataset to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'examples': self.examples, 'config': self.config}, f)

    @classmethod
    def load(cls, path: str):
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['examples'], data['config'])


def create_training_dataset(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    problems: List[Any],  # List of ProblemRecord objects
    layer: int,
    k: int = 1,  # Predict k tokens ahead within same chunk
    hook_name: str = "resid_post",  # Which activation to use
    max_problems: Optional[int] = None,
    max_examples_per_problem: int = 100,
    min_chunk_length: int = 5,  # Skip chunks shorter than this
    verbosity: int = 1
) -> ProbeTrainingDataset:
    """
    Create training dataset for k-token-ahead prediction within reasoning chunks.

    Args:
        model: TransformerLens HookedTransformer
        tokenizer: Tokenizer
        problems: List of ProblemRecord objects
        layer: Which layer to extract activations from
        k: Number of tokens ahead to predict (within same chunk)
        hook_name: Type of activation ("resid_post", "resid_mid", "mlp_out", etc.)
        max_problems: Limit number of problems to process
        max_examples_per_problem: Max training examples per problem
        min_chunk_length: Minimum tokens in chunk to use
        verbosity: Logging level

    Returns:
        ProbeTrainingDataset with token t -> token t+k pairs
    """
    if verbosity > 0:
        print("\n" + "="*80)
        print(f"Creating Training Dataset for Layer {layer}")
        print("="*80)
        print(f"Task: Predict {k} token(s) ahead within same reasoning chunk")
        print(f"Hook: blocks.{layer}.hook_{hook_name}")
        print(f"Min chunk length: {min_chunk_length} tokens")

    # Determine hook point
    if hook_name == "resid_post":
        hook_point = f"blocks.{layer}.hook_resid_post"
    elif hook_name == "resid_mid":
        hook_point = f"blocks.{layer}.hook_resid_mid"
    elif hook_name == "resid_pre":
        hook_point = f"blocks.{layer}.hook_resid_pre"
    elif hook_name == "mlp_out":
        hook_point = f"blocks.{layer}.hook_mlp_out"
    elif hook_name == "attn_out":
        hook_point = f"blocks.{layer}.hook_attn_out"
    else:
        hook_point = f"blocks.{layer}.hook_{hook_name}"

    training_examples = []
    device = str(model.cfg.device) if hasattr(model.cfg, 'device') else 'cuda'

    # Process each problem
    problems_to_process = problems[:max_problems] if max_problems else problems

    for prob_idx, problem in enumerate(tqdm(problems_to_process, desc="Processing problems")):
        if verbosity > 1:
            print(f"\n[{prob_idx+1}/{len(problems_to_process)}] {problem.meta.problem_id}")

        examples_this_problem = 0

        # Process each chunk
        for chunk in problem.chunks:
            if examples_this_problem >= max_examples_per_problem:
                break

            # Tokenize the chunk
            tokens = tokenizer(
                chunk.text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )['input_ids']

            seq_len = tokens.shape[1]

            # Skip if chunk is too short
            if seq_len < min_chunk_length or seq_len <= k:
                if verbosity > 2:
                    print(f"  Skipping chunk {chunk.chunk_idx}: too short ({seq_len} tokens)")
                continue

            # Get activations for this chunk
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens.to(device),
                    names_filter=[hook_point],
                    device=device
                )

            # Extract residual stream activations
            # Shape: [batch=1, seq_len, hidden_dim]
            activations = cache[hook_point].cpu().squeeze(0)  # [seq_len, hidden_dim]

            # Create training pairs: token_i -> token_{i+k}
            num_pairs = min(seq_len - k, max_examples_per_problem - examples_this_problem)

            for i in range(num_pairs):
                src_activation = activations[i]  # [hidden_dim]
                tgt_activation = activations[i + k]  # [hidden_dim]

                example = TrainingExample(
                    problem_id=problem.meta.problem_id,
                    chunk_idx=chunk.chunk_idx,
                    src_token_idx=i,
                    tgt_token_idx=i + k,
                    src_activation=src_activation,
                    tgt_activation=tgt_activation,
                    metadata={
                        'chunk_text': chunk.text[:100],  # Store snippet
                        'chunk_tags': chunk.function_tags,
                        'importance_score': chunk.resampling_importance_accuracy or 0.0,
                        'layer': layer,
                        'k': k,
                        'hook_name': hook_name,
                        'seq_len': seq_len,
                    }
                )
                training_examples.append(example)
                examples_this_problem += 1

            if verbosity > 2:
                print(f"  Chunk {chunk.chunk_idx}: {num_pairs} pairs from {seq_len} tokens")

    if verbosity > 0:
        print(f"\n✓ Created {len(training_examples)} training examples")

        # Statistics
        problems_seen = set(ex.problem_id for ex in training_examples)
        chunks_seen = set((ex.problem_id, ex.chunk_idx) for ex in training_examples)

        print(f"  Problems: {len(problems_seen)}")
        print(f"  Unique chunks: {len(chunks_seen)}")
        print(f"  Average examples per chunk: {len(training_examples) / len(chunks_seen):.1f}")

        # Show activation shape
        if training_examples:
            print(f"  Activation shape: {training_examples[0].src_activation.shape}")

    config = {
        'layer': layer,
        'k': k,
        'hook_name': hook_name,
        'num_examples': len(training_examples),
        'num_problems': len(problems_seen),
        'min_chunk_length': min_chunk_length,
    }

    return ProbeTrainingDataset(training_examples, config)


def create_multi_k_dataset(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    problems: List[Any],
    layer: int,
    k_values: List[int] = [1, 2, 4, 8],
    **kwargs
) -> Dict[int, ProbeTrainingDataset]:
    """
    Create datasets for multiple k values to test temporal distance effects.

    Returns:
        Dictionary mapping k -> ProbeTrainingDataset
    """
    datasets = {}

    for k in k_values:
        print(f"\n--- Creating dataset for k={k} ---")
        datasets[k] = create_training_dataset(
            model, tokenizer, problems, layer, k=k, **kwargs
        )

    return datasets