"""Create dataset for k=10 token-ahead prediction."""

import torch
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class TrainingExample:
    """Single training example."""
    src_activation: torch.Tensor
    tgt_activation: torch.Tensor
    src_token_idx: int
    tgt_token_idx: int
    metadata: Dict[str, Any]

class ProbeTrainingDataset:
    """Dataset for probe training."""

    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example.src_activation, example.tgt_activation

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


def create_dataset_for_k10():
    """Create dataset for k=10 prediction."""

    k = 10
    layer = 19

    print(f"Creating dataset for k={k}, layer={layer}")

    # Load saved activations
    base_dir = Path("large_files/activations")
    all_examples = []

    # Process each saved problem
    for activation_file in sorted(base_dir.glob("problem_*_activations.pkl")):
        with open(activation_file, 'rb') as f:
            data = pickle.load(f)

        activations = data['activations']  # shape: (seq_len, hidden_dim)
        tokens = data['tokens']
        chunk_text = data.get('chunk_text', '')

        seq_len = activations.shape[0]

        # Skip if chunk is too short for k=10
        if seq_len <= k:
            continue

        # Create training pairs within this chunk
        # Limit to reasonable number per chunk
        max_pairs_per_chunk = min(20, seq_len - k)

        for i in range(max_pairs_per_chunk):
            src_idx = i
            tgt_idx = i + k

            if tgt_idx >= seq_len:
                break

            example = TrainingExample(
                src_activation=torch.tensor(activations[src_idx]),
                tgt_activation=torch.tensor(activations[tgt_idx]),
                src_token_idx=src_idx,
                tgt_token_idx=tgt_idx,
                metadata={
                    'chunk_text': chunk_text,
                    'k': k,
                    'layer': layer
                }
            )
            all_examples.append(example)

    print(f"Created {len(all_examples)} training examples for k={k}")

    # Create and save dataset
    dataset = ProbeTrainingDataset(all_examples)

    save_dir = Path("large_files/training_datasets")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"training_dataset_k{k}_l{layer}.pkl"

    dataset.save(save_path)
    print(f"Saved dataset to {save_path}")

    return dataset


if __name__ == "__main__":
    create_dataset_for_k10()