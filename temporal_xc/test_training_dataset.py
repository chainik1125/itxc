"""Quick test to verify training datasets are working."""

import torch
from pathlib import Path
from temporal_xc.make_dataset import ProbeTrainingDataset

# Load a dataset
dataset_path = Path("large_files/training_datasets/training_dataset_k1_l19.pkl")
dataset = ProbeTrainingDataset.load(str(dataset_path))

print(f"Dataset loaded: {len(dataset)} examples")
print(f"Config: {dataset.config}")

# Get first example
src_act, tgt_act = dataset[0]
print(f"\nFirst example:")
print(f"  Source activation shape: {src_act.shape}")
print(f"  Target activation shape: {tgt_act.shape}")
print(f"  Activation dtype: {src_act.dtype}")

# Test train/val split
train_ds, val_ds = dataset.get_splits(train_ratio=0.8)
print(f"\nTrain/Val split:")
print(f"  Train: {len(train_ds)} examples")
print(f"  Val: {len(val_ds)} examples")

# Check a few examples
print(f"\nExample metadata:")
for i in range(min(3, len(dataset.examples))):
    ex = dataset.examples[i]
    print(f"  [{i}] {ex.problem_id} chunk_{ex.chunk_idx}: "
          f"token {ex.src_token_idx} -> {ex.tgt_token_idx}, "
          f"importance={ex.metadata['importance_score']:.2f}")

# Verify activations are valid
print(f"\nActivation statistics:")
all_src = torch.stack([ex.src_activation for ex in dataset.examples[:10]])
all_tgt = torch.stack([ex.tgt_activation for ex in dataset.examples[:10]])
print(f"  Source mean: {all_src.mean():.4f}, std: {all_src.std():.4f}")
print(f"  Target mean: {all_tgt.mean():.4f}, std: {all_tgt.std():.4f}")
print(f"  Correlation: {torch.corrcoef(torch.stack([all_src.flatten(), all_tgt.flatten()]))[0,1]:.4f}")

print("\n✅ Dataset test passed!")