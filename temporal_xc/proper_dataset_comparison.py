"""Use the SAME dataset that gave R²=0.85 to properly compare SAE vs raw."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
import json
import pickle

print("="*60)
print("PROPER COMPARISON WITH ORIGINAL HIGH-R² DATASET")
print("="*60)

# Load the original k=1 dataset that gave us R²=0.85
print("\nLoading original training dataset...")
with open('large_files/training_datasets/training_dataset_k1_l19.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"Dataset has {len(dataset['examples'])} examples")

# Extract examples
examples = dataset['examples']

# Get raw activations
raw_pairs = []
for ex in examples:
    src = ex['src_activation'] if isinstance(ex, dict) else ex.src_activation
    tgt = ex['tgt_activation'] if isinstance(ex, dict) else ex.tgt_activation

    # Convert to tensors and ensure float
    if not isinstance(src, torch.Tensor):
        src = torch.tensor(src)
    if not isinstance(tgt, torch.Tensor):
        tgt = torch.tensor(tgt)

    raw_pairs.append((src.float(), tgt.float()))

print(f"Collected {len(raw_pairs)} raw pairs")

# Check cosine similarity to verify this is the right dataset
sample_cosines = []
for i in range(min(10, len(raw_pairs))):
    src, tgt = raw_pairs[i]
    cos = torch.nn.functional.cosine_similarity(src.unsqueeze(0), tgt.unsqueeze(0)).item()
    sample_cosines.append(cos)

print(f"Sample direct cosine similarities: {np.mean(sample_cosines):.4f} ± {np.std(sample_cosines):.4f}")

# Load SAE
print("\nLoading SAE...")
sae = SAE.from_pretrained("deepseek-r1-distill-llama-8b-qresearch", "blocks.19.hook_resid_post", device='cpu')
print(f"SAE: {sae.cfg.d_in}D → {sae.cfg.d_sae}D")

# Encode to SAE features
print("\nEncoding to SAE features...")
sae_pairs = []
sae.eval()

with torch.no_grad():
    for src, tgt in raw_pairs:
        src_sae = sae.encode(src.unsqueeze(0)).squeeze(0).float()
        tgt_sae = sae.encode(tgt.unsqueeze(0)).squeeze(0).float()
        sae_pairs.append((src_sae, tgt_sae))

print(f"Created {len(sae_pairs)} SAE pairs")

# Function to train and evaluate
def train_and_evaluate(pairs, name):
    print(f"\n{'='*40}")
    print(f"{name}")
    print("="*40)

    X = torch.stack([p[0] for p in pairs])
    y = torch.stack([p[1] for p in pairs])

    # For SAE, use only active dimensions
    if "SAE" in name and X.shape[1] > 10000:
        active_mask = (X.abs() > 1e-6).float().mean(0) > 0.01
        n_active = active_mask.sum().item()
        print(f"Using {n_active}/{X.shape[1]} active dimensions ({n_active/X.shape[1]*100:.1f}%)")
        X = X[:, active_mask]
        y = y[:, active_mask]

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Split 80/20
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Train probe
    probe = LinearProbe(X.shape[1], y.shape[1])
    trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)

    print("Training probe...")
    trainer.train(30, verbose=False)

    # Evaluate
    train_loss, train_r2 = trainer.evaluate(train_loader)
    test_loss, test_r2 = trainer.evaluate(test_loader)

    # Get prediction cosine similarities
    probe.eval()
    cos_sims = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = probe(x_batch)
            for i in range(len(x_batch)):
                cos = torch.nn.functional.cosine_similarity(
                    pred[i].unsqueeze(0), y_batch[i].unsqueeze(0)
                ).item()
                cos_sims.append(cos)

    mean_cos_pred = np.mean(cos_sims)

    # Direct cosine
    direct_cos = []
    for i in range(min(10, len(X_test))):
        cos = torch.nn.functional.cosine_similarity(
            X_test[i].unsqueeze(0), y_test[i].unsqueeze(0)
        ).item()
        direct_cos.append(cos)
    mean_cos_direct = np.mean(direct_cos)

    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Cosine (predictions): {mean_cos_pred:.4f}")
    print(f"  Cosine (direct): {mean_cos_direct:.4f}")

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cosine_pred': mean_cos_pred,
        'cosine_direct': mean_cos_direct,
        'n_features': X.shape[1]
    }

# Train and evaluate both
results = {}
results['raw'] = train_and_evaluate(raw_pairs, "RAW RESIDUALS")
results['sae'] = train_and_evaluate(sae_pairs, "SAE LATENTS")

# Save results
with open('large_files/viz/proper_dataset_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Final comparison
print("\n" + "="*60)
print("FINAL COMPARISON (Using Original Dataset)")
print("="*60)
print(f"{'Metric':<30} {'Raw':<15} {'SAE':<15} {'Winner':<10}")
print("-" * 70)

metrics = [
    ('Train R²', 'train_r2'),
    ('Test R²', 'test_r2'),
    ('Cosine (predictions)', 'cosine_pred'),
    ('Cosine (direct)', 'cosine_direct')
]

for label, key in metrics:
    raw_val = results['raw'][key]
    sae_val = results['sae'][key]
    winner = "SAE" if sae_val > raw_val else "Raw"
    print(f"{label:<30} {raw_val:<15.4f} {sae_val:<15.4f} {winner:<10}")

print(f"{'Feature dimensions':<30} {results['raw']['n_features']:<15} {results['sae']['n_features']:<15}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

r2_diff = (results['sae']['test_r2'] - results['raw']['test_r2']) / abs(results['raw']['test_r2']) * 100 if results['raw']['test_r2'] != 0 else 0
cos_diff = (results['sae']['cosine_pred'] - results['raw']['cosine_pred']) / abs(results['raw']['cosine_pred']) * 100 if results['raw']['cosine_pred'] != 0 else 0

if results['sae']['test_r2'] > results['raw']['test_r2'] and results['sae']['cosine_pred'] > results['raw']['cosine_pred']:
    print("✅ SAE latents preserve temporal structure BETTER")
    print(f"   R² improvement: {r2_diff:.1f}%")
    print(f"   Cosine improvement: {cos_diff:.1f}%")
else:
    print("❌ Raw residuals preserve temporal structure BETTER")
    print(f"   R² difference: {r2_diff:.1f}%")
    print(f"   Cosine difference: {cos_diff:.1f}%")

print(f"\nExpected raw R² ~0.85 (from original experiments)")
print(f"Actual raw R²: {results['raw']['test_r2']:.3f}")
if abs(results['raw']['test_r2'] - 0.85) > 0.1:
    print("⚠️ Warning: Raw R² doesn't match expected - may be different dataset")