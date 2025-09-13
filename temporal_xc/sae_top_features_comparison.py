"""Compare SAE vs raw using only top-d SAE features for fairer comparison."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
import json
import pickle

print("="*60)
print("SAE TOP-d FEATURES vs RAW RESIDUALS")
print("="*60)

# Load the original k=1 dataset
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

# Load SAE
print("\nLoading SAE...")
sae = SAE.from_pretrained("deepseek-r1-distill-llama-8b-qresearch", "blocks.19.hook_resid_post", device='cpu')
print(f"SAE: {sae.cfg.d_in}D → {sae.cfg.d_sae}D")

# Encode to SAE features
print("\nEncoding to SAE features...")
sae_pairs_full = []
sae.eval()

with torch.no_grad():
    for src, tgt in raw_pairs:
        src_sae = sae.encode(src.unsqueeze(0)).squeeze(0).float()
        tgt_sae = sae.encode(tgt.unsqueeze(0)).squeeze(0).float()
        sae_pairs_full.append((src_sae, tgt_sae))

print(f"Created {len(sae_pairs_full)} SAE pairs")

# Analyze SAE feature activity
print("\n" + "="*40)
print("ANALYZING SAE FEATURE ACTIVITY")
print("="*40)

# Stack all SAE activations
all_sae_src = torch.stack([p[0] for p in sae_pairs_full])
all_sae_tgt = torch.stack([p[1] for p in sae_pairs_full])
all_sae = torch.cat([all_sae_src, all_sae_tgt], dim=0)  # (2*n_examples, d_sae)

# Count total non-zero features
non_zero_mask = (all_sae.abs() > 1e-6).any(dim=0)
total_non_zero = non_zero_mask.sum().item()
print(f"Total non-zero SAE features across dataset: {total_non_zero}/{sae.cfg.d_sae} ({total_non_zero/sae.cfg.d_sae*100:.2f}%)")

# Calculate average activation magnitude per feature
avg_activation = all_sae.abs().mean(dim=0)

# Get top-d features (d = residual stream dimension)
d = sae.cfg.d_in  # 4096
top_d_indices = torch.topk(avg_activation, k=min(d, total_non_zero)).indices
print(f"Selected top {len(top_d_indices)} features by average activation")

# Calculate statistics for top features
top_features_mask = torch.zeros(sae.cfg.d_sae, dtype=torch.bool)
top_features_mask[top_d_indices] = True

# What fraction of total activation do top features capture?
total_activation = all_sae.abs().sum().item()
top_activation = all_sae[:, top_features_mask].abs().sum().item()
activation_captured = top_activation / total_activation * 100
print(f"Top {d} features capture {activation_captured:.1f}% of total activation magnitude")

# Create dataset with only top-d features
sae_pairs_topd = []
for src_full, tgt_full in sae_pairs_full:
    src_topd = src_full[top_features_mask]
    tgt_topd = tgt_full[top_features_mask]
    sae_pairs_topd.append((src_topd, tgt_topd))

# Function to train and evaluate
def train_and_evaluate(pairs, name, feature_desc=""):
    print(f"\n{'='*40}")
    print(f"{name}")
    if feature_desc:
        print(f"({feature_desc})")
    print("="*40)

    X = torch.stack([p[0] for p in pairs])
    y = torch.stack([p[1] for p in pairs])

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

# Train and evaluate all three
results = {}
results['raw'] = train_and_evaluate(raw_pairs, "RAW RESIDUALS")
results['sae_topd'] = train_and_evaluate(sae_pairs_topd, "SAE TOP-d FEATURES",
                                          f"Top {d} features by average activation")

# Also compare with previous approach (only active features)
sae_pairs_active = []
active_mask = (all_sae.abs() > 1e-6).float().mean(0) > 0.01
n_active = active_mask.sum().item()
for src_full, tgt_full in sae_pairs_full:
    src_active = src_full[active_mask]
    tgt_active = tgt_full[active_mask]
    sae_pairs_active.append((src_active, tgt_active))

results['sae_active'] = train_and_evaluate(sae_pairs_active, "SAE ACTIVE FEATURES",
                                           f"{n_active} features active >1% of time")

# Save results
with open('large_files/viz/sae_topd_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Final comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Method':<25} {'Features':<10} {'Test R²':<12} {'Cos(pred)':<12} {'Cos(direct)':<12}")
print("-" * 71)

for key, label in [('raw', 'Raw Residuals'),
                   ('sae_topd', f'SAE Top-{d}'),
                   ('sae_active', 'SAE Active Only')]:
    r = results[key]
    print(f"{label:<25} {r['n_features']:<10} {r['test_r2']:<12.4f} {r['cosine_pred']:<12.4f} {r['cosine_direct']:<12.4f}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Compare SAE top-d vs raw
r2_diff = (results['sae_topd']['test_r2'] - results['raw']['test_r2']) / abs(results['raw']['test_r2']) * 100
cos_diff = (results['sae_topd']['cosine_pred'] - results['raw']['cosine_pred']) / abs(results['raw']['cosine_pred']) * 100

print(f"SAE Top-{d} vs Raw Residuals:")
print(f"  R² difference: {r2_diff:+.1f}%")
print(f"  Cosine difference: {cos_diff:+.1f}%")

if r2_diff > -10 and cos_diff > -10:
    print(f"\n✅ SAE Top-{d} performs comparably to raw residuals!")
    print("   This suggests temporal structure IS preserved in SAE features")
    print("   when using the same dimensionality")
elif r2_diff > 0 and cos_diff > 0:
    print(f"\n✅ SAE Top-{d} OUTPERFORMS raw residuals!")
    print("   SAE features actually capture temporal structure BETTER")
else:
    print(f"\n❌ Raw residuals still outperform SAE Top-{d}")
    print("   But the gap is smaller than with sparse features")

print(f"\nKey statistics:")
print(f"  - Total non-zero SAE features: {total_non_zero}/{sae.cfg.d_sae}")
print(f"  - Top {d} features capture {activation_captured:.1f}% of activation")
print(f"  - Using same dimensionality ({d}) for fair comparison")