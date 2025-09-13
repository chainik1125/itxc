"""Fast SAE vs raw comparison using pre-computed activations."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
import json
import pickle
import time

print("="*60)
print("FAST SAE vs RAW COMPARISON")
print("="*60)

# Load SAE only (no model needed if we use saved activations)
print("\nLoading SAE...")
start = time.time()
sae = SAE.from_pretrained("deepseek-r1-distill-llama-8b-qresearch", "blocks.19.hook_resid_post", device='cpu')
print(f"SAE loaded in {time.time()-start:.1f}s: {sae.cfg.d_in}D → {sae.cfg.d_sae}D")

# Load existing activation dataset
print("\nLoading saved activations...")
with open('large_files/activations/activation_dataset_tl.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"Loaded {len(dataset)} activation pairs")

# Test k=1
k = 1
results = {}

# Collect raw and SAE pairs
raw_pairs = []
sae_pairs = []

sae.eval()

print("\nProcessing activation pairs...")
for item in dataset:
    if item.tgt_idx - item.src_idx != k:
        continue

    # Get layer 19 activations
    layer_key = 'blocks.19.hook_resid_mid'
    if layer_key not in item.src_activations:
        continue

    src_raw = torch.tensor(item.src_activations[layer_key]).float()
    tgt_raw = torch.tensor(item.tgt_activations[layer_key]).float()
    raw_pairs.append((src_raw, tgt_raw))

    # Encode to SAE
    with torch.no_grad():
        src_sae = sae.encode(src_raw.unsqueeze(0)).squeeze(0).float()
        tgt_sae = sae.encode(tgt_raw.unsqueeze(0)).squeeze(0).float()
    sae_pairs.append((src_sae, tgt_sae))

    if len(raw_pairs) >= 50:  # Use 50 pairs for quick test
        break

print(f"Collected {len(raw_pairs)} pairs")

# Function to train and evaluate probe
def evaluate_probe(pairs, name, input_dim=None):
    print(f"\n{'='*40}")
    print(f"{name}")
    print("="*40)

    X = torch.stack([p[0] for p in pairs])
    y = torch.stack([p[1] for p in pairs])

    # For SAE, use only active dimensions
    if name == "SAE LATENTS" and X.shape[1] > 10000:
        active_mask = (X.abs() > 1e-6).float().mean(0) > 0.01
        n_active = active_mask.sum().item()
        print(f"Using {n_active}/{X.shape[1]} active dimensions ({n_active/X.shape[1]*100:.1f}%)")
        X = X[:, active_mask]
        y = y[:, active_mask]
        input_dim = n_active

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
    trainer.train(20, verbose=False)

    # Evaluate R²
    train_loss, train_r2 = trainer.evaluate(train_loader)
    test_loss, test_r2 = trainer.evaluate(test_loader)

    # Evaluate cosine similarity of predictions
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

    mean_cos_pred = np.mean(cos_sims) if cos_sims else 0

    # Direct cosine similarity (no probe)
    direct_cos = []
    for i in range(min(10, len(X_test))):
        cos = torch.nn.functional.cosine_similarity(
            X_test[i].unsqueeze(0), y_test[i].unsqueeze(0)
        ).item()
        direct_cos.append(cos)
    mean_cos_direct = np.mean(direct_cos) if direct_cos else 0

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

# Evaluate both
results['raw'] = evaluate_probe(raw_pairs, "RAW RESIDUALS", 4096)
results['sae'] = evaluate_probe(sae_pairs, "SAE LATENTS")

# Save results
with open('large_files/viz/fast_sae_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary with proper comparison
print("\n" + "="*60)
print("SUMMARY: Apples-to-Apples Comparison")
print("="*60)
print(f"{'Metric':<30} {'Raw':<15} {'SAE':<15} {'Winner':<10}")
print("-" * 70)

# Test R²
r2_winner = "SAE" if results['sae']['test_r2'] > results['raw']['test_r2'] else "Raw"
r2_diff = abs(results['sae']['test_r2'] - results['raw']['test_r2'])
print(f"{'Test R² (probe performance)':<30} {results['raw']['test_r2']:<15.4f} {results['sae']['test_r2']:<15.4f} {r2_winner:<10}")

# Prediction cosine
cos_pred_winner = "SAE" if results['sae']['cosine_pred'] > results['raw']['cosine_pred'] else "Raw"
cos_pred_diff = abs(results['sae']['cosine_pred'] - results['raw']['cosine_pred'])
print(f"{'Cosine (probe predictions)':<30} {results['raw']['cosine_pred']:<15.4f} {results['sae']['cosine_pred']:<15.4f} {cos_pred_winner:<10}")

# Direct cosine
cos_direct_winner = "SAE" if results['sae']['cosine_direct'] > results['raw']['cosine_direct'] else "Raw"
print(f"{'Cosine (direct, no probe)':<30} {results['raw']['cosine_direct']:<15.4f} {results['sae']['cosine_direct']:<15.4f} {cos_direct_winner:<10}")

# Feature count
print(f"{'Feature dimensions':<30} {results['raw']['n_features']:<15} {results['sae']['n_features']:<15}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# Overall assessment based on probe performance
if results['sae']['test_r2'] > results['raw']['test_r2'] and results['sae']['cosine_pred'] > results['raw']['cosine_pred']:
    print("✅ SAE latents preserve temporal structure BETTER than raw residuals")
    r2_imp = (results['sae']['test_r2'] - results['raw']['test_r2']) / abs(results['raw']['test_r2']) * 100
    cos_imp = (results['sae']['cosine_pred'] - results['raw']['cosine_pred']) / abs(results['raw']['cosine_pred']) * 100
    print(f"   - R² improvement: {r2_imp:.1f}%")
    print(f"   - Cosine improvement: {cos_imp:.1f}%")
    print("   → SAE features capture temporal dependencies effectively")
elif results['raw']['test_r2'] > results['sae']['test_r2'] and results['raw']['cosine_pred'] > results['sae']['cosine_pred']:
    print("❌ Raw residuals preserve temporal structure BETTER than SAE latents")
    r2_loss = (results['raw']['test_r2'] - results['sae']['test_r2']) / abs(results['raw']['test_r2']) * 100
    cos_loss = (results['raw']['cosine_pred'] - results['sae']['cosine_pred']) / abs(results['raw']['cosine_pred']) * 100
    print(f"   - R² advantage: {r2_loss:.1f}%")
    print(f"   - Cosine advantage: {cos_loss:.1f}%")
    print("   → Sparsification loses temporal information")
else:
    print("🔍 MIXED RESULTS")
    print(f"   - R² favors: {r2_winner}")
    print(f"   - Cosine favors: {cos_pred_winner}")
    print("   → Inconclusive - need more data")

print(f"\nNote: SAE uses only {results['sae']['n_features']}/{sae.cfg.d_sae} dimensions")
print(f"      Sparsity: {results['sae']['n_features']/sae.cfg.d_sae*100:.1f}%")