"""Quick proper comparison of SAE vs raw with probes."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
import json

print("="*60)
print("QUICK PROPER COMPARISON: SAE vs RAW")
print("="*60)

# Load models
print("\nLoading models...")
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading SAE...")
sae = SAE.from_pretrained("deepseek-r1-distill-llama-8b-qresearch", "blocks.19.hook_resid_post", device='cpu')
print(f"SAE: {sae.cfg.d_in}D → {sae.cfg.d_sae}D")

# Simple test texts
texts = [
    "Let me solve this step by step. First, identify the problem. Second, analyze it. Third, find the solution.",
    "To answer this question, we need to examine the data. Then apply the formula. Finally compute the result."
] * 3

# Test k=1 only for speed
k = 1
print(f"\nTesting k={k}")

# Collect pairs
raw_pairs = []
sae_pairs = []

model.eval()
sae.eval()

for text in texts:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)['input_ids']

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        layer_19 = outputs.hidden_states[19][0]  # (seq_len, 4096)

        for i in range(min(10, layer_19.shape[0] - k)):
            # Raw
            raw_src = layer_19[i]
            raw_tgt = layer_19[i + k]
            raw_pairs.append((raw_src, raw_tgt))

            # SAE
            sae_src = sae.encode(raw_src.unsqueeze(0)).squeeze(0).float()
            sae_tgt = sae.encode(raw_tgt.unsqueeze(0)).squeeze(0).float()
            sae_pairs.append((sae_src, sae_tgt))

print(f"Collected {len(raw_pairs)} pairs")

results = {}

# Train and evaluate RAW probe
print("\n" + "="*40)
print("RAW RESIDUALS")
print("="*40)

X_raw = torch.stack([p[0] for p in raw_pairs])
y_raw = torch.stack([p[1] for p in raw_pairs])

# Split
n_train = int(0.8 * len(X_raw))
X_train = X_raw[:n_train]
y_train = y_raw[:n_train]
X_test = X_raw[n_train:]
y_test = y_raw[n_train:]

# Create simple dataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Train probe
probe_raw = LinearProbe(4096, 4096)
trainer = ProbeTrainer(probe_raw, train_loader, test_loader, device="cpu", lr=1e-3)
trainer.train(20, verbose=False)

# Evaluate
_, test_r2 = trainer.evaluate(test_loader)

# Get prediction cosine similarities
probe_raw.eval()
cos_sims = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = probe_raw(x_batch)
        for i in range(len(x_batch)):
            cos = torch.nn.functional.cosine_similarity(
                pred[i].unsqueeze(0), y_batch[i].unsqueeze(0)
            ).item()
            cos_sims.append(cos)

mean_cos_pred = np.mean(cos_sims)

# Direct cosine (no probe)
direct_cos = []
for i in range(min(20, len(X_test))):
    cos = torch.nn.functional.cosine_similarity(
        X_test[i].unsqueeze(0), y_test[i].unsqueeze(0)
    ).item()
    direct_cos.append(cos)
mean_cos_direct = np.mean(direct_cos)

print(f"Test R²: {test_r2:.4f}")
print(f"Cosine (probe predictions): {mean_cos_pred:.4f}")
print(f"Cosine (direct, no probe): {mean_cos_direct:.4f}")

results['raw'] = {
    'test_r2': test_r2,
    'cosine_pred': mean_cos_pred,
    'cosine_direct': mean_cos_direct
}

# Train and evaluate SAE probe
print("\n" + "="*40)
print("SAE LATENTS")
print("="*40)

X_sae = torch.stack([p[0] for p in sae_pairs])
y_sae = torch.stack([p[1] for p in sae_pairs])

# Use only active dims
active_mask = (X_sae.abs() > 1e-6).float().mean(0) > 0.01
n_active = active_mask.sum().item()
print(f"Using {n_active}/{X_sae.shape[1]} active dimensions")

X_sae = X_sae[:, active_mask]
y_sae = y_sae[:, active_mask]

# Split
X_train = X_sae[:n_train]
y_train = y_sae[:n_train]
X_test = X_sae[n_train:]
y_test = y_sae[n_train:]

# Dataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Train probe
probe_sae = LinearProbe(n_active, n_active)
trainer = ProbeTrainer(probe_sae, train_loader, test_loader, device="cpu", lr=1e-3)
trainer.train(20, verbose=False)

# Evaluate
_, test_r2 = trainer.evaluate(test_loader)

# Prediction cosine
probe_sae.eval()
cos_sims = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = probe_sae(x_batch)
        for i in range(len(x_batch)):
            cos = torch.nn.functional.cosine_similarity(
                pred[i].unsqueeze(0), y_batch[i].unsqueeze(0)
            ).item()
            cos_sims.append(cos)

mean_cos_pred = np.mean(cos_sims)

# Direct cosine
direct_cos = []
for i in range(min(20, len(X_test))):
    cos = torch.nn.functional.cosine_similarity(
        X_test[i].unsqueeze(0), y_test[i].unsqueeze(0)
    ).item()
    direct_cos.append(cos)
mean_cos_direct = np.mean(direct_cos)

print(f"Test R²: {test_r2:.4f}")
print(f"Cosine (probe predictions): {mean_cos_pred:.4f}")
print(f"Cosine (direct, no probe): {mean_cos_direct:.4f}")

results['sae'] = {
    'test_r2': test_r2,
    'cosine_pred': mean_cos_pred,
    'cosine_direct': mean_cos_direct,
    'n_active': n_active
}

# Save results
with open('large_files/viz/quick_proper_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print("\n" + "="*60)
print("SUMMARY (k=1)")
print("="*60)
print(f"{'Metric':<25} {'Raw':<15} {'SAE':<15}")
print("-" * 55)
print(f"{'Test R²':<25} {results['raw']['test_r2']:<15.4f} {results['sae']['test_r2']:<15.4f}")
print(f"{'Cosine (predictions)':<25} {results['raw']['cosine_pred']:<15.4f} {results['sae']['cosine_pred']:<15.4f}")
print(f"{'Cosine (direct)':<25} {results['raw']['cosine_direct']:<15.4f} {results['sae']['cosine_direct']:<15.4f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

r2_diff = (results['sae']['test_r2'] - results['raw']['test_r2']) / abs(results['raw']['test_r2']) * 100
cos_diff = (results['sae']['cosine_pred'] - results['raw']['cosine_pred']) / abs(results['raw']['cosine_pred']) * 100

if r2_diff > 0 and cos_diff > 0:
    print(f"✅ SAE latents preserve temporal structure BETTER")
    print(f"   R² improvement: {r2_diff:.1f}%")
    print(f"   Cosine improvement: {cos_diff:.1f}%")
else:
    print(f"❌ Raw residuals preserve temporal structure BETTER")
    print(f"   R² difference: {r2_diff:.1f}%")
    print(f"   Cosine difference: {cos_diff:.1f}%")

print(f"\nNote: SAE using only {results['sae']['n_active']}/{sae.cfg.d_sae} dimensions (sparsity)")