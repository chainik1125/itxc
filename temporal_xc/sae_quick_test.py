"""Quick test of SAE latent temporal structure."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
import json


print("="*60)
print("QUICK TEST: SAE LATENT TEMPORAL STRUCTURE")
print("="*60)

# Load models
print("\nLoading model...")
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading SAE...")
release = "deepseek-r1-distill-llama-8b-qresearch"
sae_id = "blocks.19.hook_resid_post"
sae = SAE.from_pretrained(release, sae_id)
print(f"SAE: {sae.cfg.d_in}D → {sae.cfg.d_sae}D (expansion {sae.cfg.d_sae/sae.cfg.d_in:.1f}x)")

# Test on a few examples
test_texts = [
    "Let me solve this step by step. First, identify the problem. Second, analyze the data. Third, compute the result.",
    "To find the answer: Start with the given information. Apply the relevant formula. Simplify to get the solution.",
    "Breaking it down: Step one establishes the baseline. Step two makes adjustments. Step three validates the outcome."
] * 5  # 15 examples total

print(f"\nProcessing {len(test_texts)} examples...")

all_raw = []
all_sae = []

model.eval()
sae.eval()

for i, text in enumerate(test_texts):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)['input_ids']

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        layer_19 = outputs.hidden_states[19][0]  # (seq_len, 4096)
        sae_features = sae.encode(layer_19)  # (seq_len, 65536)

    all_raw.append(layer_19.cpu())
    all_sae.append(sae_features.float().cpu())

    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{len(test_texts)}")

# Test k=1 and k=2
results = []

for k in [1, 2]:
    print(f"\n{'='*40}")
    print(f"Testing k={k}")
    print("="*40)

    # Raw residuals
    X_raw_list = []
    y_raw_list = []
    for seq in all_raw:
        if seq.shape[0] > k:
            for i in range(min(10, seq.shape[0] - k)):
                X_raw_list.append(seq[i])
                y_raw_list.append(seq[i + k])

    if len(X_raw_list) > 20:
        X_raw = torch.stack(X_raw_list)
        y_raw = torch.stack(y_raw_list)

        # Train probe
        n_train = int(0.8 * len(X_raw))
        train_dataset = TensorDataset(X_raw[:n_train], y_raw[:n_train])
        test_dataset = TensorDataset(X_raw[n_train:], y_raw[n_train:])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        probe = LinearProbe(4096, 4096)
        trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)
        trainer.train(20, verbose=False)

        _, test_r2 = trainer.evaluate(test_loader)
        print(f"Raw Residuals: Test R² = {test_r2:.3f}")
        results.append({'k': k, 'type': 'raw', 'test_r2': test_r2})

    # SAE features (use subset of active dimensions)
    X_sae_list = []
    y_sae_list = []
    for seq in all_sae:
        if seq.shape[0] > k:
            for i in range(min(10, seq.shape[0] - k)):
                X_sae_list.append(seq[i])
                y_sae_list.append(seq[i + k])

    if len(X_sae_list) > 20:
        X_sae = torch.stack(X_sae_list)
        y_sae = torch.stack(y_sae_list)

        # Find active dimensions (non-zero in >1% of examples)
        active_mask = (X_sae.abs() > 1e-6).float().mean(0) > 0.01
        n_active = active_mask.sum().item()
        print(f"Using {n_active}/{X_sae.shape[1]} active SAE dimensions")

        X_sae = X_sae[:, active_mask]
        y_sae = y_sae[:, active_mask]

        # Train probe
        n_train = int(0.8 * len(X_sae))
        train_dataset = TensorDataset(X_sae[:n_train], y_sae[:n_train])
        test_dataset = TensorDataset(X_sae[n_train:], y_sae[n_train:])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        probe = LinearProbe(X_sae.shape[1], y_sae.shape[1])
        trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)
        trainer.train(20, verbose=False)

        _, test_r2 = trainer.evaluate(test_loader)
        print(f"SAE Latents: Test R² = {test_r2:.3f}")
        results.append({'k': k, 'type': 'sae', 'test_r2': test_r2, 'n_active': n_active})

# Save results
with open('large_files/viz/sae_quick_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for k in [1, 2]:
    raw_r2 = next((r['test_r2'] for r in results if r['k'] == k and r['type'] == 'raw'), None)
    sae_r2 = next((r['test_r2'] for r in results if r['k'] == k and r['type'] == 'sae'), None)

    if raw_r2 is not None and sae_r2 is not None:
        improvement = (sae_r2 - raw_r2) / abs(raw_r2) * 100 if raw_r2 != 0 else 0
        better = "SAE" if improvement > 0 else "Raw"
        print(f"k={k}: Raw R²={raw_r2:.3f}, SAE R²={sae_r2:.3f} → {better} is {abs(improvement):.0f}% better")

print("\n💾 Results saved to large_files/viz/sae_quick_test_results.json")