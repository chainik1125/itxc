"""Compare SAE latent vs raw residual temporal prediction - optimized version."""

import torch
import numpy as np
from sae_lens import SAE
import json

print("="*60)
print("SAE LATENT vs RAW RESIDUAL TEMPORAL COMPARISON")
print("="*60)

# Load SAE
print("\nLoading SAE...")
release = "deepseek-r1-distill-llama-8b-qresearch"
sae_id = "blocks.19.hook_resid_post"
sae = SAE.from_pretrained(release, sae_id, device='cpu')
print(f"SAE loaded: {sae.cfg.d_in}D → {sae.cfg.d_sae}D")

# Use existing dataset
print("\nLoading existing activation dataset...")
import pickle
with open('large_files/activations/activation_dataset_tl.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"Loaded {len(dataset)} activation pairs")

# Process first 5 problems
results = {'raw': {}, 'sae': {}}

for k in [1, 2]:
    print(f"\n{'='*40}")
    print(f"Testing k={k}")
    print("="*40)

    raw_pairs = []
    sae_pairs = []

    # Collect activation pairs
    for item in dataset[:5]:
        if hasattr(item, 'src_activations') and hasattr(item, 'tgt_activations'):
            # Raw residuals
            src_raw = torch.tensor(item.src_activations[19])  # Layer 19
            tgt_raw = torch.tensor(item.tgt_activations[19])

            # Check if this is k tokens apart
            if item.tgt_idx - item.src_idx == k:
                raw_pairs.append((src_raw, tgt_raw))

                # Encode to SAE features
                with torch.no_grad():
                    src_sae = sae.encode(src_raw.unsqueeze(0)).squeeze(0)
                    tgt_sae = sae.encode(tgt_raw.unsqueeze(0)).squeeze(0)
                sae_pairs.append((src_sae.float(), tgt_sae.float()))

    print(f"Collected {len(raw_pairs)} pairs")

    if len(raw_pairs) >= 10:
        # Evaluate raw residuals
        X_raw = torch.stack([p[0] for p in raw_pairs])
        y_raw = torch.stack([p[1] for p in raw_pairs])

        # Simple correlation test
        cos_sims = []
        for i in range(len(X_raw)):
            cos_sim = torch.nn.functional.cosine_similarity(
                X_raw[i].unsqueeze(0), y_raw[i].unsqueeze(0)
            ).item()
            cos_sims.append(cos_sim)

        raw_cos = np.mean(cos_sims)
        print(f"Raw residuals: Mean cosine similarity = {raw_cos:.3f}")
        results['raw'][k] = raw_cos

        # Evaluate SAE features
        X_sae = torch.stack([p[0] for p in sae_pairs])
        y_sae = torch.stack([p[1] for p in sae_pairs])

        # Find active dimensions
        active_mask = (X_sae.abs() > 1e-6).any(0)
        n_active = active_mask.sum().item()
        print(f"Active SAE dimensions: {n_active}/{X_sae.shape[1]}")

        # Cosine similarity on active dimensions
        X_sae_active = X_sae[:, active_mask]
        y_sae_active = y_sae[:, active_mask]

        cos_sims = []
        for i in range(len(X_sae_active)):
            cos_sim = torch.nn.functional.cosine_similarity(
                X_sae_active[i].unsqueeze(0), y_sae_active[i].unsqueeze(0)
            ).item()
            cos_sims.append(cos_sim)

        sae_cos = np.mean(cos_sims)
        print(f"SAE latents: Mean cosine similarity = {sae_cos:.3f}")
        results['sae'][k] = sae_cos

# Summary
print("\n" + "="*60)
print("SUMMARY: Temporal Structure Preservation")
print("="*60)

for k in [1, 2]:
    if k in results['raw'] and k in results['sae']:
        raw = results['raw'][k]
        sae_val = results['sae'][k]
        better = "SAE" if sae_val > raw else "Raw"
        diff = abs(sae_val - raw)
        print(f"k={k}: Raw={raw:.3f}, SAE={sae_val:.3f} → {better} preserves {diff:.3f} more similarity")

# Save results
with open('large_files/viz/sae_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n💾 Results saved to large_files/viz/sae_comparison_results.json")

print("\nINTERPRETATION:")
if all(results['sae'].get(k, 0) > results['raw'].get(k, 0) for k in [1, 2]):
    print("✓ SAE latents preserve temporal structure BETTER than raw residuals")
    print("  This suggests SAE features capture meaningful temporal dependencies")
else:
    print("✗ Raw residuals preserve temporal structure better than SAE latents")
    print("  This suggests some temporal information is lost in SAE encoding")