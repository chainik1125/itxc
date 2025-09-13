"""Proper comparison: Train probes on both raw and SAE, measure R² and cosine sim of predictions."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
import json
import matplotlib.pyplot as plt


def collect_activation_pairs(model, tokenizer, sae, texts, k=1, max_pairs=200):
    """Collect pairs of (activation[t], activation[t+k]) for both raw and SAE."""

    raw_pairs = []
    sae_pairs = []

    model.eval()
    sae.eval()

    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)['input_ids']

        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
            # Layer 19 residuals (what hook_resid_post would see)
            layer_19 = outputs.hidden_states[19][0]  # (seq_len, 4096)

            # Collect pairs for positions t and t+k
            for i in range(layer_19.shape[0] - k):
                # Raw residuals
                raw_src = layer_19[i]
                raw_tgt = layer_19[i + k]
                raw_pairs.append((raw_src, raw_tgt))

                # SAE encodings
                sae_src = sae.encode(raw_src.unsqueeze(0)).squeeze(0).float()
                sae_tgt = sae.encode(raw_tgt.unsqueeze(0)).squeeze(0).float()
                sae_pairs.append((sae_src, sae_tgt))

                if len(raw_pairs) >= max_pairs:
                    return raw_pairs, sae_pairs

    return raw_pairs, sae_pairs


def train_and_evaluate_probe(pairs, feature_type="Raw", k=1):
    """Train probe and evaluate both R² and cosine similarity of predictions."""

    print(f"\n{'='*40}")
    print(f"Training {feature_type} probe for k={k}")
    print("="*40)

    # Stack pairs into tensors
    X = torch.stack([p[0] for p in pairs])
    y = torch.stack([p[1] for p in pairs])

    # For SAE, use only active dimensions
    if feature_type == "SAE" and X.shape[1] > 10000:
        active_mask = (X.abs() > 1e-6).float().mean(0) > 0.01
        n_active = active_mask.sum().item()
        print(f"Using {n_active}/{X.shape[1]} active SAE dimensions")
        X = X[:, active_mask]
        y = y[:, active_mask]

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Split data
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train probe
    probe = LinearProbe(X.shape[1], y.shape[1])
    trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)
    trainer.train(30, verbose=False)

    # Evaluate R²
    train_loss, train_r2 = trainer.evaluate(train_loader)
    test_loss, test_r2 = trainer.evaluate(test_loader)

    # Evaluate cosine similarity of predictions
    probe.eval()
    cosine_sims = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = probe(x_batch)
            for i in range(len(x_batch)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    pred[i].unsqueeze(0),
                    y_batch[i].unsqueeze(0)
                ).item()
                cosine_sims.append(cos_sim)

    mean_cosine = np.mean(cosine_sims)
    std_cosine = np.std(cosine_sims)

    print(f"Results:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Mean cosine (predictions): {mean_cosine:.4f} ± {std_cosine:.4f}")

    # Also compute direct cosine similarity (no probe)
    direct_cosines = []
    for x, y_true in zip(X_test[:50], y_test[:50]):
        direct_cos = torch.nn.functional.cosine_similarity(
            x.unsqueeze(0), y_true.unsqueeze(0)
        ).item()
        direct_cosines.append(direct_cos)

    mean_direct = np.mean(direct_cosines)
    print(f"  Direct cosine (no probe): {mean_direct:.4f}")

    return {
        'feature_type': feature_type,
        'k': k,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mean_cosine_pred': mean_cosine,
        'std_cosine_pred': std_cosine,
        'mean_cosine_direct': mean_direct,
        'n_features': X.shape[1]
    }


def main():
    print("="*60)
    print("PROPER SAE vs RAW COMPARISON")
    print("="*60)
    print("Training probes to predict activations at t+k from t")
    print("Measuring both R² and cosine similarity of predictions")

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading SAE...")
    sae = SAE.from_pretrained("deepseek-r1-distill-llama-8b-qresearch", "blocks.19.hook_resid_post", device='cpu')
    print(f"SAE: {sae.cfg.d_in}D → {sae.cfg.d_sae}D")

    # Test texts
    test_texts = [
        "Let me solve this step by step. First, I need to identify the key components. Then I'll analyze their relationships. Finally, I'll compute the solution.",
        "To find the answer, we start by examining the given conditions. Next, we apply the relevant formula. After that, we simplify the expression.",
        "Breaking down the problem: Step 1 involves setting up the equation. Step 2 requires solving for the variable. Step 3 is to verify our solution.",
        "The solution process begins with understanding the constraints. We then formulate the objective function. Following this, we optimize the parameters.",
        "Analyzing this systematically: Initially, we gather all the data. Subsequently, we process the information. Then we draw conclusions.",
    ] * 10  # Repeat to get more examples

    all_results = []

    # Test different k values
    for k in [1, 2, 4]:
        print(f"\n{'='*60}")
        print(f"Testing k={k}")
        print("="*60)

        # Collect activation pairs
        print("Collecting activation pairs...")
        raw_pairs, sae_pairs = collect_activation_pairs(model, tokenizer, sae, test_texts, k=k, max_pairs=200)
        print(f"Collected {len(raw_pairs)} pairs")

        # Train and evaluate raw residual probe
        raw_result = train_and_evaluate_probe(raw_pairs, "Raw", k)
        all_results.append(raw_result)

        # Train and evaluate SAE latent probe
        sae_result = train_and_evaluate_probe(sae_pairs, "SAE", k)
        all_results.append(sae_result)

    # Save results
    with open('large_files/viz/proper_sae_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create comparison plot
    create_comparison_plot(all_results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Probe Performance Comparison")
    print("="*60)
    print(f"{'Type':<5} {'k':<3} {'Test R²':<10} {'Cos(pred)':<12} {'Direct Cos':<12}")
    print("-" * 50)

    for r in all_results:
        print(f"{r['feature_type']:<5} {r['k']:<3} {r['test_r2']:<10.4f} "
              f"{r['mean_cosine_pred']:<12.4f} {r['mean_cosine_direct']:<12.4f}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    for k in [1, 2, 4]:
        raw = next(r for r in all_results if r['k'] == k and r['feature_type'] == 'Raw')
        sae = next(r for r in all_results if r['k'] == k and r['feature_type'] == 'SAE')

        r2_diff = (sae['test_r2'] - raw['test_r2']) / abs(raw['test_r2']) * 100
        cos_diff = (sae['mean_cosine_pred'] - raw['mean_cosine_pred']) / abs(raw['mean_cosine_pred']) * 100

        print(f"\nk={k}:")
        print(f"  R² comparison: SAE {sae['test_r2']:.4f} vs Raw {raw['test_r2']:.4f}")
        print(f"    → {'SAE' if r2_diff > 0 else 'Raw'} is {abs(r2_diff):.1f}% better")
        print(f"  Cosine comparison: SAE {sae['mean_cosine_pred']:.4f} vs Raw {raw['mean_cosine_pred']:.4f}")
        print(f"    → {'SAE' if cos_diff > 0 else 'Raw'} is {abs(cos_diff):.1f}% better")

    return all_results


def create_comparison_plot(results):
    """Create visualization comparing SAE vs raw probe performance."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Separate results
    raw_results = [r for r in results if r['feature_type'] == 'Raw']
    sae_results = [r for r in results if r['feature_type'] == 'SAE']

    k_values = sorted(set(r['k'] for r in raw_results))

    # Plot 1: Test R² Comparison
    ax1 = axes[0, 0]
    raw_r2 = [r['test_r2'] for r in raw_results]
    sae_r2 = [r['test_r2'] for r in sae_results]

    x = np.arange(len(k_values))
    width = 0.35

    ax1.bar(x - width/2, raw_r2, width, label='Raw Residuals', color='blue', alpha=0.7)
    ax1.bar(x + width/2, sae_r2, width, label='SAE Latents', color='red', alpha=0.7)
    ax1.set_xlabel('k (tokens ahead)')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Probe Test R² Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_values)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (r, s) in enumerate(zip(raw_r2, sae_r2)):
        ax1.text(i - width/2, r + 0.01, f'{r:.3f}', ha='center', fontsize=9)
        ax1.text(i + width/2, s + 0.01, f'{s:.3f}', ha='center', fontsize=9)

    # Plot 2: Cosine Similarity of Predictions
    ax2 = axes[0, 1]
    raw_cos = [r['mean_cosine_pred'] for r in raw_results]
    sae_cos = [r['mean_cosine_pred'] for r in sae_results]

    ax2.bar(x - width/2, raw_cos, width, label='Raw Residuals', color='blue', alpha=0.7)
    ax2.bar(x + width/2, sae_cos, width, label='SAE Latents', color='red', alpha=0.7)
    ax2.set_xlabel('k (tokens ahead)')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Prediction Cosine Similarity', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_values)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Direct vs Predicted Cosine
    ax3 = axes[0, 2]
    for i, k in enumerate(k_values):
        raw_r = raw_results[i]
        sae_r = sae_results[i]

        categories = ['Raw\nDirect', 'Raw\nPredicted', 'SAE\nDirect', 'SAE\nPredicted']
        values = [raw_r['mean_cosine_direct'], raw_r['mean_cosine_pred'],
                  sae_r['mean_cosine_direct'], sae_r['mean_cosine_pred']]
        colors = ['lightblue', 'blue', 'lightcoral', 'red']

        x_pos = np.arange(len(categories)) + i * (len(categories) + 1)
        bars = ax3.bar(x_pos, values, color=colors, alpha=0.7)

        if i == 0:
            ax3.text(x_pos[1], -0.1, f'k={k}', ha='center', fontsize=10)

    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Direct vs Predicted Cosine', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks([])

    # Plot 4: Relative Performance
    ax4 = axes[1, 0]
    r2_improvement = [(s['test_r2'] - r['test_r2']) / abs(r['test_r2']) * 100
                      for r, s in zip(raw_results, sae_results)]
    cos_improvement = [(s['mean_cosine_pred'] - r['mean_cosine_pred']) / abs(r['mean_cosine_pred']) * 100
                       for r, s in zip(raw_results, sae_results)]

    x = np.arange(len(k_values))
    ax4.bar(x - width/2, r2_improvement, width, label='R² Change', color='green' if np.mean(r2_improvement) > 0 else 'orange')
    ax4.bar(x + width/2, cos_improvement, width, label='Cosine Change', color='purple')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('k (tokens ahead)')
    ax4.set_ylabel('SAE vs Raw Improvement (%)')
    ax4.set_title('SAE Performance Relative to Raw', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_values)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Feature Dimensions
    ax5 = axes[1, 1]
    raw_dims = [r['n_features'] for r in raw_results[:1]]
    sae_dims = [r['n_features'] for r in sae_results[:1]]

    bars = ax5.bar(['Raw', 'SAE\n(active)'], [raw_dims[0], sae_dims[0]], color=['blue', 'red'], alpha=0.7)
    ax5.set_ylabel('Number of Features')
    ax5.set_title('Feature Dimensionality')
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, [raw_dims[0], sae_dims[0]]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val}', ha='center', fontsize=11)

    # Plot 6: Summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    avg_r2_imp = np.mean(r2_improvement)
    avg_cos_imp = np.mean(cos_improvement)

    summary = f"""SUMMARY

Average Performance:
• R² : {'SAE' if avg_r2_imp > 0 else 'Raw'} {abs(avg_r2_imp):.1f}% better
• Cosine: {'SAE' if avg_cos_imp > 0 else 'Raw'} {abs(avg_cos_imp):.1f}% better

Conclusion:
{'SAE latents preserve temporal structure better' if avg_r2_imp > 0 and avg_cos_imp > 0 else 'Raw residuals preserve temporal structure better'}
"""

    ax6.text(0.1, 0.7, summary, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('SAE vs Raw: Proper Probe Comparison\n'
                 'Training probes to predict activations at t+k from t',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig('large_files/viz/proper_sae_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved plot to large_files/viz/proper_sae_comparison.png")


if __name__ == "__main__":
    main()