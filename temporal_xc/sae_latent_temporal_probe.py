"""Compare temporal probes: predicting SAE latents vs raw residuals at t+k from t."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pickle
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from temporal_xc.train_probe import LinearProbe, ProbeTrainer
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def load_models_and_sae():
    """Load the model, tokenizer, and SAE."""
    print("Loading models...")
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

    print("Loading SAE for layer 19...")
    release = "deepseek-r1-distill-llama-8b-qresearch"
    sae_id = "blocks.19.hook_resid_post"
    sae = SAE.from_pretrained(release, sae_id)

    print(f"SAE loaded: {sae.cfg.d_in} -> {sae.cfg.d_sae} dimensions")
    print(f"Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in:.1f}x")

    return model, tokenizer, sae


def harvest_activations_and_encode(
    model,
    tokenizer,
    sae,
    texts: List[str],
    layer: int = 19,
    max_examples: int = 100
) -> Dict:
    """Harvest activations and encode them to SAE features."""

    all_raw_sequences = []  # List of sequences of raw activations
    all_sae_sequences = []  # List of sequences of SAE features
    metadata = []

    model.eval()
    sae.eval()

    for text_idx, text in enumerate(texts[:max_examples]):
        if len(text) < 50:
            continue

        # Tokenize
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)['input_ids']

        # Get raw activations at layer 19
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Layer 19 residual stream (after attention and MLP)
            # This is what we hook with hook_resid_post
            layer_19_residuals = hidden_states[layer][0]  # (seq_len, d_model)

            # Encode EACH position's activation to SAE features
            # This gives us SAE latents at each position
            sae_features_seq = sae.encode(layer_19_residuals)  # (seq_len, d_sae)

        all_raw_sequences.append(layer_19_residuals.cpu())
        all_sae_sequences.append(sae_features_seq.float().cpu())
        metadata.append({
            'text': text,
            'tokens': tokens[0].cpu().numpy(),
            'seq_len': layer_19_residuals.shape[0]
        })

        if (text_idx + 1) % 10 == 0:
            print(f"Processed {text_idx + 1}/{max_examples} examples")

    return {
        'raw_sequences': all_raw_sequences,
        'sae_sequences': all_sae_sequences,
        'metadata': metadata
    }


def create_temporal_prediction_dataset(
    sequences: List[torch.Tensor],
    k: int,
    max_pairs_per_seq: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dataset for predicting activations at t+k from t."""

    src_list = []
    tgt_list = []

    for seq in sequences:
        seq_len = seq.shape[0]

        if seq_len <= k:
            continue

        # Create pairs: (activation_t, activation_{t+k})
        num_pairs = min(max_pairs_per_seq, seq_len - k)

        for i in range(num_pairs):
            src_list.append(seq[i])
            tgt_list.append(seq[i + k])

    if len(src_list) == 0:
        raise ValueError(f"No valid examples for k={k}")

    X = torch.stack(src_list)
    y = torch.stack(tgt_list)

    return X, y


def train_and_evaluate_temporal_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    feature_type: str,
    device: str = "cpu"
) -> Dict:
    """Train a probe to predict features at t+k from features at t."""

    print(f"\nTraining {feature_type} temporal probe for k={k}...")
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Handle sparse SAE features - use only active dimensions
    if feature_type == "SAE" and X.shape[1] > 10000:
        # For SAE features, many dimensions are zero
        # Find dimensions that are active in at least 1% of examples
        active_dims = (X.abs() > 1e-6).float().mean(0) > 0.01
        n_active = active_dims.sum().item()
        print(f"Using {n_active}/{X.shape[1]} active SAE dimensions")

        # Keep only active dimensions
        X = X[:, active_dims]
        y = y[:, active_dims]

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
    probe = LinearProbe(X.shape[1], y.shape[1]).to(device)
    trainer = ProbeTrainer(probe, train_loader, test_loader, device=device, lr=1e-3)
    trainer.train(30, verbose=False)

    # Evaluate
    train_loss, train_r2 = trainer.evaluate(train_loader)
    test_loss, test_r2 = trainer.evaluate(test_loader)

    # Compute cosine similarity
    probe.eval()
    cosine_sims = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = probe(x_batch.to(device))
            for i in range(len(x_batch)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    pred[i].unsqueeze(0),
                    y_batch[i].to(device).unsqueeze(0)
                ).item()
                cosine_sims.append(cos_sim)
            if len(cosine_sims) >= 50:
                break

    mean_cosine = np.mean(cosine_sims)

    print(f"{feature_type} k={k}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Cosine={mean_cosine:.3f}")

    # For SAE, also compute sparsity
    sparsity = None
    if feature_type == "SAE":
        sparsity = (X_test.abs() > 1e-6).float().mean().item()
        print(f"SAE sparsity: {sparsity:.3f} (fraction of non-zero features)")

    return {
        'k': k,
        'feature_type': feature_type,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mean_cosine': mean_cosine,
        'n_features': X.shape[1],
        'sparsity': sparsity
    }


def main():
    """Main comparison of SAE latent vs raw residual temporal prediction."""

    print("="*60)
    print("TEMPORAL STRUCTURE: SAE LATENTS vs RAW RESIDUALS")
    print("="*60)
    print("\nComparing: Predict features at t+k from features at t")
    print("- Raw: residual[t] → residual[t+k]")
    print("- SAE: latent[t] → latent[t+k]")

    # Load models
    model, tokenizer, sae = load_models_and_sae()

    # Prepare test texts (reasoning chains)
    test_texts = [
        "Let me solve this step by step. First, I need to identify the key components. Then I'll analyze their relationships. Finally, I'll compute the solution.",
        "To find the answer, we start by examining the given conditions. Next, we apply the relevant formula. After that, we simplify the expression. The result gives us our answer.",
        "Breaking down the problem: Step 1 involves setting up the equation. Step 2 requires solving for the variable. Step 3 is to verify our solution. Step 4 confirms the final answer.",
        "The solution process begins with understanding the constraints. We then formulate the objective function. Following this, we optimize the parameters. The optimal value is our result.",
        "Analyzing this systematically: Initially, we gather all the data. Subsequently, we process the information. Then we draw conclusions. Finally, we validate our findings.",
        "Consider the mathematical approach: We define our variables first. Then establish the relationships between them. Next, we solve the system of equations. The solution emerges from this analysis.",
        "Working through the logic: The premise leads to an initial hypothesis. Testing this hypothesis reveals new insights. These insights guide us to the conclusion. The conclusion validates our approach.",
        "Step-by-step reasoning shows: The first observation points to a pattern. The pattern suggests a rule. Applying the rule gives predictions. The predictions match our expectations.",
    ] * 15  # Repeat to get more examples

    # Harvest activations and encode to SAE features
    print("\nHarvesting activations and encoding to SAE features...")
    data = harvest_activations_and_encode(model, tokenizer, sae, test_texts, layer=19)

    # Store results
    all_results = []

    # Test different k values
    k_values = [1, 2, 4]

    # Evaluate RAW RESIDUAL temporal prediction
    print("\n" + "="*40)
    print("RAW RESIDUAL TEMPORAL PREDICTION")
    print("="*40)

    for k in k_values:
        try:
            X, y = create_temporal_prediction_dataset(data['raw_sequences'], k)
            result = train_and_evaluate_temporal_probe(X, y, k, "Raw")
            all_results.append(result)
        except Exception as e:
            print(f"Error for raw k={k}: {e}")

    # Evaluate SAE LATENT temporal prediction
    print("\n" + "="*40)
    print("SAE LATENT TEMPORAL PREDICTION")
    print("="*40)

    for k in k_values:
        try:
            X, y = create_temporal_prediction_dataset(data['sae_sequences'], k)
            result = train_and_evaluate_temporal_probe(X, y, k, "SAE")
            all_results.append(result)
        except Exception as e:
            print(f"Error for SAE k={k}: {e}")

    # Create comparison visualization
    create_comparison_plot(all_results)

    # Save results
    with open('large_files/viz/sae_latent_vs_raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n💾 Results saved to large_files/viz/sae_latent_vs_raw_results.json")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: SAE LATENTS vs RAW RESIDUALS")
    print("="*60)
    print(f"{'Type':<10} {'k':<5} {'Test R²':<10} {'Cosine':<10} {'Features':<10}")
    print("-" * 50)

    for result in all_results:
        print(f"{result['feature_type']:<10} {result['k']:<5} {result['test_r2']:<10.3f} "
              f"{result['mean_cosine']:<10.3f} {result['n_features']:<10}")

    # Analyze improvement
    print("\n" + "="*60)
    print("ANALYSIS: Does SAE preserve temporal structure better?")
    print("="*60)

    for k in k_values:
        raw_r2 = next((r['test_r2'] for r in all_results if r['k'] == k and r['feature_type'] == 'Raw'), None)
        sae_r2 = next((r['test_r2'] for r in all_results if r['k'] == k and r['feature_type'] == 'SAE'), None)

        if raw_r2 and sae_r2:
            improvement = (sae_r2 - raw_r2) / abs(raw_r2) * 100 if raw_r2 != 0 else 0
            print(f"k={k}: SAE R²={sae_r2:.3f}, Raw R²={raw_r2:.3f}, "
                  f"Improvement: {improvement:+.1f}%")

    return all_results


def create_comparison_plot(results: List[Dict]):
    """Create visualization comparing SAE latent vs raw residual temporal prediction."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Separate results by type
    raw_results = [r for r in results if r['feature_type'] == 'Raw']
    sae_results = [r for r in results if r['feature_type'] == 'SAE']

    if not raw_results or not sae_results:
        print("Not enough results for comparison plot")
        return

    k_values = sorted(set(r['k'] for r in raw_results))

    # Plot 1: Test R² Comparison
    ax1 = axes[0, 0]
    raw_r2 = [next(r['test_r2'] for r in raw_results if r['k'] == k) for k in k_values]
    sae_r2 = [next(r['test_r2'] for r in sae_results if r['k'] == k) for k in k_values]

    ax1.plot(k_values, raw_r2, 'o-', label='Raw Residuals', linewidth=2, markersize=8, color='blue')
    ax1.plot(k_values, sae_r2, 's-', label='SAE Latents', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('k (tokens ahead)')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Temporal Prediction: Test R²')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(min(raw_r2), min(sae_r2)) - 0.1, 1.0])

    # Plot 2: Cosine Similarity Comparison
    ax2 = axes[0, 1]
    raw_cos = [next(r['mean_cosine'] for r in raw_results if r['k'] == k) for k in k_values]
    sae_cos = [next(r['mean_cosine'] for r in sae_results if r['k'] == k) for k in k_values]

    ax2.plot(k_values, raw_cos, 'o-', label='Raw Residuals', linewidth=2, markersize=8, color='blue')
    ax2.plot(k_values, sae_cos, 's-', label='SAE Latents', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('k (tokens ahead)')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Prediction Quality: Cosine Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Relative Improvement
    ax3 = axes[0, 2]
    r2_improvement = [(s - r) / abs(r) * 100 if r != 0 else 0
                      for s, r in zip(sae_r2, raw_r2)]
    cos_improvement = [(s - r) / abs(r) * 100 if r != 0 else 0
                       for s, r in zip(sae_cos, raw_cos)]

    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax3.bar(x - width/2, r2_improvement, width, label='R² Improvement', alpha=0.8, color='green')
    bars2 = ax3.bar(x + width/2, cos_improvement, width, label='Cosine Improvement', alpha=0.8, color='orange')

    ax3.set_xlabel('k (tokens ahead)')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('SAE Improvement over Raw Residuals')
    ax3.set_xticks(x)
    ax3.set_xticklabels(k_values)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom' if height > 0 else 'top')

    # Plot 4: Feature Dimensionality
    ax4 = axes[1, 0]
    raw_dims = [r['n_features'] for r in raw_results[:1]]  # Same for all k
    sae_dims = [r['n_features'] for r in sae_results[:1]]

    bars = ax4.bar(['Raw', 'SAE'], [raw_dims[0], sae_dims[0]], color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Number of Features')
    ax4.set_title('Feature Dimensionality')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, [raw_dims[0], sae_dims[0]]):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom')

    # Plot 5: Performance Degradation
    ax5 = axes[1, 1]
    # Normalize to k=1 performance
    raw_norm = [r/raw_r2[0] if raw_r2[0] != 0 else 0 for r in raw_r2]
    sae_norm = [s/sae_r2[0] if sae_r2[0] != 0 else 0 for s in sae_r2]

    ax5.plot(k_values, raw_norm, 'o-', label='Raw (normalized)', linewidth=2, markersize=8, color='blue')
    ax5.plot(k_values, sae_norm, 's-', label='SAE (normalized)', linewidth=2, markersize=8, color='red')
    ax5.set_xlabel('k (tokens ahead)')
    ax5.set_ylabel('Normalized R² (k=1 as baseline)')
    ax5.set_title('Relative Performance Degradation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.1])

    # Plot 6: Summary Text
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = "KEY FINDINGS:\n\n"
    if len(r2_improvement) > 0:
        avg_improvement = np.mean(r2_improvement)
        if avg_improvement > 0:
            summary_text += f"✓ SAE latents show {avg_improvement:.0f}% better\n  temporal structure preservation\n\n"
        else:
            summary_text += f"✗ Raw residuals show {-avg_improvement:.0f}% better\n  temporal structure preservation\n\n"

    summary_text += "INTERPRETATION:\n"
    if avg_improvement > 0:
        summary_text += "• SAE features capture temporal\n  dependencies better\n"
        summary_text += "• Sparse features preserve\n  temporal information\n"
        summary_text += "• Dimensionality reduction helps\n  temporal prediction"
    else:
        summary_text += "• Raw residuals preserve more\n  temporal information\n"
        summary_text += "• SAE compression may lose\n  temporal structure\n"
        summary_text += "• Dense representations better\n  for temporal prediction"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Temporal Structure Preservation: SAE Latents vs Raw Residuals\n'
                 'Predicting Features at t+k from Features at t',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    plt.savefig('large_files/viz/sae_latent_vs_raw_temporal.png', dpi=150, bbox_inches='tight')
    print("📊 Saved comparison plot to large_files/viz/sae_latent_vs_raw_temporal.png")


if __name__ == "__main__":
    main()