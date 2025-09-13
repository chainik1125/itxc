"""Compare temporal probes on SAE features vs raw residuals."""

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


def load_sae_model():
    """Load the layer 19 SAE for DeepSeek model."""
    print("Loading SAE for layer 19...")

    release = "deepseek-r1-distill-llama-8b-qresearch"
    sae_id = "blocks.19.hook_resid_post"
    sae = SAE.from_pretrained(release, sae_id)

    print(f"SAE loaded: {sae.cfg.d_in} -> {sae.cfg.d_sae} dimensions")
    print(f"Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in:.1f}x")

    return sae


def harvest_sae_activations(
    model,
    tokenizer,
    sae,
    texts: List[str],
    layer: int = 19,
    max_examples: int = 100
) -> Dict:
    """Harvest both raw and SAE activations for comparison."""

    raw_activations = []
    sae_features = []
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
            layer_19_residuals = hidden_states[layer][0].cpu()  # (seq_len, d_model)

            # Pass through SAE to get features
            # SAE.encode returns the feature activations directly
            sae_feats = sae.encode(layer_19_residuals)

        raw_activations.append(layer_19_residuals.numpy())
        sae_features.append(sae_feats.float().cpu().numpy())
        metadata.append({
            'text': text,
            'tokens': tokens[0].cpu().numpy(),
            'seq_len': layer_19_residuals.shape[0]
        })

        if (text_idx + 1) % 10 == 0:
            print(f"Processed {text_idx + 1}/{max_examples} examples")

    return {
        'raw_activations': raw_activations,
        'sae_features': sae_features,
        'metadata': metadata
    }


def create_temporal_dataset(activations_dict: Dict, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dataset for k-token-ahead prediction."""

    src_acts = []
    tgt_acts = []

    for i, meta in enumerate(activations_dict['metadata']):
        seq_len = meta['seq_len']

        if seq_len <= k:
            continue

        # Get activations (either raw or SAE)
        acts = activations_dict.get('current_activations')[i]

        # Create pairs
        for j in range(min(20, seq_len - k)):
            src_acts.append(acts[j])
            tgt_acts.append(acts[j + k])

    if len(src_acts) == 0:
        raise ValueError(f"No valid examples for k={k}")

    X = torch.tensor(np.array(src_acts), dtype=torch.float32)
    y = torch.tensor(np.array(tgt_acts), dtype=torch.float32)

    return X, y


def train_and_evaluate_probe(X: torch.Tensor, y: torch.Tensor, k: int, feature_type: str) -> Dict:
    """Train probe and evaluate performance."""

    print(f"\nTraining {feature_type} probe for k={k}...")
    print(f"Data shape: {X.shape}")

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

    # Evaluate
    train_loss, train_r2 = trainer.evaluate(train_loader)
    test_loss, test_r2 = trainer.evaluate(test_loader)

    # Compute cosine similarity
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
            if len(cosine_sims) >= 50:
                break

    mean_cosine = np.mean(cosine_sims)

    print(f"{feature_type} k={k}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Cosine={mean_cosine:.3f}")

    return {
        'k': k,
        'feature_type': feature_type,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mean_cosine': mean_cosine,
        'n_features': X.shape[1]
    }


def main():
    """Main comparison of SAE vs raw residuals."""

    print("="*60)
    print("SAE vs RAW RESIDUALS: TEMPORAL PROBE COMPARISON")
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SAE
    sae = load_sae_model()

    # Prepare test texts (reasoning chains)
    test_texts = [
        "Let me solve this step by step. First, I need to identify the key components. Then I'll analyze their relationships. Finally, I'll compute the solution.",
        "To find the answer, we start by examining the given conditions. Next, we apply the relevant formula. After that, we simplify the expression. The result gives us our answer.",
        "Breaking down the problem: Step 1 involves setting up the equation. Step 2 requires solving for the variable. Step 3 is to verify our solution. Step 4 confirms the final answer.",
        "The solution process begins with understanding the constraints. We then formulate the objective function. Following this, we optimize the parameters. The optimal value is our result.",
        "Analyzing this systematically: Initially, we gather all the data. Subsequently, we process the information. Then we draw conclusions. Finally, we validate our findings.",
    ] * 20  # Repeat to get more examples

    # Harvest activations
    print("\nHarvesting activations...")
    activations_dict = harvest_sae_activations(model, tokenizer, sae, test_texts, layer=19)

    # Store results
    all_results = []

    # Test different k values
    k_values = [1, 2, 4]

    # Evaluate raw residuals
    print("\n" + "="*40)
    print("EVALUATING RAW RESIDUALS")
    print("="*40)

    activations_dict['current_activations'] = activations_dict['raw_activations']

    for k in k_values:
        try:
            X, y = create_temporal_dataset(activations_dict, k)
            result = train_and_evaluate_probe(X, y, k, "Raw")
            all_results.append(result)
        except Exception as e:
            print(f"Error for raw k={k}: {e}")

    # Evaluate SAE features
    print("\n" + "="*40)
    print("EVALUATING SAE FEATURES")
    print("="*40)

    activations_dict['current_activations'] = activations_dict['sae_features']

    for k in k_values:
        try:
            X, y = create_temporal_dataset(activations_dict, k)
            result = train_and_evaluate_probe(X, y, k, "SAE")
            all_results.append(result)
        except Exception as e:
            print(f"Error for SAE k={k}: {e}")

    # Create comparison visualization
    create_comparison_plot(all_results)

    # Save results
    with open('large_files/viz/sae_vs_raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n💾 Results saved to large_files/viz/sae_vs_raw_results.json")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: SAE vs RAW RESIDUALS")
    print("="*60)
    print(f"{'Type':<10} {'k':<5} {'Test R²':<10} {'Cosine':<10} {'Features':<10}")
    print("-" * 45)

    for result in all_results:
        print(f"{result['feature_type']:<10} {result['k']:<5} {result['test_r2']:<10.3f} "
              f"{result['mean_cosine']:<10.3f} {result['n_features']:<10}")

    return all_results


def create_comparison_plot(results: List[Dict]):
    """Create visualization comparing SAE vs raw performance."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Separate results by type
    raw_results = [r for r in results if r['feature_type'] == 'Raw']
    sae_results = [r for r in results if r['feature_type'] == 'SAE']

    if not raw_results or not sae_results:
        print("Not enough results for comparison plot")
        return

    k_values = sorted(set(r['k'] for r in raw_results))

    # Plot 1: Test R²
    ax1 = axes[0]
    raw_r2 = [next(r['test_r2'] for r in raw_results if r['k'] == k) for k in k_values]
    sae_r2 = [next(r['test_r2'] for r in sae_results if r['k'] == k) for k in k_values]

    ax1.plot(k_values, raw_r2, 'o-', label='Raw Residuals', linewidth=2, markersize=8)
    ax1.plot(k_values, sae_r2, 's-', label='SAE Features', linewidth=2, markersize=8)
    ax1.set_xlabel('k (tokens ahead)')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Test R² Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cosine Similarity
    ax2 = axes[1]
    raw_cos = [next(r['mean_cosine'] for r in raw_results if r['k'] == k) for k in k_values]
    sae_cos = [next(r['mean_cosine'] for r in sae_results if r['k'] == k) for k in k_values]

    ax2.plot(k_values, raw_cos, 'o-', label='Raw Residuals', linewidth=2, markersize=8)
    ax2.plot(k_values, sae_cos, 's-', label='SAE Features', linewidth=2, markersize=8)
    ax2.set_xlabel('k (tokens ahead)')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Relative Improvement
    ax3 = axes[2]
    r2_improvement = [(s - r) / r * 100 if r > 0 else 0
                      for s, r in zip(sae_r2, raw_r2)]
    cos_improvement = [(s - r) / r * 100 if r > 0 else 0
                       for s, r in zip(sae_cos, raw_cos)]

    x = np.arange(len(k_values))
    width = 0.35

    ax3.bar(x - width/2, r2_improvement, width, label='R² Improvement', alpha=0.8)
    ax3.bar(x + width/2, cos_improvement, width, label='Cosine Improvement', alpha=0.8)
    ax3.set_xlabel('k (tokens ahead)')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('SAE Improvement over Raw')
    ax3.set_xticks(x)
    ax3.set_xticklabels(k_values)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('SAE vs Raw Residuals: Temporal Probe Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    plt.savefig('large_files/viz/sae_vs_raw_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved comparison plot to large_files/viz/sae_vs_raw_comparison.png")


if __name__ == "__main__":
    main()