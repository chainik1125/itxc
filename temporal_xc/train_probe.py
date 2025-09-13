"""Train linear probes to predict t+k activations from t-th activations."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from temporal_xc.make_dataset import ProbeTrainingDataset


class LinearProbe(nn.Module):
    """Simple linear probe for predicting activations."""

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x):
        return self.linear(x)


class ProbeTrainer:
    """Trainer for linear probes with SAE-style training loop."""

    def __init__(
        self,
        probe: LinearProbe,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.probe = probe.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            probe.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Loss function - MSE for continuous activation prediction
        self.criterion = nn.MSELoss()

        # Metrics storage
        self.train_losses = []
        self.test_losses = []
        self.train_r2_scores = []
        self.test_r2_scores = []

    def compute_r2_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute R^2 score for continuous predictions."""
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.probe.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for src_batch, tgt_batch in self.train_loader:
            src_batch = src_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)

            # Forward pass
            predictions = self.probe(src_batch)
            loss = self.criterion(predictions, tgt_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(predictions.detach())
            all_targets.append(tgt_batch.detach())

        # Compute R^2 score
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        r2_score = self.compute_r2_score(all_preds, all_targets)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, r2_score

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a dataset."""
        self.probe.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)

            predictions = self.probe(src_batch)
            loss = self.criterion(predictions, tgt_batch)

            total_loss += loss.item()
            all_preds.append(predictions)
            all_targets.append(tgt_batch)

        # Compute R^2 score
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        r2_score = self.compute_r2_score(all_preds, all_targets)

        avg_loss = total_loss / len(loader)
        return avg_loss, r2_score

    def train(self, num_epochs: int, verbose: bool = True):
        """Full training loop."""
        for epoch in range(num_epochs):
            # Train
            train_loss, train_r2 = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_r2_scores.append(train_r2)

            # Test
            test_loss, test_r2 = self.evaluate(self.test_loader)
            self.test_losses.append(test_loss)
            self.test_r2_scores.append(test_r2)

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train R²: {train_r2:.4f} | "
                      f"Test Loss: {test_loss:.4f} | Test R²: {test_r2:.4f}")

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plots")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.test_losses, label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training & Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # R² curves
        ax2.plot(self.train_r2_scores, label='Train R²')
        ax2.plot(self.test_r2_scores, label='Test R²')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Training & Test R² Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        plt.close()


def create_dataloaders(
    dataset: ProbeTrainingDataset,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders from dataset."""
    # Get train/test split
    train_ds, test_ds = dataset.get_splits(train_ratio=train_ratio, seed=seed)

    # Extract all activations and convert to float32
    train_src = torch.stack([ex.src_activation.float() for ex in train_ds.examples])
    train_tgt = torch.stack([ex.tgt_activation.float() for ex in train_ds.examples])
    test_src = torch.stack([ex.src_activation.float() for ex in test_ds.examples])
    test_tgt = torch.stack([ex.tgt_activation.float() for ex in test_ds.examples])

    # Create TensorDatasets
    train_dataset = TensorDataset(train_src, train_tgt)
    test_dataset = TensorDataset(test_src, test_tgt)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_probe_for_k(
    dataset_path: str,
    k: int,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """Train a probe for a specific k value."""

    # Load dataset
    dataset = ProbeTrainingDataset.load(dataset_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training probe for k={k}")
        print(f"{'='*60}")
        print(f"Dataset: {len(dataset)} examples")
        print(f"Layer: {dataset.config['layer']}")
        print(f"Hook: {dataset.config['hook_name']}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(dataset, batch_size=batch_size)

    # Get dimensions
    example_src, example_tgt = dataset[0]
    input_dim = example_src.shape[0]
    output_dim = example_tgt.shape[0]

    if verbose:
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Create probe
    probe = LinearProbe(input_dim, output_dim)

    # Create trainer
    trainer = ProbeTrainer(
        probe,
        train_loader,
        test_loader,
        device=device,
        lr=lr
    )

    # Train
    trainer.train(num_epochs, verbose=verbose)

    # Final evaluation
    final_train_loss, final_train_r2 = trainer.evaluate(train_loader)
    final_test_loss, final_test_r2 = trainer.evaluate(test_loader)

    results = {
        'k': k,
        'final_train_loss': final_train_loss,
        'final_test_loss': final_test_loss,
        'final_train_r2': final_train_r2,
        'final_test_r2': final_test_r2,
        'train_losses': trainer.train_losses,
        'test_losses': trainer.test_losses,
        'train_r2_scores': trainer.train_r2_scores,
        'test_r2_scores': trainer.test_r2_scores,
    }

    return results, trainer


def train_k0_baseline(dataset_path: str, **kwargs) -> Dict:
    """Train k=0 baseline (predicting same activation)."""

    # Load any dataset just to get dimensions
    dataset = ProbeTrainingDataset.load(dataset_path)
    train_loader, test_loader = create_dataloaders(dataset, batch_size=kwargs.get('batch_size', 32))

    print(f"\n{'='*60}")
    print(f"Computing k=0 baseline (identity prediction)")
    print(f"{'='*60}")

    # For k=0, the optimal prediction is identity
    # Compute baseline MSE and R² for identity prediction
    total_mse = 0
    total_r2 = 0
    n_batches = 0

    with torch.no_grad():
        for src_batch, tgt_batch in test_loader:
            # For k=0 baseline, predict src as tgt (they should be similar)
            src_batch = src_batch.float()
            tgt_batch = tgt_batch.float()
            mse = ((src_batch - tgt_batch) ** 2).mean()

            # R² score
            ss_res = ((tgt_batch - src_batch) ** 2).sum()
            ss_tot = ((tgt_batch - tgt_batch.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot)

            total_mse += mse.item()
            total_r2 += r2.item()
            n_batches += 1

    avg_mse = total_mse / n_batches
    avg_r2 = total_r2 / n_batches

    print(f"Baseline MSE: {avg_mse:.4f}")
    print(f"Baseline R²: {avg_r2:.4f}")

    return {
        'k': 0,
        'final_test_loss': avg_mse,
        'final_test_r2': avg_r2,
        'is_baseline': True
    }


def plot_comprehensive_results(
    results: Dict,
    recovery_results: Dict = None,
    save_path: str = "temporal_xc/probe_results_comprehensive.png"
):
    """Plot comprehensive results including train/test R² and recovery accuracy."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping comprehensive plot")
        return

    # Prepare data
    k_values = []
    train_r2_scores = []
    test_r2_scores = []
    recovery_acc = []

    for k in sorted(results.keys()):
        if k == 0 or 'is_baseline' in results[k]:  # Skip baseline
            continue
        k_values.append(k)
        train_r2_scores.append(results[k].get('final_train_r2', 0))
        test_r2_scores.append(results[k].get('final_test_r2', 0))

        if recovery_results and k in recovery_results:
            recovery_acc.append(recovery_results[k]['accuracy'])
        else:
            recovery_acc.append(None)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Train vs Test R²
    ax1.plot(k_values, train_r2_scores, 'o-', linewidth=2, markersize=8, label='Train R²', color='blue')
    ax1.plot(k_values, test_r2_scores, 's-', linewidth=2, markersize=8, label='Test R²', color='red')
    ax1.set_xlabel('k (tokens ahead)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Train vs Test R² Scores', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right')

    # Add value labels
    for k, train_r2, test_r2 in zip(k_values, train_r2_scores, test_r2_scores):
        ax1.annotate(f'{test_r2:.3f}',
                    xy=(k, test_r2),
                    xytext=(5, -15),
                    textcoords='offset points',
                    fontsize=9,
                    color='red')

    # Plot 2: Test R² and Recovery Accuracy
    ax2.plot(k_values, test_r2_scores, 'o-', linewidth=2, markersize=8, label='Test R²', color='red')

    # Add recovery accuracy if available
    if any(acc is not None for acc in recovery_acc):
        # Create secondary y-axis for recovery accuracy
        ax2_twin = ax2.twinx()
        valid_k = [k for k, acc in zip(k_values, recovery_acc) if acc is not None]
        valid_acc = [acc for acc in recovery_acc if acc is not None]
        ax2_twin.plot(valid_k, valid_acc, '^-', linewidth=2, markersize=8,
                     label='Recovery Acc', color='green')
        ax2_twin.set_ylabel('Recovery Accuracy', fontsize=12, color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2_twin.set_ylim([0, 1])

        # Add value labels for recovery accuracy
        for k, acc in zip(valid_k, valid_acc):
            ax2_twin.annotate(f'{acc:.1%}',
                            xy=(k, acc),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=9,
                            color='green')

    ax2.set_xlabel('k (tokens ahead)', fontsize=12)
    ax2.set_ylabel('Test R² Score', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_title('Test Performance & Recovery Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Add legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    if any(acc is not None for acc in recovery_acc):
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax2.legend(loc='upper right')

    plt.suptitle('Temporal Probe Performance Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved comprehensive results plot to {save_path}")


def evaluate_recovery_accuracy(
    probe: LinearProbe,
    dataset: ProbeTrainingDataset,
    model,
    tokenizer,
    layer: int,
    k: int,
    num_examples: int = 5,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """
    Evaluate if patching predicted activations recovers the correct next tokens.

    This tests if the probe's predictions are good enough that when we patch them
    into the model's forward pass, it generates the correct next tokens.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Recovery Accuracy Evaluation for k={k}")
        print(f"{'='*60}")

    probe.eval()
    results = []

    # Use TransformerLens for easier patching
    from temporal_xc.make_dataset import load_model_with_tl

    # Get a few test examples
    test_examples = dataset.examples[:num_examples]

    for idx, example in enumerate(test_examples):
        # Get the chunk text
        chunk_text = example.metadata['chunk_text']

        # Tokenize the chunk
        tokens = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=256)['input_ids']

        if tokens.shape[1] <= example.tgt_token_idx:
            continue

        # Get source activation and predict target
        src_activation = example.src_activation.float().unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_activation = probe(src_activation)

        # Get actual target token
        actual_token_id = tokens[0, example.tgt_token_idx].item()
        actual_token = tokenizer.decode([actual_token_id])

        # Run model with patching at layer
        hook_name = f"blocks.{layer}.hook_resid_post"

        def patch_hook(activation, hook):
            # Patch in the predicted activation at the target position
            if activation.shape[1] > example.src_token_idx:
                activation[:, example.src_token_idx, :] = predicted_activation
            return activation

        # Forward pass with patching
        with torch.no_grad():
            # Register hook for patching
            if hasattr(model, 'run_with_hooks'):
                # TransformerLens model
                logits = model.run_with_hooks(
                    tokens.to(device),
                    fwd_hooks=[(hook_name, patch_hook)]
                )[0]
            else:
                # Regular model - simplified version
                outputs = model(tokens.to(device))
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Get predicted token at the target position
        predicted_token_id = logits[0, example.tgt_token_idx - 1].argmax().item()
        predicted_token = tokenizer.decode([predicted_token_id])

        # Check if correct
        is_correct = (predicted_token_id == actual_token_id)

        result = {
            'example_idx': idx,
            'chunk_idx': example.chunk_idx,
            'src_pos': example.src_token_idx,
            'tgt_pos': example.tgt_token_idx,
            'actual_token': actual_token,
            'predicted_token': predicted_token,
            'is_correct': is_correct
        }
        results.append(result)

        if verbose:
            print(f"\nExample {idx+1}:")
            print(f"  Chunk {example.chunk_idx}, pos {example.src_token_idx} → {example.tgt_token_idx}")
            print(f"  Actual token: '{actual_token}'")
            print(f"  Predicted token: '{predicted_token}'")
            print(f"  Correct: {is_correct}")

    # Calculate accuracy
    accuracy = sum(r['is_correct'] for r in results) / len(results) if results else 0.0

    if verbose:
        print(f"\nRecovery Accuracy: {accuracy:.2%} ({sum(r['is_correct'] for r in results)}/{len(results)})")

    return {
        'accuracy': accuracy,
        'num_correct': sum(r['is_correct'] for r in results),
        'num_total': len(results),
        'examples': results
    }


def train_single_k(k_value, args_dict):
    """Train a single k value probe (defined at module level for pickling)."""
    from pathlib import Path

    dataset_path = Path(args_dict['dataset_dir']) / f"training_dataset_k{k_value}_l{args_dict['layer']}.pkl"

    if not dataset_path.exists():
        return k_value, None, None

    probe_results, trainer = train_probe_for_k(
        str(dataset_path),
        k=k_value,
        num_epochs=args_dict['num_epochs'],
        batch_size=args_dict['batch_size'],
        lr=args_dict['lr'],
        device=args_dict['device']
    )

    # Save training curves
    plot_path = Path(args_dict['dataset_dir']) / f"training_curves_k{k_value}_l{args_dict['layer']}.png"
    trainer.plot_training_curves(str(plot_path))

    return k_value, probe_results, trainer


def main():
    parser = argparse.ArgumentParser(description="Train linear probes for temporal prediction")
    parser.add_argument('--dataset_dir', type=str, default='large_files/training_datasets',
                       help='Directory containing training datasets')
    parser.add_argument('--layer', type=int, default=19,
                       help='Layer number in dataset filenames')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 4],
                       help='k values to train probes for')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--save_results', type=str, default='probe_results.json',
                       help='Path to save results')
    parser.add_argument('--eval_recovery', action='store_true',
                       help='Evaluate recovery accuracy by patching activations')

    args = parser.parse_args()

    results = {}

    # Train k=0 baseline
    baseline_dataset = Path(args.dataset_dir) / f"training_dataset_k1_l{args.layer}.pkl"
    if baseline_dataset.exists():
        results[0] = train_k0_baseline(
            str(baseline_dataset),
            batch_size=args.batch_size
        )

    # Train probes for each k in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # Prepare args for parallel execution
    args_dict = {
        'dataset_dir': args.dataset_dir,
        'layer': args.layer,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device
    }

    # Use multiprocessing for parallel training
    if len(args.k_values) > 1 and args.device == 'cpu':
        print(f"Training {len(args.k_values)} probes in parallel...")
        with ProcessPoolExecutor(max_workers=min(len(args.k_values), mp.cpu_count())) as executor:
            futures = {executor.submit(train_single_k, k, args_dict): k for k in args.k_values}

            for future in as_completed(futures):
                k_value, probe_results, trainer = future.result()
                if probe_results is not None:
                    results[k_value] = probe_results
                else:
                    print(f"Warning: Dataset for k={k_value} not found")
    else:
        # Sequential training (for GPU or single k value)
        for k in args.k_values:
            k_value, probe_results, trainer = train_single_k(k, args_dict)
            if probe_results is not None:
                results[k] = probe_results
            else:
                print(f"Warning: Dataset for k={k} not found")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Final Test R² Scores")
    print(f"{'='*60}")

    for k in sorted(results.keys()):
        if 'is_baseline' in results[k]:
            print(f"k={k} (baseline): R² = {results[k]['final_test_r2']:.4f}")
        else:
            print(f"k={k}: R² = {results[k]['final_test_r2']:.4f} "
                  f"(Train R² = {results[k]['final_train_r2']:.4f}, "
                  f"Loss = {results[k]['final_test_loss']:.4f})")

    # Evaluate recovery accuracy if requested
    if args.eval_recovery:
        print(f"\n{'='*60}")
        print("RECOVERY ACCURACY EVALUATION")
        print(f"{'='*60}")

        # Load model for patching experiments
        from temporal_xc.make_dataset import load_model_with_tl

        print("\nLoading model for recovery evaluation...")
        model, tokenizer = load_model_with_tl(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # Or from config
            device=args.device,
            dtype=torch.bfloat16,
            use_transformer_lens=True
        )

        def evaluate_single_k_recovery(k_value):
            """Evaluate recovery for a single k value."""
            dataset_path = Path(args.dataset_dir) / f"training_dataset_k{k_value}_l{args.layer}.pkl"
            if not dataset_path.exists():
                return k_value, None

            # Load dataset
            dataset = ProbeTrainingDataset.load(str(dataset_path))

            # Get dimensions and recreate probe
            example_src, example_tgt = dataset[0]
            input_dim = example_src.shape[0]
            output_dim = example_tgt.shape[0]

            # Create and train probe
            probe = LinearProbe(input_dim, output_dim).to(args.device)
            train_loader, test_loader = create_dataloaders(dataset, batch_size=args.batch_size)

            trainer = ProbeTrainer(
                probe, train_loader, test_loader,
                device=args.device, lr=args.lr
            )
            trainer.train(min(30, args.num_epochs), verbose=False)  # Quick training

            # Evaluate recovery
            recovery_result = evaluate_recovery_accuracy(
                probe, dataset, model, tokenizer,
                layer=args.layer, k=k_value,
                num_examples=5,
                device=args.device,
                verbose=False  # Quiet for parallel execution
            )
            return k_value, recovery_result

        recovery_results = {}

        # Parallel recovery evaluation
        if len(args.k_values) > 1 and args.device == 'cpu':
            print(f"Evaluating recovery accuracy for {len(args.k_values)} probes in parallel...")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(len(args.k_values), 4)) as executor:
                futures = {executor.submit(evaluate_single_k_recovery, k): k for k in args.k_values}

                for future in as_completed(futures):
                    k_value, recovery_result = future.result()
                    if recovery_result is not None:
                        recovery_results[k_value] = recovery_result
                        print(f"  k={k_value}: {recovery_result['accuracy']:.1%} accuracy")
        else:
            # Sequential evaluation
            for k in args.k_values:
                k_value, recovery_result = evaluate_single_k_recovery(k)
                if recovery_result is not None:
                    recovery_results[k] = recovery_result
                    print(f"  k={k}: {recovery_result['accuracy']:.1%} accuracy")

        # Summary of recovery results
        print(f"\n{'='*60}")
        print("RECOVERY ACCURACY SUMMARY")
        print(f"{'='*60}")
        for k, res in recovery_results.items():
            print(f"k={k}: {res['accuracy']:.1%} ({res['num_correct']}/{res['num_total']})")
    else:
        recovery_results = None

    # Create comprehensive plot
    plot_comprehensive_results(results, recovery_results)

    # Save results
    # Convert numpy/torch values to Python natives for JSON serialization
    json_results = {}
    for k, res in results.items():
        json_results[k] = {
            'k': res['k'],
            'final_test_loss': float(res.get('final_test_loss', 0)),
            'final_test_r2': float(res.get('final_test_r2', 0)),
            'is_baseline': res.get('is_baseline', False)
        }

    with open(args.save_results, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()