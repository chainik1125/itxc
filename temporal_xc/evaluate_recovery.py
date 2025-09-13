"""Evaluate recovery accuracy by patching predicted activations into the model."""

import torch
import torch.nn as nn
from pathlib import Path
from temporal_xc.make_dataset import ProbeTrainingDataset, load_model_with_tl
from temporal_xc.train_probe import LinearProbe, ProbeTrainer, create_dataloaders
import json


def evaluate_recovery_for_k(k: int, num_examples: int = 5, verbose: bool = True):
    """Evaluate recovery accuracy for a specific k value."""

    # Load dataset
    dataset_path = f"large_files/training_datasets/training_dataset_k{k}_l19.pkl"
    dataset = ProbeTrainingDataset.load(dataset_path)

    # Create and train probe
    example_src, example_tgt = dataset[0]
    input_dim = example_src.shape[0]
    output_dim = example_tgt.shape[0]

    probe = LinearProbe(input_dim, output_dim)
    train_loader, test_loader = create_dataloaders(dataset, batch_size=32)

    trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)

    print(f"\nTraining probe for k={k}...")
    trainer.train(30, verbose=False)

    # Get test R²
    test_loss, test_r2 = trainer.evaluate(test_loader)
    print(f"Test R² = {test_r2:.3f}")

    # Load model for recovery evaluation
    print("Loading model for recovery evaluation...")
    model, tokenizer = load_model_with_tl(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device="cpu",
        dtype=torch.bfloat16,
        use_transformer_lens=True
    )

    # Evaluate recovery on test examples
    probe.eval()
    test_examples = dataset.examples[:num_examples]
    correct_count = 0

    print(f"\nEvaluating recovery accuracy for k={k}:")
    print("-" * 40)

    for idx, example in enumerate(test_examples):
        # Get chunk text and tokenize
        chunk_text = example.metadata['chunk_text']
        tokens = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=256)['input_ids']

        if tokens.shape[1] <= example.tgt_token_idx:
            continue

        # Get source activation and predict target
        src_activation = example.src_activation.float().unsqueeze(0)
        with torch.no_grad():
            predicted_activation = probe(src_activation)

        # Get actual tokens
        src_token_id = tokens[0, example.src_token_idx].item() if example.src_token_idx < tokens.shape[1] else -1
        tgt_token_id = tokens[0, example.tgt_token_idx].item()

        src_token = tokenizer.decode([src_token_id]) if src_token_id != -1 else "?"
        actual_token = tokenizer.decode([tgt_token_id])

        # Run model with patched activation
        hook_name = f"blocks.19.hook_resid_post"

        def patch_hook(value, hook):
            # Patch the predicted activation at position src_token_idx
            if value.shape[1] > example.src_token_idx:
                # Use the predicted activation for what would be at tgt position
                value = value.clone()
                value[:, example.tgt_token_idx-1, :] = predicted_activation
            return value

        # Forward pass with patching
        with torch.no_grad():
            if hasattr(model, 'run_with_hooks'):
                # TransformerLens model or wrapper
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, patch_hook)]
                )[0]
            elif hasattr(model, 'model'):
                # Wrapper with HF model inside
                outputs = model.model(tokens)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            else:
                # Direct HF model
                outputs = model(tokens)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Get predicted token at target position
        predicted_token_id = logits[0, example.tgt_token_idx - 1].argmax().item()
        predicted_token = tokenizer.decode([predicted_token_id])

        # Check if correct
        is_correct = (predicted_token_id == tgt_token_id)
        if is_correct:
            correct_count += 1

        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"{status} Example {idx+1}:")
            print(f"  Position: {example.src_token_idx} → {example.tgt_token_idx}")
            print(f"  Source token: '{src_token}'")
            print(f"  Actual next: '{actual_token}'")
            print(f"  Predicted: '{predicted_token}'")

    accuracy = correct_count / num_examples
    print("-" * 40)
    print(f"Recovery Accuracy: {accuracy:.1%} ({correct_count}/{num_examples})")

    return {
        'k': k,
        'test_r2': test_r2,
        'accuracy': accuracy,
        'correct': correct_count,
        'total': num_examples
    }


def main():
    """Evaluate recovery for all k values."""
    print("="*60)
    print("RECOVERY ACCURACY EVALUATION")
    print("="*60)

    results = {}

    # Test with smaller dataset first
    for k in [1, 2, 4]:
        try:
            result = evaluate_recovery_for_k(k, num_examples=5)
            results[k] = result
        except Exception as e:
            print(f"Error evaluating k={k}: {e}")

    # Summary
    print("\n" + "="*60)
    print("RECOVERY ACCURACY SUMMARY")
    print("="*60)
    print(f"{'k':<5} {'Test R²':<10} {'Recovery Acc':<15}")
    print("-" * 30)
    for k in sorted(results.keys()):
        r = results[k]
        print(f"{r['k']:<5} {r['test_r2']:<10.3f} {r['accuracy']:<15.1%}")

    # Save results
    with open('recovery_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to recovery_results.json")


if __name__ == "__main__":
    main()