"""Create dataset and evaluate k=6."""

import torch
import torch.nn as nn
from pathlib import Path
from temporal_xc.make_dataset.create_training_dataset import ProbeTrainingDataset, TrainingExample
from temporal_xc.train_probe import LinearProbe, ProbeTrainer, create_dataloaders
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import pickle

def create_dataset_for_k6():
    """Create dataset for k=6 from existing activations."""

    k = 6
    layer = 19

    print(f"Creating dataset for k={k}, layer={layer}")

    # Load saved activations
    base_dir = Path("large_files/activations")
    all_examples = []

    # Process each saved problem
    for activation_file in sorted(base_dir.glob("problem_*_activations.pkl")):
        with open(activation_file, 'rb') as f:
            data = pickle.load(f)

        activations = data['activations']
        tokens = data['tokens']
        chunk_text = data.get('chunk_text', '')

        seq_len = activations.shape[0]

        # Skip if chunk is too short for k=6
        if seq_len <= k:
            continue

        # Create training pairs
        max_pairs_per_chunk = min(30, seq_len - k)

        for i in range(max_pairs_per_chunk):
            src_idx = i
            tgt_idx = i + k

            if tgt_idx >= seq_len:
                break

            example = TrainingExample(
                src_activation=torch.tensor(activations[src_idx]),
                tgt_activation=torch.tensor(activations[tgt_idx]),
                src_token_idx=src_idx,
                tgt_token_idx=tgt_idx,
                metadata={
                    'chunk_text': chunk_text,
                    'k': k,
                    'layer': layer
                }
            )
            all_examples.append(example)

    print(f"Created {len(all_examples)} training examples for k={k}")

    # Create and save dataset
    dataset = ProbeTrainingDataset(all_examples)

    save_dir = Path("large_files/training_datasets")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"training_dataset_k{k}_l{layer}.pkl"

    dataset.save(save_path)
    print(f"Saved dataset to {save_path}")

    return dataset


def evaluate_k6_token_recovery(num_examples=20):
    """Evaluate actual token recovery for k=6."""

    k = 6

    print(f"\n{'='*60}")
    print(f"EVALUATING k={k}")
    print(f"{'='*60}")

    # Load or create dataset
    dataset_path = Path(f"large_files/training_datasets/training_dataset_k{k}_l19.pkl")

    if not dataset_path.exists():
        print(f"Creating dataset for k={k}...")
        dataset = create_dataset_for_k6()
    else:
        print(f"Loading existing dataset for k={k}...")
        dataset = ProbeTrainingDataset.load(dataset_path)

    if len(dataset) < 100:
        print(f"Warning: Only {len(dataset)} examples available")
        num_examples = min(num_examples, len(dataset) // 5)

    # Train probe
    example_src, example_tgt = dataset[0]
    probe = LinearProbe(example_src.shape[0], example_tgt.shape[0])
    train_loader, test_loader = create_dataloaders(dataset, batch_size=32)

    trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)

    print(f"Training probe for k={k}...")
    trainer.train(30, verbose=False)

    # Get metrics
    train_loss, train_r2 = trainer.evaluate(train_loader)
    test_loss, test_r2 = trainer.evaluate(test_loader)
    print(f"Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")

    # Evaluate cosine similarity
    probe.eval()
    test_examples = dataset.examples[:num_examples]

    cosine_sims = []
    for example in test_examples:
        src_activation = example.src_activation.float().unsqueeze(0)
        tgt_activation = example.tgt_activation.float()

        with torch.no_grad():
            predicted = probe(src_activation).squeeze(0)

        cos_sim = torch.nn.functional.cosine_similarity(
            predicted.unsqueeze(0),
            tgt_activation.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)

    mean_cosine = np.mean(cosine_sims)
    print(f"Mean Cosine Similarity = {mean_cosine:.3f}")

    # Load model for token recovery
    print("Loading model for token recovery...")
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

    # Evaluate token recovery
    model.eval()
    correct = 0
    total = 0

    print(f"\nEvaluating {num_examples} examples...")

    for idx, example in enumerate(test_examples):
        chunk_text = example.metadata.get('chunk_text', '')
        if not chunk_text:
            continue

        tokens = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=256)['input_ids']

        if tokens.shape[1] <= example.tgt_token_idx:
            continue

        # Get actual target token
        actual_token_id = tokens[0, example.tgt_token_idx].item()

        # Get model prediction
        with torch.no_grad():
            truncated = tokens[:, :example.src_token_idx + 1]

            # Generate k tokens
            generated = model.generate(
                truncated,
                max_new_tokens=k,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id
            )

            if generated.shape[1] > truncated.shape[1] + k - 1:
                predicted_id = generated[0, truncated.shape[1] + k - 1].item()
            else:
                predicted_id = -1

        if predicted_id == actual_token_id:
            correct += 1
        total += 1

        # Show first few examples
        if idx < 3:
            actual_token = tokenizer.decode([actual_token_id])
            predicted_token = tokenizer.decode([predicted_id]) if predicted_id != -1 else "?"
            status = "✓" if predicted_id == actual_token_id else "✗"
            print(f"{status} Example {idx+1}: '{actual_token}' vs '{predicted_token}'")

    accuracy = correct / total if total > 0 else 0
    print(f"\nToken Recovery Accuracy: {accuracy:.1%} ({correct}/{total})")

    # Estimate accuracy based on cosine similarity
    high_quality = sum(1 for sim in cosine_sims if sim > 0.9) / len(cosine_sims)
    estimated_acc = high_quality * 0.9 + (1 - high_quality) * 0.3

    return {
        'k': k,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mean_cosine': mean_cosine,
        'actual_accuracy': accuracy,
        'estimated_accuracy': estimated_acc,
        'correct': correct,
        'total': total
    }


def main():
    """Run evaluation for k=6."""

    result = evaluate_k6_token_recovery(num_examples=20)

    # Save results
    with open('large_files/viz/k6_actual_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Saved results to large_files/viz/k6_actual_results.json")

    print("\n" + "="*60)
    print("SUMMARY FOR k=6")
    print("="*60)
    print(f"Train R²: {result['train_r2']:.3f}")
    print(f"Test R²: {result['test_r2']:.3f}")
    print(f"Cosine Similarity: {result['mean_cosine']:.3f}")
    print(f"ACTUAL Token Accuracy: {result['actual_accuracy']:.1%} ({result['correct']}/{result['total']})")

    return result


if __name__ == "__main__":
    main()