"""Simplified actual token recovery evaluation."""

import torch
from temporal_xc.make_dataset import ProbeTrainingDataset
from temporal_xc.train_probe import LinearProbe, ProbeTrainer, create_dataloaders
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_k(k: int, num_examples: int = 20):
    """Evaluate actual token recovery for k tokens ahead."""

    print(f"\n{'='*60}")
    print(f"TOKEN RECOVERY FOR k={k}")
    print(f"{'='*60}")

    # Load dataset
    dataset_path = f"large_files/training_datasets/training_dataset_k{k}_l19.pkl"
    dataset = ProbeTrainingDataset.load(dataset_path)

    # Train probe
    example_src, example_tgt = dataset[0]
    probe = LinearProbe(example_src.shape[0], example_tgt.shape[0])
    train_loader, test_loader = create_dataloaders(dataset, batch_size=32)

    trainer = ProbeTrainer(probe, train_loader, test_loader, device="cpu", lr=1e-3)
    print(f"Training probe...")
    trainer.train(30, verbose=False)

    test_loss, test_r2 = trainer.evaluate(test_loader)
    print(f"Test R² = {test_r2:.3f}")

    # Load model
    print("Loading model...")
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

    # Evaluate on examples
    model.eval()
    probe.eval()

    test_examples = dataset.examples[:num_examples]
    correct = 0
    total = 0

    print(f"\nEvaluating {num_examples} examples:")
    print("-" * 40)

    for idx, example in enumerate(test_examples):
        chunk_text = example.metadata.get('chunk_text', '')
        if not chunk_text:
            continue

        tokens = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=256)['input_ids']

        if tokens.shape[1] <= example.tgt_token_idx:
            continue

        # Get actual target token
        actual_token_id = tokens[0, example.tgt_token_idx].item()
        actual_token = tokenizer.decode([actual_token_id])

        # Get model prediction
        with torch.no_grad():
            truncated = tokens[:, :example.src_token_idx + 1]

            if k == 1:
                # Direct next token prediction
                outputs = model(truncated)
                logits = outputs.logits
                predicted_id = logits[0, -1].argmax().item()
            else:
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

        predicted_token = tokenizer.decode([predicted_id]) if predicted_id != -1 else "?"

        is_correct = (predicted_id == actual_token_id)
        if is_correct:
            correct += 1
        total += 1

        # Show first 3 examples
        if idx < 3:
            status = "✓" if is_correct else "✗"
            print(f"{status} [{example.src_token_idx}→{example.tgt_token_idx}] '{actual_token}' vs '{predicted_token}'")

    accuracy = correct / total if total > 0 else 0
    print("-" * 40)
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")

    return test_r2, accuracy, correct, total


# Main execution
print("ACTUAL TOKEN RECOVERY EVALUATION")
print("="*60)

results = {}
for k in [1, 2, 4]:
    try:
        r2, acc, correct, total = evaluate_k(k, num_examples=20)
        results[k] = {'r2': r2, 'accuracy': acc, 'correct': correct, 'total': total}
    except Exception as e:
        print(f"Error for k={k}: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'k':<5} {'Test R²':<10} {'Token Accuracy':<20}")
print("-" * 35)
for k, r in results.items():
    print(f"{k:<5} {r['r2']:<10.3f} {r['accuracy']:.1%} ({r['correct']}/{r['total']})")