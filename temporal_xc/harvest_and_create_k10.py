"""Harvest activations and create dataset for k=10."""

import torch
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TrainingExample:
    """Single training example."""
    src_activation: torch.Tensor
    tgt_activation: torch.Tensor
    src_token_idx: int
    tgt_token_idx: int
    metadata: Dict[str, Any]

class ProbeTrainingDataset:
    """Dataset for probe training."""

    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example.src_activation, example.tgt_activation

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

def harvest_activations_for_k10(num_problems=10):
    """Harvest activations with longer chunks to support k=10."""

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

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "uzaymacar/math-rollouts",
        split="train",
        streaming=True
    )

    # Create output directory
    output_dir = Path("large_files/activations_k10")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process problems
    problem_count = 0
    all_examples = []

    for idx, example in enumerate(dataset):
        if problem_count >= num_problems:
            break

        # Get reasoning text
        reasoning_text = example.get('correct_base_solution', '')
        if not reasoning_text or len(reasoning_text) < 100:
            continue

        # Split into chunks by sentences
        sentences = reasoning_text.split('. ')

        # Process chunks of 3-4 sentences to get longer sequences
        chunk_size = 4
        for i in range(0, len(sentences) - chunk_size + 1, 2):
            chunk = '. '.join(sentences[i:i+chunk_size]) + '.'

            # Tokenize
            tokens = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=128)['input_ids']
            seq_len = tokens.shape[1]

            # Skip if too short for k=10
            if seq_len <= 10:
                continue

            # Get activations
            with torch.no_grad():
                outputs = model(tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # Get layer 19 activations
                layer_19_activations = hidden_states[19][0].cpu().numpy()  # (seq_len, hidden_dim)

            # Create training examples for k=10
            k = 10
            for j in range(min(20, seq_len - k)):
                src_idx = j
                tgt_idx = j + k

                if tgt_idx >= seq_len:
                    break

                example = TrainingExample(
                    src_activation=torch.tensor(layer_19_activations[src_idx]),
                    tgt_activation=torch.tensor(layer_19_activations[tgt_idx]),
                    src_token_idx=src_idx,
                    tgt_token_idx=tgt_idx,
                    metadata={
                        'chunk_text': chunk,
                        'k': k,
                        'layer': 19,
                        'problem_idx': problem_count
                    }
                )
                all_examples.append(example)

            # Save activation data for this chunk
            save_data = {
                'activations': layer_19_activations,
                'tokens': tokens[0].cpu().numpy(),
                'chunk_text': chunk,
                'problem_idx': problem_count
            }

            save_path = output_dir / f"problem_{problem_count}_chunk_{i}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

        problem_count += 1
        print(f"Processed problem {problem_count}/{num_problems}")

    print(f"Created {len(all_examples)} training examples for k=10")

    # Save dataset
    dataset = ProbeTrainingDataset(all_examples)
    save_path = Path("large_files/training_datasets/training_dataset_k10_l19.pkl")
    dataset.save(save_path)
    print(f"Saved dataset to {save_path}")

    return dataset

if __name__ == "__main__":
    harvest_activations_for_k10(num_problems=20)