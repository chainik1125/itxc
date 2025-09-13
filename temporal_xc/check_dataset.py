#!/usr/bin/env python3
"""Quick script to understand the dataset structure."""

from datasets import load_dataset
import json

print("Loading dataset info...")
# Try to load just the dataset card/info without streaming all files
ds = load_dataset("uzaymacar/math-rollouts", split="default", streaming=True)

print("\nChecking first 10 rows to understand structure...")
for i, row in enumerate(ds):
    if i >= 10:
        break
    path = row.get("path", "")
    print(f"{i}: {path}")

    # Check if there's a pattern we can use
    if "deepseek-r1-distill-llama-8b" in path:
        print(f"  ^ Found DeepSeek file at position {i}")

print("\nDataset structure:")
print("- Each row is a file in the dataset")
print("- Path indicates model and problem number")
print("- We need to filter by path to get specific model's data")
print("\nThe issue: Files are not grouped by model, so we have to search through them")
print("This is why it's slow even with streaming!")