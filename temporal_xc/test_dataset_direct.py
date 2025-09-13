#!/usr/bin/env python3
"""Test loading a specific example from the dataset."""

from datasets import load_dataset
import json

print("Loading dataset (streaming mode)...")
ds = load_dataset("uzaymacar/math-rollouts", split="default", streaming=True)

# Target path pattern
target_path = "deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution/"

print(f"\nSearching for files in: {target_path}")
print("Looking for first problem/chunks pair...\n")

found_problem = None
found_chunks = None
problem_id = None
rows_checked = 0
max_rows = 1000  # Limit search

for row in ds:
    rows_checked += 1
    path = row.get("path", "")

    # Check if this is in the correct_base_solution directory
    if target_path in path:
        filename = row.get("filename", "")

        if filename == "problem.json" and not found_problem:
            # Extract problem ID from path
            parts = path.split("/")
            for i, part in enumerate(parts):
                if part.startswith("problem_"):
                    problem_id = part
                    break

            found_problem = row
            print(f"✓ Found problem.json at row {rows_checked}")
            print(f"  Problem ID: {problem_id}")
            print(f"  Full path: {path}")

            # Parse and show problem
            content = json.loads(row["content"])
            print(f"  Problem text: {content.get('problem', '')[:200]}...")
            print(f"  Level: {content.get('level')}")
            print(f"  Type: {content.get('type')}")

        elif filename == "chunks_labeled.json" and problem_id and path.endswith(f"{problem_id}/chunks_labeled.json"):
            found_chunks = row
            print(f"\n✓ Found matching chunks_labeled.json at row {rows_checked}")

            # Parse and show first few chunks
            chunks = json.loads(row["content"])
            print(f"  Number of chunks: {len(chunks)}")
            print("\n  First 3 chunks:")
            for i, chunk in enumerate(chunks[:3]):
                text = chunk.get("chunk", "")[:100]
                importance = chunk.get("resampling_importance_accuracy", 0)
                print(f"    [{i}] {text}... (importance: {importance:.3f})")

            break  # We have both files

    if rows_checked >= max_rows:
        print(f"\nReached max rows ({max_rows}). Stopping search.")
        break

if found_problem and found_chunks:
    print("\n" + "="*60)
    print("SUCCESS! Found a complete problem pair.")
    print("This confirms the structure and we can load specific problems.")
    print("\nPath pattern to use:")
    print(f"  {target_path}problem_*/")
else:
    print("\nCouldn't find complete pair in first", rows_checked, "rows")
    print("The files might be further in the dataset.")