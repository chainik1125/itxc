#!/usr/bin/env python3
"""Create a local subset of the dataset for faster testing."""

import json
import os
from pathlib import Path

def create_mock_dataset(output_dir="temporal_xc/mock_data", num_problems=5):
    """Create a mock dataset that mimics the structure of math-rollouts."""

    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating mock dataset in {output_dir}...")

    for i in range(num_problems):
        problem_dir = base_path / f"problem_{i}"
        problem_dir.mkdir(exist_ok=True)

        # Create problem.json
        problem_data = {
            "problem": f"Solve the equation: {i+1}x + {i+2} = {i+10}. Find x.",
            "level": "medium",
            "type": "algebra",
            "gt_solution": f"x = {(i+10-i-2)/(i+1):.2f}",
            "gt_answer": f"{(i+10-i-2)/(i+1):.2f}",
            "nickname": f"linear_equation_{i}"
        }

        with open(problem_dir / "problem.json", "w") as f:
            json.dump(problem_data, f, indent=2)

        # Create chunks_labeled.json with reasoning steps
        chunks = []
        for step in range(6):
            chunk = {
                "chunk": f"Step {step}: {'Setting up the equation' if step == 0 else 'Isolating x' if step == 1 else 'Simplifying' if step == 2 else 'Computing' if step == 3 else 'Verifying' if step == 4 else 'Final answer'}. {i+1}x + {i+2} = {i+10}",
                "chunk_idx": step,
                "function_tags": ["reasoning", "algebra"],
                "resampling_importance_accuracy": 0.3 + step * 0.1,
                "counterfactual_importance_accuracy": 0.25 + step * 0.12,
                "forced_importance_accuracy": 0.2 + step * 0.08,
                "accuracy": 0.8 + step * 0.03,
                "depends_on": list(range(step)) if step > 0 else []
            }
            chunks.append(chunk)

        with open(problem_dir / "chunks_labeled.json", "w") as f:
            json.dump(chunks, f, indent=2)

        print(f"  Created problem_{i}")

    print(f"\n✓ Created {num_problems} mock problems in {output_dir}")
    print("\nTo use this mock data:")
    print("1. Update config to use local files instead of HuggingFace")
    print("2. Or modify stream_math_rollouts() to read from local directory")

    return base_path

if __name__ == "__main__":
    create_mock_dataset()