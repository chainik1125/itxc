#!/usr/bin/env python3
"""Test script to verify the code compiles without loading the full dataset."""

import sys
import torch
from temporal_xc.first_try import (
    load_config,
    merge_args_with_config,
    load_reasoning_model,
    ProblemMeta,
    ChunkRow,
    ProblemRecord,
    CoTPairsDataset,
    make_collate_fn,
    print_problem_brief
)

def create_dummy_data():
    """Create dummy problem records for testing."""
    problems = []

    for i in range(3):
        meta = ProblemMeta(
            problem_id=f"problem_{i}",
            nickname=f"Test Problem {i}",
            level="easy",
            type="algebra",
            gt_answer="42",
            problem_text=f"This is test problem {i}. Solve for x."
        )

        chunks = []
        for j in range(5):
            chunks.append(ChunkRow(
                chunk_idx=j,
                text=f"Step {j}: Doing some reasoning here for problem {i}.",
                function_tags=["reasoning", "calculation"],
                resampling_importance_accuracy=0.5 + j * 0.1,
                counterfactual_importance_accuracy=0.4 + j * 0.1,
                forced_importance_accuracy=0.3 + j * 0.1,
                accuracy=0.8 + j * 0.02
            ))

        problems.append(ProblemRecord(meta=meta, chunks=chunks))

    return problems

def main():
    print("=" * 88)
    print("Testing Intertemporal Crosscoder compilation and basic functionality")
    print("=" * 88)

    # Test config loading
    print("\n1. Testing config loading...")
    try:
        config = load_config("temporal_xc/config.yaml")
        print("   ✓ Config loaded successfully")
    except FileNotFoundError:
        print("   ! Config file not found, using defaults")
        config = {
            'model': {'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'dtype': 'bfloat16', 'device_map': 'cpu', 'max_length': 512},
            'dataset': {'name': 'uzaymacar/math-rollouts', 'split': 'default', 'model_subdir': 'deepseek-r1-distill-llama-8b', 'limit_problems': 12, 'streaming': True, 'seed': 0},
            'training': {'batch_size': 2, 'num_workers': 0, 'shuffle': True, 'pin_memory': False},
            'display': {'max_steps_shown': 3, 'top_k_anchors': 2, 'text_truncate_length': 80, 'show_examples': 2},
            'logging': {'verbosity': 1}
        }

    # Don't load the actual model to save time/memory
    print("\n2. Skipping model loading (would load: {})".format(config['model']['model_id']))

    # Test data structures
    print("\n3. Testing data structures...")
    dummy_problems = create_dummy_data()
    print(f"   ✓ Created {len(dummy_problems)} dummy problems")

    # Test problem display
    print("\n4. Testing problem display...")
    print_problem_brief(dummy_problems[0], config.get('display', {}))

    # Test dataset creation
    print("\n5. Testing CoTPairsDataset...")
    pair_ds = CoTPairsDataset(dummy_problems)
    print(f"   ✓ Created dataset with {len(pair_ds)} sentence pairs")

    # Test a sample pair
    sample_pair = pair_ds[0]
    print(f"   Sample pair: '{sample_pair.src_text[:50]}...' -> '{sample_pair.tgt_text[:50]}...'")
    print(f"   Anchor score: {sample_pair.anchor_score:.3f} ({sample_pair.anchor_metric})")

    print("\n" + "=" * 88)
    print("All tests passed! The code compiles and basic functionality works.")
    print("=" * 88)

    return 0

if __name__ == "__main__":
    sys.exit(main())