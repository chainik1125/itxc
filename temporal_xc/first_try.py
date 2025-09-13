#!/usr/bin/env python3
# v1.py — Intertemporal Crosscoder: model+dataset bootstrap for sentence→sentence experiments
#
# What this does (v1):
#   1) Loads DeepSeek-R1-Distill-Llama-8B (HF Transformers)
#   2) Streams the Thought Anchors rollouts dataset (uzaymacar/math-rollouts) and filters the 8B subset
#   3) Summarizes: prints example problems, top anchor sentences (by importance), and short reasoning traces
#   4) Provides a batching pipeline (PyTorch Dataset/DataLoader) returning (source_step, next_step, anchor_score, metadata)
#
# Dataset fields & structure (from dataset card):
#   - Directory tree contains per-problem folders with:
#       problem.json              → {problem, level, type, gt_solution, gt_answer, nickname}
#       chunks_labeled.json       → list of dicts with fields including:
#           "chunk" (sentence text), "chunk_idx" (int), "function_tags" (list),
#           "resampling_importance_accuracy", "resampling_importance_kl",
#           "counterfactual_importance_accuracy", "counterfactual_importance_kl",
#           "forced_importance_accuracy", "accuracy", "depends_on", ...
#   - The HF dataset provides rows with columns: {path, filename, extension, size_bytes, content (json string)}
#   - We filter rows by path prefixes containing "deepseek-r1-distill-llama-8b/"
#
# Sources:
#   DeepSeek-R1-Distill-Llama-8B model card: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B  (HF)  [R1-Distill-Llama-8B]  :contentReference[oaicite:1]{index=1}
#   Thought Anchors rollouts dataset card (uzaymacar/math-rollouts): https://huggingface.co/datasets/uzaymacar/math-rollouts  (HF)           [math-rollouts]   :contentReference[oaicite:2]{index=2}

from __future__ import annotations

import argparse
import json
import math
import os
import pwd
import random
import sys
import yaml
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset, IterableDataset
    import datasets
    # Set timeout and streaming configurations
    datasets.config.STREAMING_READ_MAX_RETRIES = 3
    datasets.config.STREAMING_READ_RETRY_INTERVAL = 5
    # Increase download timeout for large datasets
    datasets.config.DOWNLOAD_TIMEOUT = 30  # 30 seconds instead of 10
    datasets.config.STREAMING_DOWNLOAD_MANAGER_TIMEOUT = 30
except Exception as e:
    raise RuntimeError(
        "Please `pip install datasets` (and transformers, torch)."
    ) from e


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ProblemMeta:
    problem_id: str
    nickname: Optional[str]
    level: Optional[str]
    type: Optional[str]
    gt_answer: Optional[str]
    problem_text: str


@dataclass
class ChunkRow:
    chunk_idx: int
    text: str
    function_tags: List[str]
    resampling_importance_accuracy: Optional[float]
    counterfactual_importance_accuracy: Optional[float]
    forced_importance_accuracy: Optional[float]
    accuracy: Optional[float]


@dataclass
class ProblemRecord:
    meta: ProblemMeta
    chunks: List[ChunkRow]  # sorted by chunk_idx


@dataclass
class SentencePair:
    src_text: str
    tgt_text: str
    src_idx: int
    tgt_idx: int
    problem_id: str
    anchor_score: float  # our chosen "anchor-ness" for src step
    anchor_metric: str   # which metric we used (resampling or counterfactual)
    function_tags: List[str]


# -----------------------------
# Configuration
# -----------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge command-line arguments with config, with args taking precedence."""
    # Convert args to dict
    args_dict = vars(args)

    # Create merged config with CLI args overriding config file
    merged = config.copy()

    # Map CLI args to config structure
    if args_dict.get('model_id') is not None:
        merged['model']['model_id'] = args_dict['model_id']
    if args_dict.get('dataset_split') is not None:
        merged['dataset']['split'] = args_dict['dataset_split']
    if args_dict.get('model_subdir') is not None:
        merged['dataset']['model_subdir'] = args_dict['model_subdir']
    if args_dict.get('limit_problems') is not None:
        merged['dataset']['limit_problems'] = args_dict['limit_problems']
    if args_dict.get('seed') is not None:
        merged['dataset']['seed'] = args_dict['seed']
    if args_dict.get('batch_size') is not None:
        merged['training']['batch_size'] = args_dict['batch_size']
    if args_dict.get('max_length') is not None:
        merged['model']['max_length'] = args_dict['max_length']

    return merged


# -----------------------------
# Utilities
# -----------------------------

def json_loads_or_none(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def path_parts(p: str) -> List[str]:
    return [q for q in p.split("/") if q]


def looks_like_problem_json(path: str) -> Optional[str]:
    # Example:
    # math_rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/.../problem_1591/problem.json
    if path.endswith("problem.json") and "problem_" in path:
        # Return problem id like "problem_1591"
        base = path.split("/")[-2]
        if base.startswith("problem_"):
            return base
    return None


def looks_like_chunks_labeled(path: str) -> Optional[str]:
    # .../problem_1591/chunks_labeled.json
    if path.endswith("chunks_labeled.json") and "problem_" in path:
        base = path.split("/")[-2]
        if base.startswith("problem_"):
            return base
    return None


def choose_anchor_score(row: Dict[str, Any]) -> Tuple[float, str]:
    """
    Prefer resampling_importance_accuracy; fallback to counterfactual_importance_accuracy; else 0.0.
    These are documented on the dataset card (importance metrics for sentences).  :contentReference[oaicite:3]{index=3}
    """
    if isinstance(row.get("resampling_importance_accuracy"), (int, float)):
        return float(row["resampling_importance_accuracy"]), "resampling_importance_accuracy"
    if isinstance(row.get("counterfactual_importance_accuracy"), (int, float)):
        return float(row["counterfactual_importance_accuracy"]), "counterfactual_importance_accuracy"
    return 0.0, "none"


# -----------------------------
# Dataset loader for math-rollouts (Thought Anchors)
# -----------------------------

def load_local_problems(
    local_dir: str,
    limit_problems: int = 20,
    verbosity: int = 1,
) -> List[ProblemRecord]:
    """Load problems from local directory (faster than streaming)."""
    from pathlib import Path

    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    problems = []
    problem_dirs = sorted([d for d in local_path.iterdir() if d.is_dir() and d.name.startswith("problem_")])

    if verbosity > 0:
        print(f"  Loading from local directory: {local_dir}")
        print(f"  Found {len(problem_dirs)} problem directories")

    for problem_dir in problem_dirs[:limit_problems]:
        problem_file = problem_dir / "problem.json"
        chunks_file = problem_dir / "chunks_labeled.json"

        if not problem_file.exists() or not chunks_file.exists():
            continue

        # Load problem metadata
        with open(problem_file) as f:
            problem_data = json.load(f)

        meta = ProblemMeta(
            problem_id=problem_dir.name,
            nickname=problem_data.get("nickname"),
            level=problem_data.get("level"),
            type=problem_data.get("type"),
            gt_answer=problem_data.get("gt_answer"),
            problem_text=problem_data.get("problem", "")
        )

        # Load chunks
        with open(chunks_file) as f:
            chunks_data = json.load(f)

        chunks = []
        for c in chunks_data:
            chunks.append(ChunkRow(
                chunk_idx=c.get("chunk_idx", -1),
                text=c.get("chunk", ""),
                function_tags=c.get("function_tags", []),
                resampling_importance_accuracy=c.get("resampling_importance_accuracy"),
                counterfactual_importance_accuracy=c.get("counterfactual_importance_accuracy"),
                forced_importance_accuracy=c.get("forced_importance_accuracy"),
                accuracy=c.get("accuracy")
            ))

        chunks.sort(key=lambda x: x.chunk_idx)
        problems.append(ProblemRecord(meta=meta, chunks=chunks))

        if len(problems) >= limit_problems:
            break

    if verbosity > 0:
        print(f"  ✓ Loaded {len(problems)} problems from local files")

    return problems


def stream_math_rollouts(
    dataset_config: Dict[str, Any],
    verbosity: int = 1,
) -> List[ProblemRecord]:
    """
    Streams the HF dataset and assembles per-problem records based on config.

    Returns:
        A list of ProblemRecord with sorted chunk indices.
    """
    model_subdir = dataset_config.get('model_subdir', 'deepseek-r1-distill-llama-8b')
    split = dataset_config.get('split', 'default')
    limit_problems = dataset_config.get('limit_problems', 20)
    seed = dataset_config.get('seed', 0)
    dataset_name = dataset_config.get('name', 'uzaymacar/math-rollouts')
    streaming = dataset_config.get('streaming', True)
    max_retries = dataset_config.get('max_retries', 3)

    # New: specify which solution type to load
    solution_type = dataset_config.get('solution_type', 'correct_base_solution')
    temperature = dataset_config.get('temperature', 'temperature_0.6_top_p_0.95')

    random.seed(seed)

    # Try loading the dataset with retries
    import time
    import os

    # Set longer timeout for HuggingFace
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'  # 2 minutes timeout

    for attempt in range(max_retries):
        try:
            if verbosity > 0 and attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                print(f"  Retry attempt {attempt + 1}/{max_retries} (waiting {wait_time}s)...")
                time.sleep(wait_time)

            if verbosity > 1:
                print(f"  Loading dataset: {dataset_name}")
                print(f"  Streaming mode: {streaming}")
                print(f"  This should NOT download the full dataset, only metadata...")

            ds = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                trust_remote_code=False,
                download_mode='force_redownload' if attempt > 1 else None,
            )  # IterableDataset of rows with columns: path, filename, extension, size_bytes, content

            if streaming:
                assert isinstance(ds, IterableDataset)
                if verbosity > 1:
                    print(f"  ✓ Dataset loaded as IterableDataset (streaming mode confirmed)")
            else:
                if verbosity > 1:
                    print(f"  Dataset loaded in non-streaming mode")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                if verbosity > 0:
                    print(f"  Dataset loading failed: {str(e)[:100]}... Retrying...")
                import time
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                if verbosity > 0:
                    print(f"  Failed to load dataset after {max_retries} attempts")
                    print("  Consider:")
                    print("    1. Check your internet connection")
                    print("    2. Try again later (HuggingFace might be down)")
                    print("    3. Set streaming=false in config.yaml to download the full dataset")
                    print("    4. Use a smaller limit_problems value")
                raise

    problems: Dict[str, Dict[str, Any]] = {}  # pid -> partial {meta, chunks}
    collected: List[ProblemRecord] = []

    # Build the specific path to search for
    # e.g., "deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution/"
    target_path = f"{model_subdir}/{temperature}/{solution_type}/"

    if verbosity > 0:
        print(f"  Target path: {target_path}")
        print(f"  Looking for {limit_problems} problems...")

    rows_checked = 0
    rows_matched = 0
    import time
    start_time = time.time()
    max_search_time = dataset_config.get('max_search_time', 60)  # From config, default 60s
    max_rows = dataset_config.get('max_rows_to_check', None)  # Optional row limit

    for row in ds:
        rows_checked += 1

        # Check row limit
        if max_rows and rows_checked > max_rows:
            if verbosity > 0:
                print(f"  ⚠ Reached maximum row limit ({max_rows} rows)")
                print(f"  Collected {len(collected)} complete problems so far")
            break

        # Check timeout
        if time.time() - start_time > max_search_time:
            if verbosity > 0:
                print(f"  ⚠ Search timeout after {max_search_time}s (checked {rows_checked} rows, found {rows_matched} matches)")
                print(f"  Collected {len(collected)} complete problems so far")
            break

        # Progress indicator every 100 rows
        if verbosity > 1 and rows_checked % 100 == 0:
            print(f"    Checked {rows_checked} rows, found {rows_matched} matches, {len(collected)} complete problems...")

        p: str = row.get("path", "")
        # Check if this is in our target path
        if not p.startswith(target_path):
            continue

        rows_matched += 1
        content = row.get("content", None)
        if not isinstance(content, str):
            continue

        # Identify problem id & file type
        pid_a = looks_like_problem_json(p)
        pid_b = looks_like_chunks_labeled(p)

        if pid_a:
            j = json_loads_or_none(content)
            if isinstance(j, dict):
                meta = ProblemMeta(
                    problem_id=pid_a,
                    nickname=j.get("nickname"),
                    level=j.get("level"),
                    type=j.get("type"),
                    gt_answer=j.get("gt_answer"),
                    problem_text=j.get("problem") or "",
                )
                entry = problems.setdefault(pid_a, {})
                entry["meta"] = meta

        elif pid_b:
            j = json_loads_or_none(content)
            if isinstance(j, list):
                chunk_rows: List[ChunkRow] = []
                for c in j:
                    # Defensive parsing
                    if not isinstance(c, dict):
                        continue
                    text = str(c.get("chunk", "")).strip()
                    idx = int(c.get("chunk_idx", -1))
                    ftags = list(c.get("function_tags", []) or [])
                    ria = c.get("resampling_importance_accuracy", None)
                    cia = c.get("counterfactual_importance_accuracy", None)
                    fia = c.get("forced_importance_accuracy", None)
                    acc = c.get("accuracy", None)
                    chunk_rows.append(
                        ChunkRow(
                            chunk_idx=idx,
                            text=text,
                            function_tags=ftags,
                            resampling_importance_accuracy=float(ria) if isinstance(ria, (int, float)) else None,
                            counterfactual_importance_accuracy=float(cia) if isinstance(cia, (int, float)) else None,
                            forced_importance_accuracy=float(fia) if isinstance(fia, (int, float)) else None,
                            accuracy=float(acc) if isinstance(acc, (int, float)) else None,
                        )
                    )
                entry = problems.setdefault(pid_b, {})
                entry["chunks"] = chunk_rows

        # Whenever we have both, finalize this problem
        pid = pid_a or pid_b
        if pid and "meta" in problems.get(pid, {}) and "chunks" in problems.get(pid, {}):
            entry = problems.pop(pid)
            meta: ProblemMeta = entry["meta"]
            chunks: List[ChunkRow] = sorted(entry["chunks"], key=lambda r: r.chunk_idx if r.chunk_idx is not None else 1e9)
            collected.append(ProblemRecord(meta=meta, chunks=chunks))
            if len(collected) >= limit_problems:
                if verbosity > 0:
                    elapsed = time.time() - start_time
                    print(f"  ✓ Found {limit_problems} problems in {elapsed:.1f}s (checked {rows_checked} rows)")
                break

    if verbosity > 0 and len(collected) < limit_problems:
        elapsed = time.time() - start_time
        print(f"  Finished searching in {elapsed:.1f}s")
        print(f"  Total rows checked: {rows_checked}")
        print(f"  Rows matching model filter: {rows_matched}")
        print(f"  Complete problems found: {len(collected)}/{limit_problems}")

    return collected


# -----------------------------
# Summaries / Pretty printing
# -----------------------------

def print_problem_brief(p: ProblemRecord, display_config: Dict[str, Any]) -> None:
    max_steps = display_config.get('max_steps_shown', 6)
    top_k_anchors = display_config.get('top_k_anchors', 3)
    truncate_len = display_config.get('text_truncate_length', 180)
    print("=" * 88)
    print(f"[{p.meta.problem_id}] {p.meta.nickname or '(no nickname)'}  |  level={p.meta.level}  type={p.meta.type}  GT={p.meta.gt_answer}")
    print("- Problem text:")
    print(indent_lines(p.meta.problem_text.strip(), prefix="  "))

    # Show first N steps
    steps = [c for c in p.chunks if isinstance(c.chunk_idx, int)]
    print(f"- First {min(max_steps, len(steps))} steps:")
    for c in steps[:max_steps]:
        print(f"  [{c.chunk_idx:>3}] {truncate(c.text, truncate_len)}  tags={','.join(c.function_tags[:3])}")

    # Rank anchors by our chosen metric (resampling_importance_accuracy preferred)
    scored: List[Tuple[float, ChunkRow, str]] = []
    for c in steps:
        s, metric = choose_anchor_score({
            "resampling_importance_accuracy": c.resampling_importance_accuracy,
            "counterfactual_importance_accuracy": c.counterfactual_importance_accuracy,
        })
        scored.append((float(s or 0.0), c, metric))
    scored.sort(key=lambda t: t[0], reverse=True)

    print(f"- Top {min(top_k_anchors, len(scored))} anchor candidates (by importance metric):")
    for score, c, metric in scored[:top_k_anchors]:
        print(f"  [score={score:+.3f} | {metric:>30}]  step[{c.chunk_idx}]: {truncate(c.text, truncate_len)}")


def indent_lines(s: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in s.splitlines())


def truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


# -----------------------------
# Sentence-pair dataset and dataloader
# -----------------------------

class CoTPairsDataset(torch.utils.data.Dataset):
    """
    Builds (src_step -> next_step) sentence pairs across all parsed problems.
    Each item is a SentencePair with auxiliary metadata (anchor score for src).
    """

    def __init__(self, problems: List[ProblemRecord], anchor_metric_order: Tuple[str, str] = ("resampling_importance_accuracy", "counterfactual_importance_accuracy")):
        self._pairs: List[SentencePair] = []
        self._build(problems, anchor_metric_order)

    def _build(self, problems: List[ProblemRecord], anchor_metric_order: Tuple[str, str]):
        for pr in problems:
            steps = [c for c in pr.chunks if isinstance(c.chunk_idx, int)]
            steps.sort(key=lambda c: c.chunk_idx)
            for i in range(len(steps) - 1):
                a = steps[i]
                b = steps[i + 1]
                # Select anchor score metric by preference
                metric_used = "none"
                score_value = 0.0
                for k in anchor_metric_order:
                    v = getattr(a, k, None)
                    if isinstance(v, (int, float)):
                        score_value = float(v)
                        metric_used = k
                        break
                self._pairs.append(
                    SentencePair(
                        src_text=(a.text or "").strip(),
                        tgt_text=(b.text or "").strip(),
                        src_idx=a.chunk_idx,
                        tgt_idx=b.chunk_idx,
                        problem_id=pr.meta.problem_id,
                        anchor_score=score_value,
                        anchor_metric=metric_used,
                        function_tags=a.function_tags or [],
                    )
                )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> SentencePair:
        return self._pairs[idx]


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int = 512):
    """
    Collate function that tokenizes source and target sentences separately.
    We keep them separate so you can feed either or both into the model and hook hidden states.
    """
    def _collate(batch: List[SentencePair]) -> Dict[str, Any]:
        src_texts = [b.src_text for b in batch]
        tgt_texts = [b.tgt_text for b in batch]

        src_enc = tokenizer(
            src_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tgt_enc = tokenizer(
            tgt_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        meta = {
            "problem_id": [b.problem_id for b in batch],
            "src_idx": torch.tensor([b.src_idx for b in batch], dtype=torch.long),
            "tgt_idx": torch.tensor([b.tgt_idx for b in batch], dtype=torch.long),
            "anchor_score": torch.tensor([b.anchor_score for b in batch], dtype=torch.float),
            "anchor_metric": [b.anchor_metric for b in batch],
            "function_tags": [b.function_tags for b in batch],
        }

        return {
            "src": src_enc,
            "tgt": tgt_enc,
            "meta": meta,
        }

    return _collate


# -----------------------------
# Model loader (HF Transformers)
# -----------------------------

def load_reasoning_model(
    model_config: Dict[str, Any],
):
    """
    Loads the reasoning model based on config.
    Enable output_hidden_states=True at call-time when you want internal layer activations.
    """
    model_id = model_config.get('model_id', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    device_map = model_config.get('device_map', 'auto')
    dtype_str = model_config.get('dtype', 'bfloat16')

    # Convert dtype string to torch dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        None: None,
        'null': None,
        'auto': None
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype if torch.cuda.is_available() and dtype is not None else None,
        device_map=device_map,
    )
    return tok, model


# -----------------------------
# Visualization
# -----------------------------

def get_scp_command(filepath: str) -> str:
    """Generate SCP command for downloading files from RunPod."""
    import os
    import socket

    # Try to get RunPod connection info from environment
    runpod_ip = os.environ.get('RUNPOD_PUBLIC_IP', '')
    runpod_pod_id = os.environ.get('RUNPOD_POD_ID', '')

    # If not in RunPod environment, try to get hostname
    if not runpod_ip:
        try:
            hostname = socket.gethostname()
            # Try to resolve public IP
            runpod_ip = socket.gethostbyname(hostname)
        except:
            runpod_ip = 'YOUR_RUNPOD_IP'

    # RunPod maps SSH to a specific external port
    # This is typically a high port number like 49778
    ssh_port = 49778  # RunPod external SSH port

    # Get current user
    username = pwd.getpwuid(os.getuid()).pw_name

    # Get absolute path
    abs_path = os.path.abspath(filepath)

    # Destination directory for downloaded files
    dest_dir = "~/Desktop/scratch/itxc_results/"
    filename = os.path.basename(filepath)

    # Generate SCP command
    if runpod_pod_id:
        # RunPod format with pod ID
        scp_cmd = f"scp -P {ssh_port} {username}@{runpod_ip}:{abs_path} {dest_dir}{filename}"

        # Also provide the RunPod CLI format if available
        runpod_cli_cmd = f"runpod ssh {runpod_pod_id} 'cat {abs_path}' > {dest_dir}{filename}"

        return f"{scp_cmd}\n   # Or using RunPod CLI:\n   # {runpod_cli_cmd}"
    else:
        # Standard format
        return f"scp -P {ssh_port} {username}@{runpod_ip}:{abs_path} {dest_dir}{filename}"


def generate_visualization(problems: List[ProblemRecord], config: Dict[str, Any]) -> List[str]:
    """Generate HTML visualizations for problems if enabled in config."""

    viz_config = config.get('visualization', {})
    if not viz_config.get('enabled', False):
        return []

    verbosity = config.get('logging', {}).get('verbosity', 1)
    output_dir = Path(viz_config.get('output_dir', 'temporal_xc/viz'))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_problems = viz_config.get('max_problems', 3)
    if max_problems == -1:
        max_problems = len(problems)

    generated_files = []

    if verbosity > 0:
        print(f"\n📊 Generating visualizations...")

    # Import visualization module
    try:
        from viz.visualize_problem import generate_html
    except ImportError:
        # Fallback - include the function inline if module not found
        from temporal_xc.viz.visualize_problem import generate_html

    for i, problem in enumerate(problems[:max_problems]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem.meta.problem_id}_{timestamp}.html"
        filepath = output_dir / filename

        if verbosity > 1:
            print(f"  Generating dashboard for {problem.meta.problem_id}...")

        html_content = generate_html(problem, config)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        generated_files.append(str(filepath))

        if verbosity > 0:
            print(f"  ✓ Saved: {filepath}")

    # Auto-open in browser if configured
    if viz_config.get('auto_open', False) and generated_files:
        try:
            webbrowser.open(f"file://{Path(generated_files[0]).absolute()}")
            if verbosity > 0:
                print(f"  🌐 Opened in browser: {generated_files[0]}")
        except Exception as e:
            if verbosity > 0:
                print(f"  ⚠️ Could not auto-open browser: {e}")

    # Print SCP commands for downloading files
    if generated_files and verbosity > 0:
        print("\n📥 To download visualization files to your local machine:")
        print("   (First create the destination directory if it doesn't exist:)")
        print("   mkdir -p ~/Desktop/scratch/itxc_results/")
        print("\n   Then download files:")
        for filepath in generated_files[:3]:  # Show first 3 files
            print(f"\n   {get_scp_command(filepath)}")

        if len(generated_files) > 3:
            print(f"\n   ... and {len(generated_files) - 3} more files")

        # Also show a command to download all files at once
        if len(generated_files) > 1:
            import os
            viz_dir = os.path.dirname(generated_files[0])
            runpod_ip = os.environ.get('RUNPOD_PUBLIC_IP', 'YOUR_RUNPOD_IP')
            username = pwd.getpwuid(os.getuid()).pw_name
            abs_viz_dir = os.path.abspath(viz_dir)
            dest_dir = "~/Desktop/scratch/itxc_results/"
            print(f"\n   # Or download all visualizations at once:")
            print(f"   mkdir -p {dest_dir} && scp -P 49778 {username}@{runpod_ip}:{abs_viz_dir}/*.html {dest_dir}")

    return generated_files


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="v1 bootstrap: model+dataset+summary+batches for sentence→sentence probes")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model_id", type=str, default=None, help="Override model ID from config")
    parser.add_argument("--dataset_split", type=str, default=None, help="Override dataset split from config")
    parser.add_argument("--model_subdir", type=str, default=None, help="Override model subdir from config")
    parser.add_argument("--limit_problems", type=int, default=None, help="Override limit_problems from config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--max_length", type=int, default=None, help="Override max_length from config")
    parser.add_argument("--skip_dataset", action="store_true", help="Skip dataset loading for testing")
    parser.add_argument("--skip_model", action="store_true", help="Skip model loading for testing")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization generation")
    parser.add_argument("--viz_only", action="store_true", help="Only generate visualizations, skip batching demo")
    args = parser.parse_args()

    # Load config and merge with command-line args
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default values...")
        config = {
            'model': {'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'dtype': 'bfloat16', 'device_map': 'auto', 'max_length': 512},
            'dataset': {'name': 'uzaymacar/math-rollouts', 'split': 'default', 'model_subdir': 'deepseek-r1-distill-llama-8b', 'limit_problems': 12, 'streaming': True, 'seed': 0},
            'training': {'batch_size': 12, 'num_workers': 0, 'shuffle': True, 'pin_memory': True},
            'display': {'max_steps_shown': 8, 'top_k_anchors': 3, 'text_truncate_length': 180, 'show_examples': 2},
            'logging': {'verbosity': 1}
        }

    config = merge_args_with_config(args, config)

    torch.set_grad_enabled(False)

    if config.get('logging', {}).get('verbosity', 1) > 0:
        print("=" * 88)
        print("Intertemporal Crosscoder — v1 bootstrap (reasoning model + Thought Anchors rollouts)")
        print("=" * 88)
        print(f"Config file:       {args.config}")
        print(f"Model ID:          {config['model']['model_id']}")
        print(f"Dataset (HF):      {config['dataset']['name']}  [streaming={config['dataset']['streaming']}]")
        print(f"Subset path match: {config['dataset']['model_subdir']}/")
        print(f"Problems to parse: {config['dataset']['limit_problems']}")

    # 1) Load model+tokenizer (HF Transformers)
    if args.skip_model:
        if config.get('logging', {}).get('verbosity', 1) > 0:
            print(f"\nSkipping model loading (--skip_model flag)")
        # Create a dummy tokenizer for testing
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")  # Small dummy tokenizer
        model = None
        device = torch.device("cpu")
    else:
        tok, model = load_reasoning_model(config['model'])
        device = next(model.parameters()).device

        if config.get('logging', {}).get('verbosity', 1) > 0:
            print(f"\nLoaded model. Device map: {device} | dtype: {next(model.parameters()).dtype}")

    # 2) Load dataset slice (Thought Anchors rollouts) — streaming
    verbosity = config.get('logging', {}).get('verbosity', 1)

    if args.skip_dataset:
        if verbosity > 0:
            print("\nSkipping dataset loading (--skip_dataset flag)")
            print("Creating dummy data for testing...")
        # Create dummy problems for testing
        from temporal_xc.test_compile import create_dummy_data
        problems = create_dummy_data()
    elif config['dataset'].get('use_local', False):
        # Use local directory if specified
        local_dir = config['dataset'].get('local_dir', 'temporal_xc/mock_data')
        if verbosity > 0:
            print(f"\nLoading from local directory: {local_dir}")
        problems = load_local_problems(
            local_dir=local_dir,
            limit_problems=config['dataset']['limit_problems'],
            verbosity=verbosity
        )
    else:
        if verbosity > 0:
            print("\nStreaming Thought Anchors rollouts (this only pulls the metadata/content we touch)...")
            print(f"Note: Dataset is set to streaming={config['dataset']['streaming']}")

        problems = stream_math_rollouts(config['dataset'], verbosity=verbosity)
        if not problems:
            print("No problems parsed. Check your internet access or dataset path filter.")
            sys.exit(1)

    # 3) Summarize dataset: show a few problems, top anchor sentences & early steps
    if config.get('logging', {}).get('verbosity', 1) > 0:
        print(f"\nParsed {len(problems)} problems for subset '{config['dataset']['model_subdir']}'. Examples:\n")
        show_examples = min(config.get('display', {}).get('show_examples', 3), len(problems))
        for p in problems[:show_examples]:
            print_problem_brief(p, config.get('display', {}))

    # 3.5) Generate HTML visualizations if enabled
    if args.no_viz:
        if config.get('logging', {}).get('verbosity', 1) > 0:
            print("\n📊 Visualization disabled (--no_viz flag)")
        generated_visualizations = []
    else:
        generated_visualizations = generate_visualization(problems, config)

    # If viz_only mode, exit after generating visualizations
    if args.viz_only:
        if config.get('logging', {}).get('verbosity', 1) > 0:
            print("\n✅ Visualization-only mode complete (--viz_only flag)")
            if generated_visualizations:
                print(f"Generated {len(generated_visualizations)} visualization(s)")
        sys.exit(0)

    # 3.6) Harvest activations for probe training dataset (if configured)
    harvest_config = config['dataset'].get('activation_harvest', {})
    training_config = config['dataset'].get('training_dataset', {})

    # Load TransformerLens model if needed for either harvesting or training dataset
    tl_model = None
    tl_tokenizer = None

    if (harvest_config.get('output_dir') and harvest_config.get('use_transformer_lens', False)) or training_config.get('enabled', False):
        from temporal_xc.make_dataset import load_model_with_tl

        print("\n🔄 Loading model with TransformerLens...")
        tl_model, tl_tokenizer = load_model_with_tl(
            config['model']['model_id'],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if config['model']['dtype'] == 'bfloat16' else torch.float32,
            use_transformer_lens=True,
            hf_model=model if not args.skip_model else None  # Pass existing model if available
        )

    # Original activation harvesting (if configured)
    if harvest_config.get('output_dir'):
        if harvest_config.get('use_transformer_lens', False):
            from temporal_xc.make_dataset import create_activation_dataset_tl

            activation_pairs = create_activation_dataset_tl(
                tl_model, tl_tokenizer, problems, config,
                output_dir=harvest_config['output_dir'],
                verbosity=config.get('logging', {}).get('verbosity', 1)
            )
        else:
            # Use regular PyTorch hooks
            from temporal_xc.make_dataset import create_activation_dataset
            activation_pairs = create_activation_dataset(
                model, tok, problems, config,
                output_dir=harvest_config['output_dir'],
                verbosity=config.get('logging', {}).get('verbosity', 1)
            )

    # 3.7) Create training dataset for k-token-ahead prediction (if configured)
    if training_config.get('enabled', False):
        from temporal_xc.make_dataset import create_multi_k_dataset
        from pathlib import Path

        output_dir = Path(training_config.get('output_dir', 'large_files/training_datasets'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets for different k values
        k_datasets = create_multi_k_dataset(
            tl_model,
            tl_tokenizer,
            problems,
            layer=training_config.get('layer', 19),
            k_values=training_config.get('k_values', [1, 2, 4]),
            hook_name=training_config.get('hook_name', 'resid_post'),
            max_examples_per_problem=training_config.get('max_examples_per_problem', 50),
            min_chunk_length=training_config.get('min_chunk_length', 10),
            verbosity=config.get('logging', {}).get('verbosity', 1)
        )

        # Save datasets
        for k, dataset in k_datasets.items():
            save_path = output_dir / f"training_dataset_k{k}_l{training_config['layer']}.pkl"
            dataset.save(str(save_path))
            print(f"  Saved k={k} dataset to {save_path}")

        # Show summary
        print(f"\n📊 Training Dataset Summary:")
        for k, dataset in k_datasets.items():
            print(f"  k={k}: {len(dataset)} examples")
            train_ds, val_ds = dataset.get_splits(train_ratio=0.8)
            print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 4) Build sentence-pair dataset & a dataloader
    pair_ds = CoTPairsDataset(problems)
    collate = make_collate_fn(tok, max_length=config['model']['max_length'])

    training_config = config.get('training', {})
    loader = torch.utils.data.DataLoader(
        pair_ds,
        batch_size=training_config.get('batch_size', 12),
        shuffle=training_config.get('shuffle', True),
        num_workers=training_config.get('num_workers', 0),
        collate_fn=collate,
        pin_memory=training_config.get('pin_memory', torch.cuda.is_available()),
    )

    # Demo: draw one batch and print shapes & a couple sample pairs
    if config.get('logging', {}).get('verbosity', 1) > 0:
        print("\nBatching demo:")
        batch = next(iter(loader))
        print(f"- src input_ids shape: {tuple(batch['src']['input_ids'].shape)}")
        print(f"- tgt input_ids shape: {tuple(batch['tgt']['input_ids'].shape)}")
        print(f"- meta keys: {list(batch['meta'].keys())}")

        # Show examples based on config
        show_examples = min(config.get('display', {}).get('show_examples', 2), batch['src']['input_ids'].shape[0])
        truncate_len = config.get('display', {}).get('text_truncate_length', 220)

        for i in range(show_examples):
            src_txt = tok.decode(batch["src"]["input_ids"][i], skip_special_tokens=True)
            tgt_txt = tok.decode(batch["tgt"]["input_ids"][i], skip_special_tokens=True)
            meta_i = {k: (batch["meta"][k][i].item() if isinstance(batch["meta"][k], torch.Tensor) else batch["meta"][k][i])
                      for k in batch["meta"]}
            print("-" * 88)
            print(f"[Example {i}] {meta_i['problem_id']}  src_idx={meta_i['src_idx']}→tgt_idx={meta_i['tgt_idx']}")
            print(f"  anchor_score={meta_i['anchor_score']:.3f} ({meta_i['anchor_metric']})  tags={meta_i['function_tags']}")
            print("  src:", truncate(src_txt, truncate_len))
            print("  tgt:", truncate(tgt_txt, truncate_len))

    if config.get('logging', {}).get('verbosity', 1) > 0:
        print("\nReady for next steps:")
        print("  - Hook hidden states with output_hidden_states=True for sentence pooling")
        if config.get('sae', {}).get('enabled', False):
            sae_model = config.get('sae', {}).get('model_id', 'qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19')
            print(f"  - SAE model configured: {sae_model}")
        else:
            print("  - Plug in a frozen SAE for layer ℓ (e.g., qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19) to get feature activations")
        print("  - Train ridge probes Raw→Raw, SAE→SAE, Raw→SAE, SAE→Raw on these (src→tgt) sentence pairs")

        print("\nConfiguration loaded from:", args.config)
        print("Use --config to specify a different configuration file")


if __name__ == "__main__":
    main()
