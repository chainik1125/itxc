#!/usr/bin/env python3
"""Generate HTML dashboard to visualize math problems and their reasoning chains."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import colorsys

from first_try import (
    stream_math_rollouts,
    load_local_problems,
    ProblemRecord,
    ChunkRow,
    load_config
)

def importance_to_color(importance: float, alpha: float = 0.3) -> str:
    """Convert importance score to a color (red=high, blue=low)."""
    if importance is None or importance == 0:
        return f"rgba(128, 128, 128, {alpha})"

    # Normalize to 0-1 range (importance can be negative)
    norm_importance = (importance + 1) / 2  # Assuming importance is in [-1, 1]
    norm_importance = max(0, min(1, norm_importance))

    # Use hue from blue (240) to red (0)
    hue = 240 * (1 - norm_importance)
    r, g, b = colorsys.hsv_to_rgb(hue/360, 0.7, 0.9)

    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"

def generate_thought_anchors_url(
    model: str = "deepseek-r1-distill-llama-8b",
    problem_id: str = "problem_1591",
    temperature: str = "temperature_0.6_top_p_0.95",
    solution_type: str = "correct_base_solution"
) -> str:
    """Generate URL to view this problem on thought-anchors.com if it exists."""
    # Based on the paper/website structure
    base_url = "https://thought-anchors.com"
    # This is speculative - the actual URL structure might be different
    return f"{base_url}/explore?model={model}&problem={problem_id}&temp={temperature}&type={solution_type}"

def analyze_importance_flow(problem: ProblemRecord, top_k: int = 5) -> List[Dict[str, Any]]:
    """Analyze which steps are most important and what they influence."""

    # Get importance scores for all steps
    step_importance = []
    for i, chunk in enumerate(problem.chunks):
        importance = chunk.resampling_importance_accuracy or chunk.counterfactual_importance_accuracy or 0
        step_importance.append({
            'idx': i,
            'chunk_idx': chunk.chunk_idx,
            'text': chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text,
            'importance': importance,
            'chunk': chunk
        })

    # Sort by importance and get top K
    top_anchors = sorted(step_importance, key=lambda x: x['importance'], reverse=True)[:top_k]

    # For each top anchor, find the 3 most influenced subsequent steps
    anchor_influences = []
    for anchor in top_anchors:
        anchor_idx = anchor['idx']

        # Look at subsequent steps (within next 10 steps)
        subsequent_scores = []
        for j in range(anchor_idx + 1, min(anchor_idx + 11, len(problem.chunks))):
            if j < len(problem.chunks):
                # Calculate influence based on position and importance
                distance_weight = 1.0 / (j - anchor_idx)  # Closer steps weighted higher
                subsequent_importance = step_importance[j]['importance']
                influence_score = distance_weight * (0.5 + subsequent_importance)

                subsequent_scores.append({
                    'idx': j,
                    'chunk_idx': problem.chunks[j].chunk_idx,
                    'text': problem.chunks[j].text[:80] + '...' if len(problem.chunks[j].text) > 80 else problem.chunks[j].text,
                    'distance': j - anchor_idx,
                    'influence_score': influence_score,
                    'own_importance': subsequent_importance
                })

        # Get top 3 influenced steps
        top_influenced = sorted(subsequent_scores, key=lambda x: x['influence_score'], reverse=True)[:3]

        anchor_influences.append({
            'anchor': anchor,
            'influenced_steps': top_influenced
        })

    return anchor_influences


def generate_html(problem: ProblemRecord, config: Dict[str, Any]) -> str:
    """Generate HTML visualization for a single problem."""

    # Extract config info for Thought Anchors link
    model = config.get('dataset', {}).get('model_subdir', 'deepseek-r1-distill-llama-8b')
    temperature = config.get('dataset', {}).get('temperature', 'temperature_0.6_top_p_0.95')
    solution_type = config.get('dataset', {}).get('solution_type', 'correct_base_solution')

    thought_anchors_url = generate_thought_anchors_url(
        model=model,
        problem_id=problem.meta.problem_id,
        temperature=temperature,
        solution_type=solution_type
    )

    # Analyze importance flow
    anchor_influences = analyze_importance_flow(problem, top_k=5)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Problem Visualization: {problem.meta.problem_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .meta-info {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}

        .meta-badge {{
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}

        .external-link {{
            background: rgba(255,255,255,0.3);
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 5px;
            display: inline-block;
            margin-top: 10px;
            transition: background 0.3s;
        }}

        .external-link:hover {{
            background: rgba(255,255,255,0.4);
        }}

        .main-content {{
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            padding: 30px;
        }}

        .problem-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .problem-text {{
            font-size: 1.1em;
            line-height: 1.8;
            color: #2c3e50;
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 5px;
        }}

        .solution-section {{
            max-height: 80vh;
            overflow-y: auto;
        }}

        .chunk {{
            background: white;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ddd;
            transition: all 0.3s ease;
            position: relative;
        }}

        .chunk:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .chunk-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}

        .chunk-idx {{
            background: #667eea;
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
        }}

        .chunk-tags {{
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }}

        .tag {{
            background: #e9ecef;
            color: #495057;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
        }}

        .chunk-text {{
            margin: 10px 0;
            line-height: 1.7;
        }}

        .importance-bar {{
            height: 4px;
            background: #e9ecef;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }}

        .importance-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
            font-size: 0.85em;
            color: #6c757d;
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 5px;
            background: #f8f9fa;
            border-radius: 4px;
        }}

        .metric-value {{
            font-weight: bold;
            color: #495057;
        }}

        .high-importance {{
            border-left-color: #dc3545 !important;
            background: #fff5f5 !important;
        }}

        .medium-importance {{
            border-left-color: #ffc107 !important;
            background: #fffdf5 !important;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}

        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}

        .stat-label {{
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 5px;
        }}

        .anchor-flow-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-top: 20px;
        }}

        .anchor-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }}

        .anchor-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}

        .anchor-rank {{
            background: #667eea;
            color: white;
            padding: 8px 12px;
            border-radius: 50%;
            font-weight: bold;
            margin-right: 15px;
            min-width: 40px;
            text-align: center;
        }}

        .anchor-text {{
            flex: 1;
            font-weight: 500;
        }}

        .anchor-score {{
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            color: #495057;
            margin-left: 10px;
        }}

        .influenced-steps {{
            margin-top: 15px;
            padding-left: 60px;
        }}

        .influenced-label {{
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 10px;
            font-weight: 500;
        }}

        .influenced-step {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 8px;
            border-left: 3px solid #dee2e6;
            display: flex;
            align-items: center;
            transition: all 0.2s;
        }}

        .influenced-step:hover {{
            background: #e9ecef;
            border-left-color: #667eea;
            transform: translateX(3px);
        }}

        .influenced-step-number {{
            background: #dee2e6;
            color: #495057;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-right: 10px;
            min-width: 60px;
            text-align: center;
        }}

        .influenced-step-text {{
            flex: 1;
            font-size: 0.9em;
        }}

        .influence-indicator {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.8em;
            color: #6c757d;
        }}

        .influence-arrow {{
            color: #667eea;
            font-size: 1.2em;
        }}

        @media (max-width: 768px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}

            h1 {{
                font-size: 1.8em;
            }}

            .influenced-steps {{
                padding-left: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧮 {problem.meta.nickname or problem.meta.problem_id}</h1>
            <div class="meta-info">
                <span class="meta-badge">📊 {problem.meta.level}</span>
                <span class="meta-badge">🏷️ {problem.meta.type}</span>
                <span class="meta-badge">✅ GT: {problem.meta.gt_answer}</span>
                <span class="meta-badge">🔢 {len(problem.chunks)} steps</span>
            </div>
            <a href="{thought_anchors_url}" target="_blank" class="external-link">
                🔗 View on Thought Anchors (if available)
            </a>
        </header>

        <div class="main-content">
            <div class="left-panel">
                <div class="problem-section">
                    <h2>📝 Problem Statement</h2>
                    <div class="problem-text">
                        {problem.meta.problem_text}
                    </div>
                </div>

                <div class="anchor-flow-section">
                    <h2>🎯 Key Anchor Steps & Their Influence</h2>
                    <p style="color: #6c757d; font-size: 0.9em; margin-bottom: 20px;">
                        Top {len(anchor_influences)} most important steps and the subsequent steps they influence most
                    </p>
    """

    # Add anchor influence cards
    for i, influence_data in enumerate(anchor_influences):
        anchor = influence_data['anchor']
        influenced = influence_data['influenced_steps']

        html += f"""
                    <div class="anchor-card">
                        <div class="anchor-header">
                            <div class="anchor-rank">#{i+1}</div>
                            <div class="anchor-text">
                                Step {anchor['chunk_idx']}: {anchor['text']}
                            </div>
                            <div class="anchor-score">Score: {anchor['importance']:.3f}</div>
                        </div>
                        <div class="influenced-steps">
                            <div class="influenced-label">
                                <span class="influence-arrow">→</span> Most influenced subsequent steps:
                            </div>
        """

        for step in influenced:
            distance_label = f"+{step['distance']}"
            html += f"""
                            <div class="influenced-step">
                                <div class="influenced-step-number">Step {step['chunk_idx']} ({distance_label})</div>
                                <div class="influenced-step-text">{step['text']}</div>
                                <div class="influence-indicator">
                                    <span title="Influence score: {step['influence_score']:.2f}">💡</span>
                                </div>
                            </div>
            """

        html += """
                        </div>
                    </div>
        """

    html += """
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{len(problem.chunks)}</div>
                        <div class="stat-label">Total Steps</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len([c for c in problem.chunks if (c.resampling_importance_accuracy or 0) > 0.1])}</div>
                        <div class="stat-label">High Importance</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(1 for c in problem.chunks if c.accuracy and c.accuracy > 0.8)}</div>
                        <div class="stat-label">High Accuracy</div>
                    </div>
                </div>
            </div>

            <div class="solution-section">
                <h2>🤔 Chain of Thought</h2>
                <div class="chunks">
    """

    # Add each chunk
    for chunk in problem.chunks:
        importance = chunk.resampling_importance_accuracy or chunk.counterfactual_importance_accuracy or 0

        # Determine importance class
        importance_class = ""
        if importance > 0.15:
            importance_class = "high-importance"
        elif importance > 0.05:
            importance_class = "medium-importance"

        # Create tags HTML
        tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in (chunk.function_tags or [])])

        # Calculate importance bar width
        importance_width = max(0, min(100, abs(importance) * 100))

        html += f"""
                    <div class="chunk {importance_class}" style="border-left-color: {importance_to_color(importance, 0.8)}">
                        <div class="chunk-header">
                            <span class="chunk-idx">Step {chunk.chunk_idx}</span>
                            <div class="chunk-tags">{tags_html}</div>
                        </div>
                        <div class="chunk-text">{chunk.text}</div>
                        <div class="importance-bar">
                            <div class="importance-fill" style="width: {importance_width}%; background: {importance_to_color(importance, 0.6)}"></div>
                        </div>
                        <div class="metrics">
        """

        if chunk.resampling_importance_accuracy is not None:
            html += f"""
                            <div class="metric">
                                <span>Resampling Importance</span>
                                <span class="metric-value">{chunk.resampling_importance_accuracy:.3f}</span>
                            </div>
            """

        if chunk.counterfactual_importance_accuracy is not None:
            html += f"""
                            <div class="metric">
                                <span>Counterfactual Importance</span>
                                <span class="metric-value">{chunk.counterfactual_importance_accuracy:.3f}</span>
                            </div>
            """

        if chunk.accuracy is not None:
            html += f"""
                            <div class="metric">
                                <span>Accuracy</span>
                                <span class="metric-value">{chunk.accuracy:.3f}</span>
                            </div>
            """

        html += """
                        </div>
                    </div>
        """

    html += """
                </div>
            </div>
        </div>
    </div>

    <script>
        // Add smooth scrolling to high-importance chunks
        document.addEventListener('DOMContentLoaded', function() {
            const chunks = document.querySelectorAll('.chunk');
            chunks.forEach(chunk => {
                chunk.addEventListener('click', function() {
                    this.scrollIntoView({ behavior: 'smooth', block: 'center' });
                });
            });

            // Make anchor cards clickable to jump to the step
            const anchorCards = document.querySelectorAll('.anchor-card');
            anchorCards.forEach(card => {
                card.style.cursor = 'pointer';
                card.addEventListener('click', function(e) {
                    // Don't trigger if clicking on influenced steps
                    if (e.target.closest('.influenced-step')) return;

                    const stepText = this.querySelector('.anchor-text').textContent;
                    const stepNum = stepText.match(/Step (\d+):/)[1];

                    // Find the corresponding chunk in the main view
                    const targetChunk = Array.from(chunks).find(chunk => {
                        const chunkIdx = chunk.querySelector('.chunk-idx').textContent;
                        return chunkIdx === 'Step ' + stepNum;
                    });

                    if (targetChunk) {
                        targetChunk.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        // Highlight briefly
                        targetChunk.style.transition = 'background 0.3s';
                        targetChunk.style.background = '#fff3cd';
                        setTimeout(() => {
                            targetChunk.style.background = '';
                        }, 1500);
                    }
                });
            });

            // Make influenced steps clickable too
            const influencedSteps = document.querySelectorAll('.influenced-step');
            influencedSteps.forEach(step => {
                step.addEventListener('click', function(e) {
                    e.stopPropagation(); // Don't trigger parent card click

                    const stepText = this.querySelector('.influenced-step-number').textContent;
                    const stepNum = stepText.match(/Step (\d+)/)[1];

                    // Find the corresponding chunk
                    const targetChunk = Array.from(chunks).find(chunk => {
                        const chunkIdx = chunk.querySelector('.chunk-idx').textContent;
                        return chunkIdx === 'Step ' + stepNum;
                    });

                    if (targetChunk) {
                        targetChunk.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        // Highlight briefly
                        targetChunk.style.transition = 'background 0.3s';
                        targetChunk.style.background = '#e7f3ff';
                        setTimeout(() => {
                            targetChunk.style.background = '';
                        }, 1500);
                    }
                });
            });
        });
    </script>
</body>
</html>
    """

    return html

def main():
    parser = argparse.ArgumentParser(description="Visualize math problems and reasoning chains")
    parser.add_argument("--config", type=str, default="temporal_xc/config_test.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default="large_files/viz/problem_dashboard.html", help="Output HTML file")
    parser.add_argument("--problem-idx", type=int, default=0, help="Which problem to visualize (0-indexed)")
    parser.add_argument("--use-local", action="store_true", help="Use local mock data")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load problems
    print(f"Loading problems from config: {args.config}")

    if args.use_local or config['dataset'].get('use_local', False):
        local_dir = config['dataset'].get('local_dir', 'temporal_xc/mock_data')
        problems = load_local_problems(
            local_dir=local_dir,
            limit_problems=args.problem_idx + 1,
            verbosity=1
        )
    else:
        problems = stream_math_rollouts(config['dataset'], verbosity=1)

    if not problems:
        print("No problems loaded!")
        return 1

    if args.problem_idx >= len(problems):
        print(f"Problem index {args.problem_idx} out of range (have {len(problems)} problems)")
        return 1

    # Generate HTML
    problem = problems[args.problem_idx]
    print(f"\nGenerating visualization for: {problem.meta.problem_id}")
    print(f"  Title: {problem.meta.nickname or 'Untitled'}")
    print(f"  Steps: {len(problem.chunks)}")

    html = generate_html(problem, config)

    # Save HTML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ Dashboard saved to: {output_path}")
    print(f"📂 Open in browser: file://{output_path.absolute()}")

    return 0

if __name__ == "__main__":
    sys.exit(main())