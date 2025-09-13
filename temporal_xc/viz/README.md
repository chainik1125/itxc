# Problem Visualization Dashboard

This folder contains tools for visualizing math problems and their chain-of-thought reasoning from the Thought Anchors dataset.

## Features

- **Interactive HTML Dashboard**: Beautiful visualization of problems and reasoning steps
- **Importance Highlighting**: Color-coded steps based on importance scores
- **Metrics Display**: Shows resampling importance, counterfactual importance, and accuracy
- **Thought Anchors Integration**: Links to view problems on thought-anchors.com (when available)

## Usage

### Visualize a Real Problem
```bash
python temporal_xc/viz/visualize_problem.py \
    --config temporal_xc/config_test.yaml \
    --output large_files/viz/problem.html
```

### Visualize Local Mock Data
```bash
python temporal_xc/viz/visualize_problem.py \
    --use-local \
    --output large_files/viz/mock_problem.html
```

### Options

- `--config`: Path to config file (default: temporal_xc/config_test.yaml)
- `--output`: Output HTML file path (default: large_files/viz/problem_dashboard.html)
- `--problem-idx`: Which problem to visualize, 0-indexed (default: 0)
- `--use-local`: Use local mock data instead of streaming from HuggingFace

## Dashboard Features

1. **Problem Overview**
   - Problem statement
   - Metadata (level, type, ground truth answer)
   - Statistics (total steps, high importance steps)

2. **Chain of Thought Visualization**
   - Step-by-step reasoning display
   - Color-coded importance (red = high, blue = low)
   - Function tags for each step
   - Importance metrics for each step

3. **External Links**
   - Direct link to Thought Anchors website (if problem exists there)
   - Based on model, temperature, and solution type

## Color Coding

- 🔴 **Red Border**: High importance steps (>0.15)
- 🟡 **Yellow Border**: Medium importance steps (0.05-0.15)
- ⚪ **Gray Border**: Low importance steps (<0.05)

## Example Output

The dashboard shows:
- Dr. Fu Manchu's compound interest problem
- 101 reasoning steps
- Importance scores ranging from -0.021 to 0.190
- Function tags like `problem_setup`, `active_computation`

## Files Generated

All visualization files are saved to `large_files/viz/` to avoid git tracking large HTML files:

- `large_files/viz/problem_1591_TIMESTAMP.html`: Visualization of actual dataset problems
- `large_files/viz/problem_0_TIMESTAMP.html`: Visualization of mock/test data
- Custom output files as specified

Open the HTML files in any modern web browser to view the interactive dashboard.

**Note**: The `large_files/` directory should be added to `.gitignore` to avoid tracking generated output files.