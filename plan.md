Intertemporal Crosscoder (Reasoning Models, Sentence→Sentence)

One-sprint hypothesis test: On a reasoning model, sentence-level future residuals are more linearly predictable in SAE feature space than in raw residual space—and this effect is stronger for high-influence (“anchor”) reasoning steps.


TL;DR

We run a hard, falsifiable benchmark on a reasoning-distilled model:

Model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (open-weights, reasoning-distilled on Llama-3.1-8B).

Latents: Pretrained SAE(s) aligned to this backbone (community release for layer 19), or fallback to Llama-3(.1)-8B SAEs (noting mismatch)

Unit: Sentence-level CoT steps (and a token→token ablation).

Task: Fit linear probes 
𝑋𝑡→𝑌𝑡+1Xt→Yt+1 for four mappings: Raw→Raw, SAE→SAE, Raw→SAE, SAE→Raw.

Metrics: Active-feature 
𝑅2R2, AUROC (feature on/off), Precision@K, CCA, ΔCE under tiny residual nudges.

Controls: Anti-copy split, adjacency shuffle, position shift; anchor vs non-anchor buckets

Go/No-Go: SAE→SAE must beat Raw→Raw on active-feature metrics and survive anti-copy; anchors should show a clear lift. Otherwise, kill the idea.

Why this project?

Reasoning models produce long CoT traces. Prior work identifies “thought anchors” (steps that disproportionately influence downstream reasoning). That gives us sentence-level supervision for “does step 
𝑡
t matter to 
𝑡
+
1
t+1?” and lets us stress-test temporal structure in feature space. 
arXiv

We target residual-stream representations at a fixed layer and compare raw vs SAE-feature dynamics. If the SAE basis captures temporally reusable structure, simple linear maps should work better in feature space than in raw space.

## File Organization

**Important**: All generated output files (visualizations, model checkpoints, probe results, etc.) should be saved to the `large_files/` directory to avoid tracking them in git:

- `large_files/viz/` - HTML visualizations and dashboards
- `large_files/models/` - Trained probe weights and checkpoints
- `large_files/results/` - Experimental results and metrics
- `large_files/cache/` - Cached dataset files or embeddings

The `large_files/` directory should be added to `.gitignore` to keep the repository clean and avoid tracking large binary files.




