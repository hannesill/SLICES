---
name: wandb-analysis
description: Analyze Weights & Biases (wandb) training runs for deep learning projects. Use when the user wants to discuss, analyze, or get feedback on ML experiment results stored in wandb. Extracts metrics, configs, and training curves in an LLM-friendly format. Especially useful for clinical deep learning projects where understanding training dynamics matters.
license: Apache-2.0
metadata:
  author: user
  version: "1.0"
  tags:
    - machine-learning
    - wandb
    - experiment-tracking
    - clinical-ml
---

# Wandb Run Analysis Skill

Analyze Weights & Biases training runs and provide insights for ML experiments, with special attention to clinical deep learning projects.

## When to Use This Skill

- User asks to analyze a wandb run
- User wants to discuss training results from an experiment
- User mentions wandb, weights & biases, or experiment tracking
- User wants feedback on training curves, loss dynamics, or model convergence
- User has a run ID/path and wants insights

## Requirements

- Python 3.8+ with `wandb` and `pandas` packages installed
- `WANDB_API_KEY` environment variable set, OR user logged in via `wandb login`
- Network access to `api.wandb.ai`

## Quick Start

1. Get the run path from the user: `<entity>/<project>/<run_id>`
2. Change to the skill directory and run the analysis script:
   ```bash
   cd .claude/skills/wandb-analysis
   python scripts/analyze_run.py <entity>/<project>/<run_id>
   ```
3. Review the output and discuss findings with the user

**Note:** All script commands assume you are in the skill directory (`.claude/skills/wandb-analysis/`).

## Workflow

### Step 1: Identify the Run

Ask the user for the wandb run they want to analyze. They can provide:
- Full run path: `entity/project/run_id`
- Run URL: `https://wandb.ai/entity/project/runs/run_id`
- Just run ID if project context is known

If they provide a URL, extract the path:
```
https://wandb.ai/myteam/clinical-segmentation/runs/abc123
→ myteam/clinical-segmentation/abc123
```

### Step 2: Run Analysis Script

Execute the main analysis script:
```bash
python scripts/analyze_run.py <run_path> [--samples 50] [--output json|text]
```

Options:
- `--samples N`: Number of points to sample from training curves (default: 50)
- `--output`: Output format, `json` for structured data, `text` for readable summary
- `--full-history`: Get complete history instead of sampled (warning: can be large)

### Step 3: Interpret Results

The script outputs:
1. **Run metadata**: Name, state, duration, created time
2. **Configuration**: Hyperparameters, model architecture, data settings
3. **Final metrics**: Summary metrics at end of training
4. **Training dynamics**: Sampled loss curves with computed diagnostics
5. **Flags**: Automatically detected issues (overfitting, non-convergence, etc.)

### Step 4: Provide Analysis

Based on the results, analyze:

**Convergence**
- Did training loss stabilize?
- How many epochs to reach plateau?
- Is learning rate appropriate?

**Overfitting Detection**
- Compare train vs validation loss trends
- Look for validation loss increasing while train loss decreases
- Check gap between final train/val metrics

**Clinical ML Considerations**
- For medical imaging: check class balance, sensitivity/specificity tradeoffs
- For clinical predictions: examine calibration, AUC trends
- Note any metrics specific to the clinical domain (Dice, IoU, etc.)

**Actionable Recommendations**
- Suggest hyperparameter adjustments if issues detected
- Recommend early stopping points
- Identify potential data issues from metric patterns

## Analysis Patterns

### Pattern: Quick Health Check
```bash
python scripts/analyze_run.py entity/project/run_id --output text
```
Use for: Fast overview, checking if training completed successfully

### Pattern: Deep Dive
```bash
python scripts/analyze_run.py entity/project/run_id --samples 100 --output json
```
Use for: Detailed analysis, comparing specific epochs, investigating anomalies

### Pattern: Query Runs by Sprint
```bash
python scripts/query_runs.py --tags "sprint:1" "phase:finetune" --state finished
```
Use for: Finding all runs from a specific sprint/phase without knowing run IDs

### Pattern: Aggregate Metrics Across Seeds
```bash
python scripts/query_runs.py --tags "sprint:1" "phase:finetune" --aggregate --group-by paradigm --metrics "test/auroc,test/auprc"
```
Use for: Computing mean/std performance per paradigm, dataset, or other grouping

### Pattern: Filter by Config Values
```bash
python scripts/query_runs.py --run-filters '{"config.optimizer.lr": 0.001}' --tags "sprint:1b"
```
Use for: Finding runs with specific hyperparameters (LR ablations, mask ratio sweeps)

### Pattern: Query Protocol A vs B
```bash
python scripts/query_runs.py --tags "sprint:1" "protocol:A" --aggregate --group-by paradigm
```
Use for: Comparing linear probing (Protocol A) vs full finetuning (Protocol B) results

### Pattern: Compare Runs
```bash
python scripts/compare_runs.py entity/project run_id_1 run_id_2 [run_id_3...]
```
Use for: Hyperparameter sweep analysis, architecture comparisons

### Pattern: Export for External Analysis
```bash
python scripts/export_run.py entity/project/run_id --format csv
```
Use for: Creating visualizations, statistical analysis outside wandb

## Interpreting Common Issues

### High Train/Val Gap
- **Symptom**: Final train loss << final val loss
- **Likely cause**: Overfitting
- **Suggestions**: Increase regularization, add dropout, reduce model capacity, get more data

### Validation Loss Oscillating
- **Symptom**: Val loss jumps around, doesn't smoothly decrease
- **Likely cause**: Learning rate too high, batch size too small
- **Suggestions**: Reduce LR, increase batch size, use LR scheduling

### Flat Loss Curves
- **Symptom**: Loss barely decreases from start
- **Likely cause**: LR too low, poor initialization, data issue
- **Suggestions**: Increase LR, check data pipeline, verify labels

### Sudden Loss Spikes
- **Symptom**: Loss jumps dramatically mid-training
- **Likely cause**: Gradient explosion, bad batch, numerical instability
- **Suggestions**: Add gradient clipping, check for NaN in data, reduce LR

## Clinical ML Red Flags

When analyzing clinical deep learning runs, specifically watch for:

1. **Class imbalance effects**: Metrics that look good overall but hide poor minority class performance
2. **Overfit to artifacts**: Perfect training metrics with poor validation on held-out data
3. **Calibration issues**: High AUC but poor probability calibration
4. **Data leakage**: Suspiciously high validation performance early in training
5. **Batch effects**: Performance varies significantly across different data sources

## Reference Files

- `clinical-metrics.md`: Guide to interpreting clinical ML metrics (ICU time-series focused)
- `common-architectures.md`: Expected training dynamics for common architectures

## Scripts

- `scripts/query_runs.py`: Query and filter runs by tags, config, state (supports aggregation)
- `scripts/analyze_run.py`: Main analysis script (approaches 1, 2, 4 combined)
- `scripts/compare_runs.py`: Compare multiple runs
- `scripts/export_run.py`: Export to CSV/JSON for external tools

## Available Tags

All runs are tagged with the following (set in configs and `setup_wandb_logger()`):

| Tag | Values | Source |
|-----|--------|--------|
| `phase:{phase}` | pretrain, finetune, supervised | config yaml |
| `dataset:{ds}` | miiv, eicu, combined | config yaml |
| `paradigm:{p}` | mae, jepa, contrastive, supervised | config yaml |
| `seed:{s}` | 42, 123, 456 | config yaml |
| `lr:{lr}` | e.g., 0.001, 0.0001 | config yaml |
| `sprint:{n}` | 1, 1b, 1c, 2, ... 8 | setup_wandb_logger() |
| `protocol:{p}` | A (frozen), B (unfrozen) | setup_wandb_logger() |
| `mask_ratio:{r}` | e.g., 0.3, 0.5, 0.75 | setup_wandb_logger() |
| `label_fraction:{f}` | e.g., 0.01, 0.1, 0.5 | setup_wandb_logger() |
| `revision:{v}` | e.g., v2 | setup_wandb_logger() |
| `rerun-reason:{r}` | free text | setup_wandb_logger() |
