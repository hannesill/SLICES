# Evaluation Module

This package covers downstream metrics, fairness analysis, statistical testing,
inference helpers, and SSL imputation checks.

## What Is Implemented

- Configurable metric collections in `metrics.py`
- Binary, multiclass, multilabel, and regression metrics
- Fairness primitives in `fairness.py`
- Structured subgroup evaluation in `fairness_evaluator.py`
- Statistical summaries and pairwise tests in `statistical.py`
- Batched inference helper in `inference.py`
- Reconstruction/imputation evaluation for SSL encoders in `imputation.py`

## Metrics

`MetricConfig` and `build_metrics()` are the main entry points.

```python
from slices.eval import MetricConfig, build_metrics

config = MetricConfig(task_type="binary", threshold=0.5)

val_metrics = build_metrics(config, prefix="val")
```

Supported task types:

- `binary`: `auroc`, `auprc`, `accuracy`, `f1`, `precision`, `recall`, `specificity`, `brier_score`, `ece`
- `multiclass`: `auroc`, `auprc`, `accuracy`, `f1`, `precision`, `recall`
- `multilabel`: `auroc`, `auprc`, `accuracy`, `f1`
- `regression`: `mse`, `mae`, `rmse`, `r2`

Default metric sets match the public benchmark export surface:

- Binary: `auroc`, `auprc`, `accuracy`, `f1`, `precision`, `recall`, `specificity`, `brier_score`, `ece`
- Multiclass: `auroc`, `accuracy`
- Multilabel: `auroc`
- Regression: `mse`, `mae`, `r2`

## Fairness

There are two layers:

1. `fairness.py` provides reusable metric primitives such as demographic parity
   difference, equalized odds difference, and disparate impact ratio.
2. `FairnessEvaluator` in `fairness_evaluator.py` builds full subgroup reports
   from predictions, labels, stay IDs, and `static.parquet` demographics.

Current evaluator behavior matches the benchmark pipeline:

- sex and age-group analysis for all supported datasets when columns exist
- race/ethnicity fairness only on MIMIC-IV rows
- canonical race bins: `White`, `Black`, `Hispanic`, `Asian`, `Other`
- subgroup thresholds enforced on unique patients rather than stays
- flattened `fairness/*` output via `flatten_fairness_report()`

Typical usage:

```python
from slices.eval import FairnessEvaluator, flatten_fairness_report

evaluator = FairnessEvaluator(
    static_df=static_df,
    protected_attributes=["gender", "age_group", "race"],
    min_subgroup_size=50,
    task_type="binary",
    dataset_name="miiv",
)

report = evaluator.evaluate(predictions, labels, stay_ids)
flat_report = flatten_fairness_report(report)
```

For batch benchmark sweeps, the repository uses
`scripts/eval/evaluate_fairness.py`, which reruns inference from the recorded
evaluation checkpoint and writes `fairness/*` keys back to W&B.

## Statistical Utilities

`statistical.py` provides the utilities used by the export pipeline:

- `bootstrap_ci`
- `paired_bootstrap_test`
- `paired_wilcoxon_signed_rank`
- `bonferroni_correction`
- `cohens_d`

These are exported from `slices.eval` for direct reuse.

## Inference Helper

`run_inference()` standardizes batched evaluation over a dataloader and returns
predictions, labels, and stay IDs in a format that downstream fairness and
export code can consume.

## Imputation Evaluation

`ImputationEvaluator` measures how well an SSL encoder reconstructs masked
values under controlled masking schemes. This is separate from downstream task
metrics and is intended for representation diagnostics rather than benchmark
headline results.

## Integration Points

- `FineTuneModule` builds metric collections from `config.eval.metrics.*`
- `scripts/eval/evaluate_fairness.py` uses `run_inference()` and `FairnessEvaluator`
- `scripts/export_results.py` consumes both task metrics and `fairness/*` keys

## Extending

When adding a new metric:

1. Register it in `AVAILABLE_METRICS` and `DEFAULT_METRICS` in `metrics.py`
2. Implement it in `_build_metric()`
3. Add regression or classification coverage in `tests/test_metrics.py`
4. Update this README if the public surface changes
