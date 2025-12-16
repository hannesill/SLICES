# Evaluation Module

This module provides configurable metrics and evaluation utilities for clinical prediction tasks.

## Overview

The `eval` module separates evaluation concerns from training logic:
- **Configurable metrics**: Specify which metrics to compute via config
- **Task-specific defaults**: Sensible minimal metrics per task type
- **Extensible design**: Easy to add new metrics and analysis functions
- **Fairness placeholders**: Interface designed for future fairness analysis

## Quick Start

### Basic Usage

```python
from slices.eval import MetricConfig, build_metrics

# Create metric configuration
config = MetricConfig(
    task_type="binary",
    n_classes=2,
    metrics=["auroc", "auprc"],
)

# Build metrics with prefix for logging
val_metrics = build_metrics(config, prefix="val")
test_metrics = build_metrics(config, prefix="test")

# Use in training loop
val_metrics.update(predictions, labels)
results = val_metrics.compute()
# Results: {'val/auroc': 0.85, 'val/auprc': 0.78}
```

### Using Defaults

```python
from slices.eval import get_default_metrics

# Get default metrics for task type
val_metrics = get_default_metrics("binary", prefix="val")
# Uses default: [auroc, auprc]
```

### In Configuration Files

```yaml
# configs/finetune.yaml or configs/eval/default.yaml
eval:
  metrics:
    # Specify metrics to compute (null = use defaults)
    names: [auroc, auprc, accuracy, f1]

    # Decision threshold for binary classification
    threshold: 0.5
```

## Available Metrics

### Binary Classification
- `auroc` - Area Under ROC Curve (default)
- `auprc` - Area Under Precision-Recall Curve (default)
- `accuracy` - Accuracy at 0.5 threshold
- `f1` - F1 Score

### Multiclass Classification
- `auroc` - One-vs-Rest AUROC (default)
- `auprc` - One-vs-Rest AUPRC
- `accuracy` - Multiclass accuracy (default)
- `f1` - Macro F1 Score

### Multilabel Classification
- `auroc` - Per-label AUROC (default)
- `auprc` - Per-label AUPRC
- `accuracy` - Subset accuracy
- `f1` - Macro F1 Score

### Regression
- Not yet implemented (placeholder)
- Planned: MSE, MAE, R², RMSE

## Integration with Training

The `FineTuneModule` automatically uses eval metrics:

```python
from slices.training import FineTuneModule

# Metrics are built from config
module = FineTuneModule(config)

# During training, metrics are logged automatically
# val/auroc, val/auprc, test/auroc, test/auprc, etc.
```

Metrics are read from `config.eval.metrics.names` if specified, otherwise defaults are used.

## Extending with New Metrics

### Adding a New Metric

1. **Update `AVAILABLE_METRICS` dict**:
```python
AVAILABLE_METRICS = {
    "binary": ["auroc", "auprc", "accuracy", "f1", "sensitivity"],  # Added
    ...
}
```

2. **Add to `_build_metric()` function**:
```python
def _build_metric(name: str, task_type: TaskType, n_classes: int):
    # ...
    elif name == "sensitivity":
        return Recall(task=task, **kwargs)
    # ...
```

3. **Update docstring** to document the new metric

### Adding a New Task Type

1. Add to `TaskType` literal
2. Add to `AVAILABLE_METRICS` and `DEFAULT_METRICS`
3. Handle in `_build_metric()` function

## Fairness Analysis (Planned)

The `fairness.py` module provides placeholders for future fairness analysis:

```python
from slices.eval.fairness import (
    compute_subgroup_metrics,
    compute_demographic_parity,
    compute_equalized_odds,
)

# Not yet implemented - raises NotImplementedError
# Interface designed for future expansion
```

### Planned Features

- **Subgroup performance**: Metrics per demographic group (age, gender, race, insurance)
- **Demographic parity**: Equal positive prediction rates across groups
- **Equalized odds**: Equal TPR/FPR across groups
- **Calibration analysis**: Calibration curves per subgroup

### Available Demographics (MIMIC-IV)

The extractor provides these demographic attributes in `static.parquet`:
- `age` - Patient age at admission
- `gender` - Patient gender (M/F)
- `race` - Patient race/ethnicity
- `insurance` - Insurance type
- `admission_type` - Emergency vs elective
- `admission_location` - Where patient was admitted from

## Design Principles

### Minimal but Extendable
- Start with essential metrics (AUROC, AUPRC)
- Easy to add more without breaking existing code
- No unnecessary complexity

### Configuration-Driven
- Metrics specified in YAML config files
- Sensible defaults that "just work"
- Override when needed for experiments

### Separation of Concerns
- Evaluation logic separate from training logic
- Metrics independent of Lightning module
- Can be used standalone or in training

### Clinical Focus
- Metrics relevant to clinical prediction (AUROC, calibration)
- Demographics available for fairness analysis
- Designed for high-stakes decision support

## Examples

### Different Task Types

```python
# Binary classification (mortality)
binary_config = MetricConfig(task_type="binary", metrics=["auroc", "auprc"])
metrics = build_metrics(binary_config, prefix="val")

# Multiclass classification (diagnosis)
multiclass_config = MetricConfig(
    task_type="multiclass",
    n_classes=5,
    metrics=["auroc", "accuracy"],
)
metrics = build_metrics(multiclass_config, prefix="test")

# Multilabel classification (complications)
multilabel_config = MetricConfig(
    task_type="multilabel",
    n_classes=10,
    metrics=["auroc", "f1"],
)
metrics = build_metrics(multilabel_config, prefix="val")
```

### Custom Metric Sets

```python
# Comprehensive evaluation
full_config = MetricConfig(
    task_type="binary",
    metrics=["auroc", "auprc", "accuracy", "f1"],
)

# Minimal evaluation (faster)
minimal_config = MetricConfig(
    task_type="binary",
    metrics=["auroc"],
)

# Task-specific focus
clinical_config = MetricConfig(
    task_type="binary",
    metrics=["auroc", "auprc"],  # Focus on ranking quality
)
```

### Integration with Hydra Config

```python
# configs/finetune.yaml
defaults:
  - eval: default  # Use configs/eval/default.yaml
  - _self_

task:
  task_name: mortality_24h
  task_type: binary_classification

eval:
  metrics:
    names: [auroc, auprc, f1]  # Override defaults
```

## Testing

See `tests/test_metrics.py` for comprehensive tests covering:
- Metric configuration validation
- Metric building for different task types
- Integration with torchmetrics
- Default metric selection
- Error handling

Run tests:
```bash
uv run pytest tests/test_metrics.py -v
```

## Related Modules

- `slices.training.finetune_module` - Uses eval metrics during training
- `slices.data.labels.base` - Defines `LabelConfig.primary_metric`
- `torchmetrics` - Underlying metric implementations

## Future Work

- [ ] Implement regression metrics (MSE, MAE, R²)
- [ ] Implement fairness analysis functions
- [ ] Add calibration metrics (ECE, Brier score)
- [ ] Add confidence intervals for metrics
- [ ] Add statistical significance testing
- [ ] Create visualization utilities (ROC curves, calibration plots)
