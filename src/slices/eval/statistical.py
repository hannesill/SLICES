"""Bootstrap confidence intervals and statistical tests for metric comparison.

Provides non-parametric bootstrap CIs for any torchmetrics-compatible metric,
and paired bootstrap tests for comparing two models on the same test set.

Example:
    >>> from torchmetrics import AUROC
    >>> ci = bootstrap_ci(AUROC(task="binary"), preds, labels)
    >>> print(f"AUROC: {ci['point']:.3f} ({ci['ci_lower']:.3f}-{ci['ci_upper']:.3f})")
    >>>
    >>> p = paired_bootstrap_test(
    ...     AUROC(task="binary"), preds_a, preds_b, labels
    ... )
    >>> print(f"p-value: {p['p_value']:.4f}")
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
from torchmetrics import Metric


def bootstrap_ci(
    metric_fn: Union[Metric, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: A torchmetrics Metric instance or callable(preds, targets) -> scalar.
        preds: Model predictions, shape (N,) or (N, C).
        targets: Ground truth labels, shape (N,).
        n_bootstraps: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
        - point: Point estimate on full data.
        - ci_lower: Lower bound of CI.
        - ci_upper: Upper bound of CI.
        - std: Bootstrap standard deviation.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    n = len(targets)
    point = _compute_metric(metric_fn, preds, targets)

    scores = []
    for _ in range(n_bootstraps):
        indices = torch.randint(0, n, (n,), generator=generator)
        boot_preds = preds[indices]
        boot_targets = targets[indices]

        score = _compute_metric(metric_fn, boot_preds, boot_targets)
        if score == score:  # skip NaN
            scores.append(score)

    if not scores:
        return {
            "point": point,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "std": float("nan"),
        }

    scores_t = torch.tensor(scores)
    alpha = 1.0 - confidence_level
    ci_lower = torch.quantile(scores_t, alpha / 2).item()
    ci_upper = torch.quantile(scores_t, 1.0 - alpha / 2).item()

    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": scores_t.std().item(),
    }


def paired_bootstrap_test(
    metric_fn: Union[Metric, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    targets: torch.Tensor,
    n_bootstraps: int = 1000,
    seed: Optional[int] = 42,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Paired bootstrap test for comparing two models.

    Tests whether model A is significantly better than model B using
    the paired bootstrap method (Efron & Tibshirani, 1993).

    Args:
        metric_fn: A torchmetrics Metric instance or callable(preds, targets) -> scalar.
        preds_a: Predictions from model A, shape (N,) or (N, C).
        preds_b: Predictions from model B, shape (N,) or (N, C).
        targets: Ground truth labels, shape (N,).
        n_bootstraps: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        higher_is_better: If True, tests H0: metric_A <= metric_B.
            If False, tests H0: metric_A >= metric_B.

    Returns:
        Dictionary with:
        - score_a: Point estimate for model A.
        - score_b: Point estimate for model B.
        - delta: score_a - score_b (positive = A better if higher_is_better).
        - p_value: One-sided p-value.
        - significant_at_005: Whether p < 0.05.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    n = len(targets)
    score_a = _compute_metric(metric_fn, preds_a, targets)
    score_b = _compute_metric(metric_fn, preds_b, targets)

    # Observed delta: positive means A is better when higher_is_better=True
    observed_delta = score_a - score_b

    # Shift-based paired bootstrap test (Efron & Tibshirani, 1993):
    # Under H0 (no difference), bootstrap deltas are centered by subtracting
    # the observed delta. The p-value is the fraction of bootstrap deltas
    # (under H0) that are at least as extreme as the observed delta.
    count_extreme = 0
    valid = 0

    for _ in range(n_bootstraps):
        indices = torch.randint(0, n, (n,), generator=generator)
        boot_a = _compute_metric(metric_fn, preds_a[indices], targets[indices])
        boot_b = _compute_metric(metric_fn, preds_b[indices], targets[indices])

        if boot_a != boot_a or boot_b != boot_b:  # skip NaN
            continue
        valid += 1

        boot_delta = boot_a - boot_b
        # Shift bootstrap delta to be centered under H0:
        # null_delta = boot_delta - observed_delta
        # Test: is null_delta >= observed_delta?
        # Equivalent: boot_delta >= 2 * observed_delta
        if higher_is_better:
            if boot_delta >= 2 * observed_delta:
                count_extreme += 1
        else:
            if boot_delta <= 2 * observed_delta:
                count_extreme += 1

    p_value = count_extreme / max(valid, 1)

    return {
        "score_a": score_a,
        "score_b": score_b,
        "delta": score_a - score_b,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
    }


def _compute_metric(
    metric_fn: Union[Metric, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute a metric value, handling both Metric instances and callables."""
    if isinstance(metric_fn, Metric):
        metric_fn.reset()
        metric_fn.update(preds, targets)
        return metric_fn.compute().item()
    else:
        return metric_fn(preds, targets).item()
