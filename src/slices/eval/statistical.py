"""Bootstrap confidence intervals and paired statistical tests.

Provides:
- non-parametric bootstrap CIs for torchmetrics-compatible metrics
- paired bootstrap tests on shared test sets
- paired Wilcoxon signed-rank tests for per-seed / per-task comparisons
- Bonferroni correction for multiple comparisons
- paired Cohen's d effect sizes
"""

import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

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


def paired_wilcoxon_signed_rank(
    values_a: Sequence[float],
    values_b: Sequence[float],
    correction: bool = True,
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test.

    Delegates to SciPy's implementation so small-sample exact handling and
    tied absolute differences match the statistical reference implementation.

    Args:
        values_a: First paired sample.
        values_b: Second paired sample.
        correction: Whether to apply a 0.5 continuity correction.

    Returns:
        Dictionary with Wilcoxon statistic, z-score, p-value, and pair counts.
    """
    pairs = _finite_pairs(values_a, values_b)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {
            "statistic": 0.0,
            "z_score": 0.0,
            "p_value": float("nan"),
            "n_pairs": 0.0,
            "n_nonzero_pairs": 0.0,
        }

    diffs = [a - b for a, b in pairs]
    nonzero_diffs = [diff for diff in diffs if diff != 0.0]
    n_nonzero = len(nonzero_diffs)

    if n_nonzero == 0:
        return {
            "statistic": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "n_pairs": float(n_pairs),
            "n_nonzero_pairs": 0.0,
        }

    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for Wilcoxon signed-rank tests. "
            "Install the project dependencies with `uv sync`."
        ) from exc

    # Exact enumeration is cheap for the benchmark's usual 5-seed comparisons
    # and avoids anti-conservative normal approximations when all ranks tie.
    method = "exact" if n_nonzero <= 50 else "auto"
    result = wilcoxon(
        nonzero_diffs,
        zero_method="wilcox",
        correction=correction,
        alternative="two-sided",
        method=method,
    )
    z_score = getattr(result, "zstatistic", float("nan"))

    return {
        "statistic": float(result.statistic),
        "z_score": float(z_score),
        "p_value": float(result.pvalue),
        "n_pairs": float(n_pairs),
        "n_nonzero_pairs": float(n_nonzero),
    }


def bonferroni_correction(p_values: Sequence[float]) -> list[float]:
    """Apply Bonferroni correction, preserving NaNs."""
    finite_count = sum(0 if _is_nan(p) else 1 for p in p_values)
    if finite_count == 0:
        return [float("nan") for _ in p_values]

    corrected = []
    for p in p_values:
        if _is_nan(p):
            corrected.append(float("nan"))
        else:
            corrected.append(min(float(p) * finite_count, 1.0))
    return corrected


def cohens_d(
    values_a: Sequence[float],
    values_b: Sequence[float],
    paired: bool = False,
) -> float:
    """Compute Cohen's d effect size.

    Args:
        values_a: First sample.
        values_b: Second sample.
        paired: When True, compute the paired effect size using the standard
            deviation of paired differences.
    """
    pairs = _finite_pairs(values_a, values_b)
    if not pairs:
        return float("nan")

    sample_a = [a for a, _ in pairs]
    sample_b = [b for _, b in pairs]

    if paired:
        diffs = [a - b for a, b in pairs]
        return _cohens_d_from_differences(diffs)

    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b
    var_a = _sample_variance(sample_a)
    var_b = _sample_variance(sample_b)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    pooled_std = math.sqrt(max(pooled_var, 0.0))
    if pooled_std == 0.0:
        diff = mean_a - mean_b
        if diff == 0.0:
            return 0.0
        return math.copysign(float("inf"), diff)

    return (mean_a - mean_b) / pooled_std


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


def _finite_pairs(
    values_a: Iterable[float],
    values_b: Iterable[float],
) -> list[tuple[float, float]]:
    pairs = []
    for a, b in zip(values_a, values_b, strict=True):
        a_val = float(a)
        b_val = float(b)
        if math.isfinite(a_val) and math.isfinite(b_val):
            pairs.append((a_val, b_val))
    return pairs


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _cohens_d_from_differences(differences: Sequence[float]) -> float:
    if len(differences) < 2:
        return float("nan")

    mean_diff = sum(differences) / len(differences)
    std_diff = math.sqrt(max(_sample_variance(differences), 0.0))
    if std_diff == 0.0:
        if mean_diff == 0.0:
            return 0.0
        return math.copysign(float("inf"), mean_diff)
    return mean_diff / std_diff


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)
