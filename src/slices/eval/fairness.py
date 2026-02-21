"""Fairness and subgroup analysis utilities.

This module provides metrics for analyzing model performance across
demographic subgroups and computing fairness metrics commonly reported
in clinical ML literature.

Implemented metrics:
- Demographic parity difference: difference in positive prediction rates across groups
- Equalized odds difference: max difference in TPR/FPR across groups
- Disparate impact ratio: ratio of positive prediction rates between groups

Example:
    >>> predictions = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    >>> labels = torch.tensor([1, 0, 1, 0, 1, 0])
    >>> group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    >>> result = compute_demographic_parity(predictions, group_ids, threshold=0.5)
    >>> result["demographic_parity_difference"]
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class FairnessConfig:
    """Configuration for fairness analysis.

    Attributes:
        protected_attributes: List of demographic attributes to analyze.
            Available in MIMIC-IV: age, gender, race, insurance.
        metrics: Metrics to compute per subgroup.
        min_subgroup_size: Minimum samples for subgroup analysis.
    """

    protected_attributes: List[str] = field(default_factory=lambda: ["gender", "race"])
    metrics: List[str] = field(default_factory=lambda: ["auroc", "auprc"])
    min_subgroup_size: int = 50


def _get_unique_groups(group_ids: torch.Tensor) -> List[int]:
    """Get sorted list of unique group IDs.

    Args:
        group_ids: Group identifiers tensor.

    Returns:
        Sorted list of unique group values.
    """
    return sorted(group_ids.unique().tolist())


def _positive_rate(predictions: torch.Tensor, threshold: float) -> float:
    """Compute positive prediction rate.

    Args:
        predictions: Model predictions (probabilities).
        threshold: Classification threshold.

    Returns:
        Fraction of predictions above threshold.
    """
    if len(predictions) == 0:
        return 0.0
    return float((predictions >= threshold).float().mean().item())


def compute_subgroup_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    config: FairnessConfig,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each subgroup.

    Computes positive prediction rate, TPR, and FPR per group.

    Args:
        predictions: Model predictions (probabilities), shape (N,).
        labels: Ground truth binary labels, shape (N,).
        group_ids: Subgroup identifiers for each sample, shape (N,).
        config: Fairness analysis configuration.
        threshold: Classification threshold for binarizing predictions.

    Returns:
        Nested dictionary: {group_id: {metric_name: value}}.
    """
    groups = _get_unique_groups(group_ids)
    result: Dict[str, Dict[str, float]] = {}

    for g in groups:
        mask = group_ids == g
        n = int(mask.sum().item())

        if n < config.min_subgroup_size:
            continue

        g_preds = predictions[mask]
        g_labels = labels[mask]

        group_metrics: Dict[str, float] = {"n_samples": float(n)}
        group_metrics["positive_rate"] = _positive_rate(g_preds, threshold)

        # TPR and FPR
        positives = g_labels == 1
        negatives = g_labels == 0
        n_pos = int(positives.sum().item())
        n_neg = int(negatives.sum().item())

        pred_pos = g_preds >= threshold

        if n_pos > 0:
            group_metrics["tpr"] = float((pred_pos & positives).sum().item()) / n_pos
        if n_neg > 0:
            group_metrics["fpr"] = float((pred_pos & negatives).sum().item()) / n_neg

        result[str(g)] = group_metrics

    return result


def demographic_parity_difference(
    predictions: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute demographic parity difference.

    Measures the maximum absolute difference in positive prediction rates
    across all pairs of groups. A value of 0 indicates perfect parity.

    Args:
        predictions: Model predictions (probabilities), shape (N,).
        group_ids: Subgroup identifiers for each sample, shape (N,).
        threshold: Classification threshold for binarizing predictions.

    Returns:
        Maximum absolute difference in positive prediction rates across groups.
        Returns 0.0 if fewer than 2 groups are present.
    """
    groups = _get_unique_groups(group_ids)
    if len(groups) < 2:
        return 0.0

    rates = []
    for g in groups:
        mask = group_ids == g
        rates.append(_positive_rate(predictions[mask], threshold))

    return max(rates) - min(rates)


def equalized_odds_difference(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute equalized odds difference.

    Measures the maximum of (TPR difference, FPR difference) across all
    pairs of groups. A value of 0 indicates perfect equalized odds.

    TPR difference: max |TPR_a - TPR_b| across group pairs
    FPR difference: max |FPR_a - FPR_b| across group pairs

    Args:
        labels: Ground truth binary labels, shape (N,).
        predictions: Model predictions (probabilities), shape (N,).
        group_ids: Subgroup identifiers for each sample, shape (N,).
        threshold: Classification threshold for binarizing predictions.

    Returns:
        Maximum of (max TPR difference, max FPR difference) across groups.
        Returns 0.0 if fewer than 2 groups are present.
    """
    groups = _get_unique_groups(group_ids)
    if len(groups) < 2:
        return 0.0

    pred_binary = (predictions >= threshold).long()

    tprs: List[float] = []
    fprs: List[float] = []
    n_groups_with_pos = 0
    n_groups_with_neg = 0

    for g in groups:
        mask = group_ids == g
        g_labels = labels[mask]
        g_preds = pred_binary[mask]

        positives = g_labels == 1
        negatives = g_labels == 0
        n_pos = int(positives.sum().item())
        n_neg = int(negatives.sum().item())

        if n_pos > 0:
            tpr = float((g_preds[positives] == 1).sum().item()) / n_pos
            tprs.append(tpr)
        n_groups_with_pos += 1 if n_pos > 0 else 0

        if n_neg > 0:
            fpr = float((g_preds[negatives] == 1).sum().item()) / n_neg
            fprs.append(fpr)
        n_groups_with_neg += 1 if n_neg > 0 else 0

    # If some groups had undefined TPR/FPR (no positives/negatives),
    # the comparison is incomplete — return NaN instead of a misleading 0.0
    if n_groups_with_pos < len(groups) or n_groups_with_neg < len(groups):
        return float("nan")

    tpr_diff = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    fpr_diff = (max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0

    return max(tpr_diff, fpr_diff)


def disparate_impact_ratio(
    predictions: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute disparate impact ratio.

    Measures the ratio of the lowest positive prediction rate to the highest
    across groups. A value of 1.0 indicates perfect parity. The "four-fifths
    rule" considers a ratio below 0.8 as evidence of adverse impact.

    Args:
        predictions: Model predictions (probabilities), shape (N,).
        group_ids: Subgroup identifiers for each sample, shape (N,).
        threshold: Classification threshold for binarizing predictions.

    Returns:
        Ratio of minimum to maximum positive prediction rate across groups.
        Returns 1.0 if fewer than 2 groups exist.
        Returns 1.0 if the maximum rate is 0 (all groups have zero positive rate = perfect parity).
    """
    groups = _get_unique_groups(group_ids)
    if len(groups) < 2:
        return 1.0

    rates = []
    for g in groups:
        mask = group_ids == g
        rates.append(_positive_rate(predictions[mask], threshold))

    max_rate = max(rates)
    min_rate = min(rates)

    if max_rate == 0.0:
        return 1.0

    return min_rate / max_rate


def compute_demographic_parity(
    predictions: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute demographic parity metrics.

    Demographic parity measures whether positive prediction rates
    are equal across groups.

    Args:
        predictions: Model predictions (probabilities), shape (N,).
        group_ids: Subgroup identifiers for each sample, shape (N,).
        threshold: Classification threshold.

    Returns:
        Dictionary with:
        - demographic_parity_difference: max |rate_a - rate_b|
        - disparate_impact_ratio: min_rate / max_rate
        - per_group_rates: {group_id: positive_rate}
    """
    groups = _get_unique_groups(group_ids)

    per_group_rates: Dict[str, float] = {}
    for g in groups:
        mask = group_ids == g
        per_group_rates[str(g)] = _positive_rate(predictions[mask], threshold)

    return {
        "demographic_parity_difference": demographic_parity_difference(
            predictions, group_ids, threshold
        ),
        "disparate_impact_ratio": disparate_impact_ratio(predictions, group_ids, threshold),
        "per_group_rates": per_group_rates,
    }


def compute_equalized_odds(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute equalized odds metrics.

    Equalized odds measures whether TPR and FPR are equal across groups.

    Args:
        predictions: Model predictions (probabilities), shape (N,).
        labels: Ground truth binary labels, shape (N,).
        group_ids: Subgroup identifiers for each sample, shape (N,).
        threshold: Classification threshold.

    Returns:
        Dictionary with:
        - equalized_odds_difference: max(tpr_diff, fpr_diff)
        - per_group_tpr: {group_id: tpr}
        - per_group_fpr: {group_id: fpr}
    """
    groups = _get_unique_groups(group_ids)
    pred_binary = (predictions >= threshold).long()

    per_group_tpr: Dict[str, float] = {}
    per_group_fpr: Dict[str, float] = {}

    for g in groups:
        mask = group_ids == g
        g_labels = labels[mask]
        g_preds = pred_binary[mask]

        positives = g_labels == 1
        negatives = g_labels == 0
        n_pos = int(positives.sum().item())
        n_neg = int(negatives.sum().item())

        if n_pos > 0:
            per_group_tpr[str(g)] = float((g_preds[positives] == 1).sum().item()) / n_pos
        if n_neg > 0:
            per_group_fpr[str(g)] = float((g_preds[negatives] == 1).sum().item()) / n_neg

    return {
        "equalized_odds_difference": equalized_odds_difference(
            labels, predictions, group_ids, threshold
        ),
        "per_group_tpr": per_group_tpr,
        "per_group_fpr": per_group_fpr,
    }
