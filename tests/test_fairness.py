"""Tests for fairness evaluation metrics.

Tests cover:
- demographic_parity_difference: perfect parity, maximum/partial disparity, multi-group
- disparate_impact_ratio: perfect parity, four-fifths rule violations
- equalized_odds_difference: TPR/FPR disparity
- compute_demographic_parity / compute_equalized_odds: aggregate functions
- compute_subgroup_metrics: per-group metrics with min_subgroup_size filtering
"""

import pytest
import torch
from slices.eval.fairness import (
    FairnessConfig,
    compute_demographic_parity,
    compute_equalized_odds,
    compute_subgroup_metrics,
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)


class TestDemographicParity:
    """Tests for demographic_parity_difference and disparate_impact_ratio."""

    def test_perfect_parity(self):
        """Equal positive rates across groups -> difference = 0."""
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(0.0)

    def test_maximum_disparity(self):
        """One group all positive, other all negative -> difference = 1.0."""
        predictions = torch.tensor([0.9, 0.9, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_partial_disparity(self):
        """50% vs 100% positive rate -> difference = 0.5."""
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(0.5)

    def test_single_group_returns_zero(self):
        predictions = torch.tensor([0.9, 0.1])
        group_ids = torch.tensor([0, 0])
        assert demographic_parity_difference(predictions, group_ids) == 0.0

    def test_three_groups(self):
        """With three groups, use max disparity."""
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_custom_threshold(self):
        """Threshold affects which predictions are considered positive."""
        predictions = torch.tensor([0.7, 0.3, 0.7, 0.3])
        group_ids = torch.tensor([0, 0, 1, 1])
        assert demographic_parity_difference(predictions, group_ids, threshold=0.5) == 0.0
        assert demographic_parity_difference(predictions, group_ids, threshold=0.8) == 0.0


class TestDisparateImpact:
    """Tests for disparate_impact_ratio."""

    def test_perfect_parity(self):
        """Equal rates -> ratio = 1.0."""
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        ratio = disparate_impact_ratio(predictions, group_ids, threshold=0.5)
        assert ratio == pytest.approx(1.0)

    def test_four_fifths_violation(self):
        """One group has 50% rate, other has 100% -> ratio = 0.5 (violates 4/5 rule)."""
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        ratio = disparate_impact_ratio(predictions, group_ids, threshold=0.5)
        assert ratio == pytest.approx(0.5)
        assert ratio < 0.8

    def test_zero_max_rate(self):
        """If no group predicts positive, return 0.0."""
        predictions = torch.tensor([0.1, 0.1, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        ratio = disparate_impact_ratio(predictions, group_ids, threshold=0.5)
        assert ratio == 0.0

    def test_single_group(self):
        predictions = torch.tensor([0.9])
        group_ids = torch.tensor([0])
        assert disparate_impact_ratio(predictions, group_ids) == 1.0


class TestEqualizedOdds:
    """Tests for equalized_odds_difference."""

    def test_perfect_equalized_odds(self):
        """Same TPR and FPR across groups -> difference = 0."""
        labels = torch.tensor([1, 0, 1, 0])
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = equalized_odds_difference(labels, predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(0.0)

    def test_tpr_disparity(self):
        """Different TPR across groups."""
        labels = torch.tensor([1, 0, 1, 0])
        predictions = torch.tensor([0.9, 0.1, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = equalized_odds_difference(labels, predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_fpr_disparity(self):
        """Different FPR across groups."""
        labels = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([0.9, 0.9, 0.1, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = equalized_odds_difference(labels, predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_single_group(self):
        labels = torch.tensor([1, 0])
        predictions = torch.tensor([0.9, 0.1])
        group_ids = torch.tensor([0, 0])
        assert equalized_odds_difference(labels, predictions, group_ids) == 0.0


class TestComputeDemographicParity:
    """Tests for the aggregate compute_demographic_parity function."""

    def test_returns_all_fields(self):
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        result = compute_demographic_parity(predictions, group_ids)
        assert "demographic_parity_difference" in result
        assert "disparate_impact_ratio" in result
        assert "per_group_rates" in result
        assert "0" in result["per_group_rates"]
        assert "1" in result["per_group_rates"]

    def test_per_group_rates_correct(self):
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        result = compute_demographic_parity(predictions, group_ids)
        assert result["per_group_rates"]["0"] == pytest.approx(0.5)
        assert result["per_group_rates"]["1"] == pytest.approx(1.0)


class TestComputeEqualizedOdds:
    """Tests for the aggregate compute_equalized_odds function."""

    def test_returns_all_fields(self):
        labels = torch.tensor([1, 0, 1, 0])
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        result = compute_equalized_odds(predictions, labels, group_ids)
        assert "equalized_odds_difference" in result
        assert "per_group_tpr" in result
        assert "per_group_fpr" in result


class TestComputeSubgroupMetrics:
    """Tests for compute_subgroup_metrics."""

    def test_basic_subgroup_metrics(self):
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        group_ids = torch.tensor([0, 0, 1, 1])
        config = FairnessConfig(min_subgroup_size=1)
        result = compute_subgroup_metrics(predictions, labels, group_ids, config)
        assert "0" in result
        assert "1" in result
        assert result["0"]["n_samples"] == 2.0
        assert result["0"]["positive_rate"] == pytest.approx(0.5)

    def test_min_subgroup_size_filters_small_groups(self):
        predictions = torch.tensor([0.9, 0.1, 0.9])
        labels = torch.tensor([1, 0, 1])
        group_ids = torch.tensor([0, 0, 1])
        config = FairnessConfig(min_subgroup_size=2)
        result = compute_subgroup_metrics(predictions, labels, group_ids, config)
        assert "0" in result
        assert "1" not in result
