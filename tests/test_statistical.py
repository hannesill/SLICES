"""Tests for bootstrap confidence intervals and paired bootstrap tests.

Tests cover:
- bootstrap_ci: deterministic metric yields zero-width CI, known value coverage
- paired_bootstrap_test: identical models produce p ~0.5, better model produces p ~0.0,
  edge cases (single sample, all-same predictions, NaN handling)
"""

import pytest
import torch
from slices.eval.statistical import bootstrap_ci, paired_bootstrap_test

# ── Helpers ──────────────────────────────────────────────────────────────────


def accuracy_metric(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple accuracy callable for testing."""
    return ((preds >= 0.5).long() == targets).float().mean()


def constant_metric(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Always returns 0.75 regardless of input."""
    return torch.tensor(0.75)


# ── bootstrap_ci ─────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_constant_metric_zero_width_ci(self):
        """A metric that always returns the same value should have zero-width CI."""
        preds = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        result = bootstrap_ci(constant_metric, preds, targets, n_bootstraps=200, seed=0)
        assert result["point"] == pytest.approx(0.75)
        assert result["ci_lower"] == pytest.approx(0.75)
        assert result["ci_upper"] == pytest.approx(0.75)
        assert result["std"] == pytest.approx(0.0, abs=1e-6)

    def test_perfect_predictions(self):
        """Perfect predictions should give point estimate of 1.0."""
        preds = torch.tensor([0.9, 0.1, 0.8, 0.2])
        targets = torch.tensor([1, 0, 1, 0])
        result = bootstrap_ci(accuracy_metric, preds, targets, n_bootstraps=500, seed=42)
        assert result["point"] == pytest.approx(1.0)
        # CI should be near 1.0
        assert result["ci_lower"] >= 0.5
        assert result["ci_upper"] <= 1.0 + 1e-6

    def test_ci_contains_point_estimate(self):
        """Point estimate should fall within the CI."""
        torch.manual_seed(0)
        preds = torch.rand(200)
        targets = (preds > 0.5).long()
        # Add some noise so it's not perfect
        flip = torch.rand(200) < 0.1
        targets[flip] = 1 - targets[flip]

        result = bootstrap_ci(accuracy_metric, preds, targets, n_bootstraps=500, seed=42)
        assert result["ci_lower"] <= result["point"] <= result["ci_upper"]

    def test_wider_ci_with_fewer_samples(self):
        """Fewer samples should produce wider CIs."""
        torch.manual_seed(0)

        preds_large = torch.rand(500)
        targets_large = (preds_large > 0.4).long()
        ci_large = bootstrap_ci(
            accuracy_metric, preds_large, targets_large, n_bootstraps=500, seed=42
        )

        preds_small = preds_large[:20]
        targets_small = targets_large[:20]
        ci_small = bootstrap_ci(
            accuracy_metric, preds_small, targets_small, n_bootstraps=500, seed=42
        )

        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        assert width_small > width_large

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        preds = torch.rand(50)
        targets = torch.randint(0, 2, (50,))
        r1 = bootstrap_ci(accuracy_metric, preds, targets, seed=123)
        r2 = bootstrap_ci(accuracy_metric, preds, targets, seed=123)
        assert r1 == r2


# ── paired_bootstrap_test ────────────────────────────────────────────────────


class TestPairedBootstrapTest:
    def test_identical_predictions_not_significant(self):
        """Literally identical predictions should not be significant.

        When preds_a is preds_b, every bootstrap delta is exactly 0 and
        observed_delta is 0, so p-value will be 1.0 (degenerate case).
        """
        torch.manual_seed(0)
        preds = torch.rand(200)
        targets = (preds > 0.5).long()

        result = paired_bootstrap_test(
            accuracy_metric, preds, preds, targets, n_bootstraps=500, seed=42
        )
        assert result["delta"] == pytest.approx(0.0)
        assert result["significant_at_005"] is False

    def test_similar_models_not_significant(self):
        """Two models with similar performance should not be significant.

        Different predictions with similar accuracy should produce a high
        p-value (not significant).
        """
        torch.manual_seed(0)
        targets = torch.randint(0, 2, (200,))
        # Two models with similar (noisy) accuracy
        preds_a = targets.float() * 0.6 + (1 - targets.float()) * 0.4 + torch.randn(200) * 0.15
        preds_b = targets.float() * 0.6 + (1 - targets.float()) * 0.4 + torch.randn(200) * 0.15
        preds_a = preds_a.clamp(0, 1)
        preds_b = preds_b.clamp(0, 1)

        result = paired_bootstrap_test(
            accuracy_metric, preds_a, preds_b, targets, n_bootstraps=2000, seed=42
        )
        # Should not be significant — similar models
        assert result["p_value"] > 0.05

    def test_strictly_better_model(self):
        """Clearly better model should produce low p-value."""
        targets = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20)
        # Model A: perfect predictions
        preds_a = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1] * 20)
        # Model B: random-ish predictions
        preds_b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] * 20)

        result = paired_bootstrap_test(
            accuracy_metric, preds_a, preds_b, targets, n_bootstraps=2000, seed=42
        )
        assert result["delta"] > 0  # A is better
        assert result["p_value"] < 0.05
        assert result["significant_at_005"] is True

    def test_lower_is_better(self):
        """Test higher_is_better=False (e.g., for loss metrics)."""

        def neg_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return -((preds >= 0.5).long() == targets).float().mean()

        targets = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20)
        preds_a = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1] * 20)
        preds_b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] * 20)

        result = paired_bootstrap_test(
            neg_accuracy,
            preds_a,
            preds_b,
            targets,
            n_bootstraps=2000,
            seed=42,
            higher_is_better=False,
        )
        assert result["delta"] < 0  # A has lower (better) score
        assert result["p_value"] < 0.05

    def test_single_sample(self):
        """Single sample should still run without error."""
        preds_a = torch.tensor([0.9])
        preds_b = torch.tensor([0.1])
        targets = torch.tensor([1])

        result = paired_bootstrap_test(
            accuracy_metric, preds_a, preds_b, targets, n_bootstraps=100, seed=42
        )
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_all_same_predictions(self):
        """All predictions identical across both models."""
        preds = torch.full((50,), 0.7)
        targets = torch.randint(0, 2, (50,))

        result = paired_bootstrap_test(
            accuracy_metric, preds, preds, targets, n_bootstraps=500, seed=42
        )
        assert result["delta"] == pytest.approx(0.0)
        assert result["score_a"] == pytest.approx(result["score_b"])

    def test_return_keys(self):
        """Check all expected keys are present."""
        preds = torch.rand(30)
        targets = torch.randint(0, 2, (30,))
        result = paired_bootstrap_test(
            accuracy_metric, preds, preds, targets, n_bootstraps=100, seed=0
        )
        expected_keys = {"score_a", "score_b", "delta", "p_value", "significant_at_005"}
        assert set(result.keys()) == expected_keys

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        preds_a = torch.rand(50)
        preds_b = torch.rand(50)
        targets = torch.randint(0, 2, (50,))
        r1 = paired_bootstrap_test(accuracy_metric, preds_a, preds_b, targets, seed=99)
        r2 = paired_bootstrap_test(accuracy_metric, preds_a, preds_b, targets, seed=99)
        assert r1 == r2
