"""Tests for regression evaluation metrics.

Tests cover:
- MetricConfig with regression task type
- Regression metric building (MSE, MAE, R2, RMSE)
- Default regression metric selection
- Metric computation with known inputs
- Error handling for invalid regression metrics
"""

import math

import pytest
import torch
from slices.eval.metrics import (
    AVAILABLE_METRICS,
    DEFAULT_METRICS,
    MetricConfig,
    _build_metric,
    build_metrics,
)
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


class TestRegressionMetricConfig:
    """Tests for MetricConfig with regression task type."""

    def test_default_regression_config(self):
        """Test default regression config uses correct defaults."""
        config = MetricConfig(task_type="regression")

        assert config.task_type == "regression"
        assert config.metrics == ["mse", "mae", "r2"]

    def test_available_regression_metrics(self):
        """Test AVAILABLE_METRICS has correct regression entries."""
        assert AVAILABLE_METRICS["regression"] == ["mse", "mae", "rmse", "r2"]

    def test_default_regression_metrics(self):
        """Test DEFAULT_METRICS has correct regression entries."""
        assert DEFAULT_METRICS["regression"] == ["mse", "mae", "r2"]

    def test_custom_regression_metrics(self):
        """Test config with custom regression metrics."""
        config = MetricConfig(
            task_type="regression",
            metrics=["mse", "rmse"],
        )
        assert config.metrics == ["mse", "rmse"]

    def test_invalid_metric_for_regression(self):
        """Test that classification metric raises error for regression."""
        with pytest.raises(ValueError, match="not available for task_type"):
            MetricConfig(task_type="regression", metrics=["auroc"])


class TestBuildRegressionMetric:
    """Tests for _build_metric with regression task type."""

    def test_build_mse(self):
        """Test building MSE metric."""
        metric = _build_metric("mse", "regression", 1)
        assert isinstance(metric, MeanSquaredError)

    def test_build_mae(self):
        """Test building MAE metric."""
        metric = _build_metric("mae", "regression", 1)
        assert isinstance(metric, MeanAbsoluteError)

    def test_build_r2(self):
        """Test building R2 metric."""
        metric = _build_metric("r2", "regression", 1)
        assert isinstance(metric, R2Score)

    def test_build_rmse(self):
        """Test building RMSE metric (MSE with squared=False)."""
        metric = _build_metric("rmse", "regression", 1)
        assert isinstance(metric, MeanSquaredError)

    def test_invalid_regression_metric(self):
        """Test that invalid metric name for regression raises ValueError."""
        with pytest.raises(ValueError, match="not available for task_type 'regression'"):
            _build_metric("auroc", "regression", 1)


class TestBuildRegressionMetrics:
    """Tests for build_metrics with regression config."""

    def test_build_default_regression_metrics(self):
        """Test build_metrics returns non-empty MetricCollection for regression."""
        config = MetricConfig(task_type="regression")
        metrics = build_metrics(config, prefix="val")

        assert isinstance(metrics, MetricCollection)
        assert len(metrics) == 3
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_build_all_regression_metrics(self):
        """Test building all available regression metrics."""
        config = MetricConfig(
            task_type="regression",
            metrics=["mse", "mae", "rmse", "r2"],
        )
        metrics = build_metrics(config)

        assert len(metrics) == 4

    def test_regression_metrics_with_prefix(self):
        """Test regression metrics apply prefix correctly."""
        config = MetricConfig(task_type="regression")
        metrics = build_metrics(config, prefix="test")

        assert metrics.prefix == "test/"


class TestRegressionMetricComputation:
    """Integration tests: verify metrics compute correct values."""

    def test_perfect_predictions(self):
        """Test MSE=0, R2=1 for perfect predictions."""
        config = MetricConfig(task_type="regression", metrics=["mse", "mae", "r2"])
        metrics = build_metrics(config)

        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        metrics.update(preds, targets)
        results = metrics.compute()

        assert results["mse"].item() == pytest.approx(0.0)
        assert results["mae"].item() == pytest.approx(0.0)
        assert results["r2"].item() == pytest.approx(1.0)

    def test_known_mse(self):
        """Test MSE with known values."""
        metric = _build_metric("mse", "regression", 1)

        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 2.0, 2.0])
        # MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(2.0 / 3.0)

    def test_known_mae(self):
        """Test MAE with known values."""
        metric = _build_metric("mae", "regression", 1)

        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 2.0, 2.0])
        # MAE = (|1-2| + |2-2| + |3-2|) / 3 = 2/3
        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(2.0 / 3.0)

    def test_rmse_is_sqrt_of_mse(self):
        """Test RMSE equals sqrt of MSE."""
        mse_metric = _build_metric("mse", "regression", 1)
        rmse_metric = _build_metric("rmse", "regression", 1)

        preds = torch.tensor([1.0, 2.0, 4.0, 5.0])
        targets = torch.tensor([2.0, 3.0, 3.0, 4.0])

        mse_metric.update(preds, targets)
        rmse_metric.update(preds, targets)

        mse_val = mse_metric.compute().item()
        rmse_val = rmse_metric.compute().item()

        assert rmse_val == pytest.approx(math.sqrt(mse_val))

    def test_r2_with_constant_prediction(self):
        """Test R2 with mean prediction gives R2=0."""
        metric = _build_metric("r2", "regression", 1)

        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = torch.full_like(targets, targets.mean().item())

        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_batch_accumulation(self):
        """Test regression metrics accumulate across batches."""
        config = MetricConfig(task_type="regression", metrics=["mse"])
        metrics = build_metrics(config)

        for _ in range(3):
            preds = torch.randn(16)
            targets = torch.randn(16)
            metrics.update(preds, targets)

        results = metrics.compute()
        assert "mse" in results
        assert results["mse"].item() >= 0.0

    def test_metric_reset(self):
        """Test regression metrics can be reset."""
        config = MetricConfig(task_type="regression", metrics=["mse"])
        metrics = build_metrics(config)

        preds1 = torch.tensor([1.0, 2.0, 3.0])
        targets1 = torch.tensor([1.0, 2.0, 3.0])
        metrics.update(preds1, targets1)
        result1 = metrics.compute()

        metrics.reset()

        preds2 = torch.tensor([1.0, 2.0, 3.0])
        targets2 = torch.tensor([4.0, 5.0, 6.0])
        metrics.update(preds2, targets2)
        result2 = metrics.compute()

        assert result1["mse"].item() != result2["mse"].item()
