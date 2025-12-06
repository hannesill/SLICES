"""Tests for evaluation metrics module.

Tests cover:
- MetricConfig creation and validation
- Metric building for different task types
- Default metric selection
- Integration with torchmetrics
- Error handling
"""

import pytest
import torch
from torchmetrics import MetricCollection

from slices.eval.metrics import (
    AVAILABLE_METRICS,
    DEFAULT_METRICS,
    MetricConfig,
    _build_metric,
    build_metrics,
    get_default_metrics,
)


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""
    
    def test_default_binary_config(self):
        """Test default binary classification config."""
        config = MetricConfig(task_type="binary")
        
        assert config.task_type == "binary"
        assert config.n_classes == 2
        assert config.metrics == DEFAULT_METRICS["binary"]
        assert config.threshold == 0.5
    
    def test_custom_metrics(self):
        """Test config with custom metrics."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "accuracy"],
        )
        
        assert config.metrics == ["auroc", "accuracy"]
    
    def test_multiclass_config(self):
        """Test multiclass config."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=5,
            metrics=["auroc", "accuracy"],
        )
        
        assert config.task_type == "multiclass"
        assert config.n_classes == 5
        assert config.metrics == ["auroc", "accuracy"]
    
    def test_multilabel_config(self):
        """Test multilabel config."""
        config = MetricConfig(
            task_type="multilabel",
            n_classes=10,
        )
        
        assert config.task_type == "multilabel"
        assert config.n_classes == 10
        assert config.metrics == DEFAULT_METRICS["multilabel"]
    
    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError, match="Unknown task_type"):
            MetricConfig(task_type="invalid")
    
    def test_invalid_metric_for_task(self):
        """Test that invalid metric for task type raises error."""
        with pytest.raises(ValueError, match="not available for task_type"):
            MetricConfig(
                task_type="binary",
                metrics=["invalid_metric"],
            )
    
    def test_all_available_metrics_valid(self):
        """Test that all available metrics are valid for each task."""
        for task_type, metrics in AVAILABLE_METRICS.items():
            if metrics:  # Skip regression (empty list)
                config = MetricConfig(
                    task_type=task_type,
                    n_classes=2 if task_type == "binary" else 5,
                    metrics=metrics,
                )
                assert config.metrics == metrics


class TestBuildMetric:
    """Tests for _build_metric function."""
    
    def test_build_auroc_binary(self):
        """Test building binary AUROC."""
        metric = _build_metric("auroc", "binary", 2)
        # torchmetrics returns BinaryAUROC, not base AUROC
        assert hasattr(metric, "update")
        assert hasattr(metric, "compute")
    
    def test_build_auprc_multiclass(self):
        """Test building multiclass AUPRC."""
        metric = _build_metric("auprc", "multiclass", 5)
        # torchmetrics returns MulticlassAveragePrecision
        assert hasattr(metric, "update")
        assert hasattr(metric, "compute")
    
    def test_build_accuracy_multilabel(self):
        """Test building multilabel accuracy."""
        metric = _build_metric("accuracy", "multilabel", 10)
        # torchmetrics returns MultilabelAccuracy
        assert hasattr(metric, "update")
        assert hasattr(metric, "compute")
    
    def test_build_f1_binary(self):
        """Test building binary F1."""
        metric = _build_metric("f1", "binary", 2)
        # torchmetrics returns BinaryF1Score
        assert hasattr(metric, "update")
        assert hasattr(metric, "compute")
    
    def test_invalid_metric_name(self):
        """Test that invalid metric name raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            _build_metric("invalid", "binary", 2)
    
    def test_invalid_task_type_for_metric(self):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError, match="Unsupported task_type"):
            _build_metric("auroc", "regression", 1)


class TestBuildMetrics:
    """Tests for build_metrics function."""
    
    def test_build_binary_metrics(self):
        """Test building binary classification metrics."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc"],
        )
        metrics = build_metrics(config)
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert len(metrics) == 2
    
    def test_build_with_prefix(self):
        """Test building metrics with prefix."""
        config = MetricConfig(task_type="binary")
        metrics = build_metrics(config, prefix="val")
        
        # Check prefix is applied
        assert metrics.prefix == "val/"
    
    def test_build_multiclass_metrics(self):
        """Test building multiclass metrics."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=5,
            metrics=["auroc", "accuracy"],
        )
        metrics = build_metrics(config, prefix="test")
        
        assert len(metrics) == 2
        assert "auroc" in metrics
        assert "accuracy" in metrics
    
    def test_build_multilabel_metrics(self):
        """Test building multilabel metrics."""
        config = MetricConfig(
            task_type="multilabel",
            n_classes=10,
            metrics=["auroc", "f1"],
        )
        metrics = build_metrics(config)
        
        assert len(metrics) == 2
    
    def test_regression_returns_empty_collection(self):
        """Test that regression returns empty MetricCollection."""
        config = MetricConfig(task_type="regression")
        metrics = build_metrics(config)
        
        assert isinstance(metrics, MetricCollection)
        assert len(metrics) == 0
    
    def test_empty_prefix(self):
        """Test building metrics with empty prefix."""
        config = MetricConfig(task_type="binary")
        metrics = build_metrics(config, prefix="")
        
        assert metrics.prefix == ""


class TestGetDefaultMetrics:
    """Tests for get_default_metrics function."""
    
    def test_binary_defaults(self):
        """Test binary classification defaults."""
        metrics = get_default_metrics("binary")
        
        assert isinstance(metrics, MetricCollection)
        # Check default metrics are present
        assert len(metrics) == len(DEFAULT_METRICS["binary"])
    
    def test_multiclass_defaults(self):
        """Test multiclass classification defaults."""
        metrics = get_default_metrics("multiclass", n_classes=5)
        
        assert len(metrics) == len(DEFAULT_METRICS["multiclass"])
    
    def test_multilabel_defaults(self):
        """Test multilabel classification defaults."""
        metrics = get_default_metrics("multilabel", n_classes=10)
        
        assert len(metrics) == len(DEFAULT_METRICS["multilabel"])
    
    def test_with_prefix(self):
        """Test defaults with prefix."""
        metrics = get_default_metrics("binary", prefix="val")
        
        assert metrics.prefix == "val/"


class TestMetricIntegration:
    """Integration tests with actual metric computation."""
    
    def test_binary_metric_computation(self):
        """Test computing binary metrics with dummy data."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc", "accuracy"],
        )
        metrics = build_metrics(config)
        
        # Create dummy predictions and labels
        probs = torch.tensor([0.1, 0.4, 0.6, 0.9])
        labels = torch.tensor([0, 0, 1, 1])
        
        # Update and compute
        metrics.update(probs, labels)
        results = metrics.compute()
        
        # Check results are computed
        assert "auroc" in results
        assert "auprc" in results
        assert "accuracy" in results
        
        # Check values are reasonable
        assert 0.0 <= results["auroc"] <= 1.0
        assert 0.0 <= results["auprc"] <= 1.0
    
    def test_multiclass_metric_computation(self):
        """Test computing multiclass metrics."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=3,
            metrics=["auroc", "accuracy"],
        )
        metrics = build_metrics(config, prefix="test")
        
        # Create dummy data (batch_size=10, n_classes=3)
        probs = torch.softmax(torch.randn(10, 3), dim=1)
        labels = torch.randint(0, 3, (10,))
        
        metrics.update(probs, labels)
        results = metrics.compute()
        
        assert "test/auroc" in results
        assert "test/accuracy" in results
    
    def test_metric_reset(self):
        """Test that metrics can be reset."""
        config = MetricConfig(task_type="binary", metrics=["auroc"])
        metrics = build_metrics(config)
        
        # Update with data
        probs = torch.tensor([0.1, 0.9])
        labels = torch.tensor([0, 1])
        metrics.update(probs, labels)
        
        # Compute and reset
        result1 = metrics.compute()
        metrics.reset()
        
        # Update with different data
        probs2 = torch.tensor([0.2, 0.8])
        labels2 = torch.tensor([1, 0])
        metrics.update(probs2, labels2)
        result2 = metrics.compute()
        
        # Results should be different after reset
        assert result1["auroc"] != result2["auroc"]
    
    def test_batch_accumulation(self):
        """Test that metrics accumulate across batches."""
        config = MetricConfig(task_type="binary", metrics=["auroc"])
        metrics = build_metrics(config)
        
        # Update with multiple batches
        for _ in range(3):
            probs = torch.rand(16)
            labels = torch.randint(0, 2, (16,))
            metrics.update(probs, labels)
        
        # Should compute over all batches
        results = metrics.compute()
        assert "auroc" in results


class TestConstants:
    """Tests for module constants."""
    
    def test_available_metrics_structure(self):
        """Test AVAILABLE_METRICS has correct structure."""
        assert "binary" in AVAILABLE_METRICS
        assert "multiclass" in AVAILABLE_METRICS
        assert "multilabel" in AVAILABLE_METRICS
        assert "regression" in AVAILABLE_METRICS
        
        # Check metrics are lists
        for task_type, metrics in AVAILABLE_METRICS.items():
            assert isinstance(metrics, list)
    
    def test_default_metrics_structure(self):
        """Test DEFAULT_METRICS has correct structure."""
        assert "binary" in DEFAULT_METRICS
        assert "multiclass" in DEFAULT_METRICS
        assert "multilabel" in DEFAULT_METRICS
        assert "regression" in DEFAULT_METRICS
        
        # Defaults should be subset of available
        for task_type, defaults in DEFAULT_METRICS.items():
            available = AVAILABLE_METRICS[task_type]
            for metric in defaults:
                assert metric in available
    
    def test_binary_defaults_minimal(self):
        """Test binary defaults are minimal."""
        defaults = DEFAULT_METRICS["binary"]
        assert len(defaults) <= 3  # Keep it minimal
        assert "auroc" in defaults  # Core clinical metric
    
    def test_all_task_types_covered(self):
        """Test all task types have entries."""
        task_types = ["binary", "multiclass", "multilabel", "regression"]
        for task_type in task_types:
            assert task_type in AVAILABLE_METRICS
            assert task_type in DEFAULT_METRICS


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_empty_metrics_list(self):
        """Test handling of empty metrics list."""
        config = MetricConfig(task_type="binary", metrics=[])
        metrics = build_metrics(config)
        
        # Should create empty collection
        assert len(metrics) == 0
    
    def test_duplicate_metrics(self):
        """Test that duplicate metrics don't cause issues."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auroc"],  # Duplicate
        )
        # Should not raise error during config creation
        assert config.metrics == ["auroc", "auroc"]
        
        # Building might deduplicate (torchmetrics behavior)
        metrics = build_metrics(config)
        assert "auroc" in metrics
    
    def test_none_metrics_uses_defaults(self):
        """Test that None metrics uses defaults."""
        config = MetricConfig(task_type="binary", metrics=None)
        
        assert config.metrics == DEFAULT_METRICS["binary"]
    
    def test_invalid_n_classes(self):
        """Test invalid n_classes values."""
        # n_classes < 2 for classification might cause issues in torchmetrics
        # but we don't validate it at config level (let torchmetrics handle it)
        config = MetricConfig(task_type="binary", n_classes=1)
        assert config.n_classes == 1  # Config accepts it
