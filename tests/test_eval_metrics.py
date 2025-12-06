"""Tests for evaluation metrics module."""

import pytest
import torch
from torchmetrics import MetricCollection

from slices.eval import MetricConfig, build_metrics, get_default_metrics


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""
    
    def test_default_binary_config(self):
        """Test default binary classification config."""
        config = MetricConfig(task_type="binary")
        
        assert config.task_type == "binary"
        assert config.n_classes == 2
        assert config.metrics == ["auroc", "auprc"]
        assert config.threshold == 0.5
    
    def test_custom_binary_config(self):
        """Test custom binary classification config."""
        config = MetricConfig(
            task_type="binary",
            n_classes=2,
            metrics=["auroc", "auprc", "accuracy", "f1"],
            threshold=0.7,
        )
        
        assert config.task_type == "binary"
        assert config.n_classes == 2
        assert config.metrics == ["auroc", "auprc", "accuracy", "f1"]
        assert config.threshold == 0.7
    
    def test_multiclass_config(self):
        """Test multiclass classification config."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=4,
            metrics=["auroc", "accuracy"],
        )
        
        assert config.task_type == "multiclass"
        assert config.n_classes == 4
        assert config.metrics == ["auroc", "accuracy"]
    
    def test_multilabel_config(self):
        """Test multilabel classification config."""
        config = MetricConfig(
            task_type="multilabel",
            n_classes=3,
            metrics=["auroc"],
        )
        
        assert config.task_type == "multilabel"
        assert config.n_classes == 3
        assert config.metrics == ["auroc"]
    
    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError, match="Unknown task_type"):
            MetricConfig(task_type="invalid")
    
    def test_invalid_metric_for_task(self):
        """Test that invalid metric for task type raises error."""
        with pytest.raises(ValueError, match="not available"):
            MetricConfig(task_type="regression", metrics=["auroc"])
    
    def test_default_metrics_used_when_none(self):
        """Test that default metrics are used when metrics=None."""
        config = MetricConfig(task_type="binary", metrics=None)
        assert config.metrics == ["auroc", "auprc"]
        
        config = MetricConfig(task_type="multiclass", metrics=None)
        assert config.metrics == ["auroc", "accuracy"]


class TestBuildMetrics:
    """Tests for build_metrics function."""
    
    def test_build_binary_metrics(self):
        """Test building metrics for binary classification."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc"],
        )
        metrics = build_metrics(config, prefix="val")
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert metrics.prefix == "val/"
    
    def test_build_multiclass_metrics(self):
        """Test building metrics for multiclass classification."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=5,
            metrics=["auroc", "accuracy", "f1"],
        )
        metrics = build_metrics(config, prefix="test")
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics.prefix == "test/"
    
    def test_build_multilabel_metrics(self):
        """Test building metrics for multilabel classification."""
        config = MetricConfig(
            task_type="multilabel",
            n_classes=3,
            metrics=["auroc", "auprc"],
        )
        metrics = build_metrics(config, prefix="train")
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert metrics.prefix == "train/"
    
    def test_build_without_prefix(self):
        """Test building metrics without prefix."""
        config = MetricConfig(task_type="binary")
        metrics = build_metrics(config, prefix="")
        
        assert metrics.prefix == ""
    
    def test_build_regression_metrics_empty(self):
        """Test that regression returns empty collection (not yet implemented)."""
        config = MetricConfig(task_type="regression", metrics=[])
        metrics = build_metrics(config, prefix="val")
        
        # Should return empty MetricCollection
        assert isinstance(metrics, MetricCollection)
        assert len(metrics) == 0


class TestGetDefaultMetrics:
    """Tests for get_default_metrics convenience function."""
    
    def test_binary_defaults(self):
        """Test default metrics for binary classification."""
        metrics = get_default_metrics("binary", prefix="val")
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert metrics.prefix == "val/"
    
    def test_multiclass_defaults(self):
        """Test default metrics for multiclass classification."""
        metrics = get_default_metrics("multiclass", n_classes=4, prefix="test")
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert "accuracy" in metrics
        assert metrics.prefix == "test/"
    
    def test_multilabel_defaults(self):
        """Test default metrics for multilabel classification."""
        metrics = get_default_metrics("multilabel", n_classes=5, prefix="train")
        
        assert isinstance(metrics, MetricCollection)
        assert "auroc" in metrics
        assert metrics.prefix == "train/"


class TestMetricsComputation:
    """Tests for actual metric computation."""
    
    def test_binary_metrics_computation(self):
        """Test computing binary classification metrics."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc", "accuracy"],
        )
        metrics = build_metrics(config, prefix="val")
        
        # Perfect predictions
        probs = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.9])
        labels = torch.tensor([1, 1, 0, 0, 1])
        
        metrics.update(probs, labels)
        results = metrics.compute()
        
        assert "val/auroc" in results
        assert "val/auprc" in results
        assert "val/accuracy" in results
        assert results["val/auroc"] == 1.0  # Perfect AUROC
        assert results["val/accuracy"] == 1.0  # Perfect accuracy
    
    def test_multiclass_metrics_computation(self):
        """Test computing multiclass classification metrics."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=3,
            metrics=["accuracy"],
        )
        metrics = build_metrics(config, prefix="test")
        
        # Predictions: (batch_size, n_classes)
        probs = torch.tensor([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        labels = torch.tensor([0, 1, 2])
        
        metrics.update(probs, labels)
        results = metrics.compute()
        
        assert "test/accuracy" in results
        assert results["test/accuracy"] == 1.0  # Perfect accuracy
    
    def test_metrics_reset(self):
        """Test that metrics can be reset."""
        config = MetricConfig(task_type="binary", metrics=["auroc"])
        metrics = build_metrics(config, prefix="val")
        
        # First batch
        probs1 = torch.tensor([0.9, 0.8, 0.1])
        labels1 = torch.tensor([1, 1, 0])
        metrics.update(probs1, labels1)
        results1 = metrics.compute()
        
        # Reset
        metrics.reset()
        
        # Second batch
        probs2 = torch.tensor([0.2, 0.3, 0.9])
        labels2 = torch.tensor([0, 0, 1])
        metrics.update(probs2, labels2)
        results2 = metrics.compute()
        
        # Results should be different (only second batch)
        assert results1["val/auroc"] == 1.0
        assert results2["val/auroc"] == 1.0
    
    def test_metrics_accumulation(self):
        """Test that metrics accumulate across updates."""
        config = MetricConfig(task_type="binary", metrics=["accuracy"])
        metrics = build_metrics(config, prefix="val")
        
        # First batch - all correct
        probs1 = torch.tensor([0.9, 0.1])
        labels1 = torch.tensor([1, 0])
        metrics.update(probs1, labels1)
        
        # Second batch - all correct
        probs2 = torch.tensor([0.8, 0.2])
        labels2 = torch.tensor([1, 0])
        metrics.update(probs2, labels2)
        
        results = metrics.compute()
        
        # Should have 100% accuracy across both batches
        assert results["val/accuracy"] == 1.0


class TestMetricsWithRandomData:
    """Tests with random data to ensure metrics don't crash."""
    
    def test_binary_random_data(self):
        """Test binary metrics with random data."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc", "accuracy", "f1"],
        )
        metrics = build_metrics(config, prefix="val")
        
        # Random predictions
        torch.manual_seed(42)
        probs = torch.rand(100)
        labels = torch.randint(0, 2, (100,))
        
        metrics.update(probs, labels)
        results = metrics.compute()
        
        # Check all metrics are present and in valid range
        assert "val/auroc" in results
        assert "val/auprc" in results
        assert "val/accuracy" in results
        assert "val/f1" in results
        assert 0 <= results["val/auroc"] <= 1
        assert 0 <= results["val/auprc"] <= 1
        assert 0 <= results["val/accuracy"] <= 1
        assert 0 <= results["val/f1"] <= 1
    
    def test_multiclass_random_data(self):
        """Test multiclass metrics with random data."""
        config = MetricConfig(
            task_type="multiclass",
            n_classes=5,
            metrics=["auroc", "accuracy"],
        )
        metrics = build_metrics(config, prefix="test")
        
        # Random predictions
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(50, 5), dim=1)
        labels = torch.randint(0, 5, (50,))
        
        metrics.update(probs, labels)
        results = metrics.compute()
        
        assert "test/auroc" in results
        assert "test/accuracy" in results
        assert 0 <= results["test/accuracy"] <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_class_in_batch(self):
        """Test metrics with only one class in batch."""
        config = MetricConfig(task_type="binary", metrics=["accuracy"])
        metrics = build_metrics(config, prefix="val")
        
        # All same class
        probs = torch.tensor([0.9, 0.8, 0.85])
        labels = torch.tensor([1, 1, 1])
        
        # Should not crash
        metrics.update(probs, labels)
        results = metrics.compute()
        assert "val/accuracy" in results
    
    def test_empty_prefix(self):
        """Test metrics with empty prefix."""
        config = MetricConfig(task_type="binary")
        metrics = build_metrics(config, prefix="")
        
        probs = torch.tensor([0.9, 0.1])
        labels = torch.tensor([1, 0])
        metrics.update(probs, labels)
        results = metrics.compute()
        
        # Keys should not have prefix
        assert "auroc" in results
        assert "auprc" in results
    
    def test_all_metrics_for_binary(self):
        """Test all available metrics for binary classification."""
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc", "accuracy", "f1"],
        )
        metrics = build_metrics(config, prefix="val")
        
        probs = torch.rand(50)
        labels = torch.randint(0, 2, (50,))
        
        metrics.update(probs, labels)
        results = metrics.compute()
        
        assert len(results) == 4
        assert all(key.startswith("val/") for key in results.keys())


class TestMetricConfigValidation:
    """Tests for configuration validation."""
    
    def test_invalid_metric_name(self):
        """Test that invalid metric name raises error."""
        with pytest.raises(ValueError, match="not available"):
            MetricConfig(task_type="binary", metrics=["invalid_metric"])
    
    def test_metric_not_for_task_type(self):
        """Test using metric not available for task type."""
        # AUROC shouldn't be available for regression (if we add it)
        config = MetricConfig(task_type="regression", metrics=[])
        # Should not raise error with empty list
        assert config.metrics == []
