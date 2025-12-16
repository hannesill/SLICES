"""Example: Using the eval module for configurable metrics.

This example demonstrates:
1. Creating metric configurations
2. Building metrics for different task types
3. Using metrics in a training-like loop
4. Computing and displaying results

Run:
    uv run python examples/eval_metrics_example.py
"""

import torch
from slices.eval import MetricConfig, build_metrics, get_default_metrics


def example_binary_classification():
    """Example: Binary classification with custom metrics."""
    print("=" * 60)
    print("Example 1: Binary Classification (Mortality Prediction)")
    print("=" * 60)

    # Create metric configuration
    config = MetricConfig(
        task_type="binary",
        metrics=["auroc", "auprc", "accuracy", "f1"],
    )

    # Build metrics for validation
    val_metrics = build_metrics(config, prefix="val")
    print(f"\nConfigured metrics: {config.metrics}")
    print(f"MetricCollection: {val_metrics}\n")

    # Simulate predictions and labels
    torch.manual_seed(42)
    _n_samples = 100  # noqa: F841 - for documentation

    # Generate realistic mortality prediction data
    # Positive class (mortality) has higher predicted probabilities
    probs_positive = torch.clamp(torch.randn(20) * 0.2 + 0.7, 0, 1)  # ~20% mortality
    probs_negative = torch.clamp(torch.randn(80) * 0.2 + 0.3, 0, 1)  # ~80% survival

    probs = torch.cat([probs_positive, probs_negative])
    labels = torch.cat([torch.ones(20), torch.zeros(80)]).long()

    # Update metrics
    val_metrics.update(probs, labels)

    # Compute results
    results = val_metrics.compute()

    print("Results on validation set:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    print()


def example_multiclass_classification():
    """Example: Multiclass classification with defaults."""
    print("=" * 60)
    print("Example 2: Multiclass Classification (Diagnosis Prediction)")
    print("=" * 60)

    # Use default metrics for multiclass
    test_metrics = get_default_metrics("multiclass", n_classes=5, prefix="test")
    print(f"\nDefault metrics: {list(test_metrics.keys())}\n")

    # Simulate predictions (batch_size=50, n_classes=5)
    torch.manual_seed(42)
    logits = torch.randn(50, 5)
    probs = torch.softmax(logits, dim=1)
    labels = torch.randint(0, 5, (50,))

    # Update and compute
    test_metrics.update(probs, labels)
    results = test_metrics.compute()

    print("Results on test set:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    print()


def example_batch_accumulation():
    """Example: Accumulating metrics across batches."""
    print("=" * 60)
    print("Example 3: Batch Accumulation (Training Loop)")
    print("=" * 60)

    # Create metrics
    config = MetricConfig(task_type="binary", metrics=["auroc", "auprc"])
    train_metrics = build_metrics(config, prefix="train")

    # Simulate training batches
    torch.manual_seed(42)
    n_batches = 10
    batch_size = 32

    print(f"\nProcessing {n_batches} batches of size {batch_size}...\n")

    for batch_idx in range(n_batches):
        # Simulate batch predictions
        probs = torch.rand(batch_size)
        labels = torch.randint(0, 2, (batch_size,))

        # Update metrics (accumulates across batches)
        train_metrics.update(probs, labels)

        if batch_idx % 3 == 0:
            print(f"  Processed batch {batch_idx + 1}/{n_batches}")

    # Compute final metrics
    results = train_metrics.compute()

    print(f"\nFinal metrics (accumulated over {n_batches} batches):")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    # Reset for next epoch
    train_metrics.reset()
    print("\n✓ Metrics reset for next epoch")

    print()


def example_configuration_validation():
    """Example: Configuration validation and error handling."""
    print("=" * 60)
    print("Example 4: Configuration Validation")
    print("=" * 60)

    # Valid configuration
    try:
        config = MetricConfig(
            task_type="binary",
            metrics=["auroc", "auprc"],
        )
        print("\n✓ Valid config created:", config)
    except ValueError as e:
        print(f"\n✗ Error: {e}")

    # Invalid task type
    try:
        config = MetricConfig(task_type="invalid_type")
        print("\n✓ Config created")
    except ValueError as e:
        print(f"\n✗ Invalid task type error (expected): {e}")

    # Invalid metric for task
    try:
        config = MetricConfig(
            task_type="binary",
            metrics=["nonexistent_metric"],
        )
        print("\n✓ Config created")
    except ValueError as e:
        print(f"\n✗ Invalid metric error (expected): {e}")

    print()


def example_minimal_vs_comprehensive():
    """Example: Minimal vs comprehensive metrics."""
    print("=" * 60)
    print("Example 5: Minimal vs Comprehensive Metrics")
    print("=" * 60)

    # Minimal (defaults) - faster computation
    minimal_config = MetricConfig(task_type="binary")
    print(f"\nMinimal (default): {minimal_config.metrics}")

    # Comprehensive - more detailed evaluation
    comprehensive_config = MetricConfig(
        task_type="binary",
        metrics=["auroc", "auprc", "accuracy", "f1"],
    )
    print(f"Comprehensive: {comprehensive_config.metrics}")

    # Build and compare
    torch.manual_seed(42)
    probs = torch.rand(100)
    labels = torch.randint(0, 2, (100,))

    # Minimal metrics
    minimal_metrics = build_metrics(minimal_config, prefix="val")
    minimal_metrics.update(probs, labels)
    minimal_results = minimal_metrics.compute()

    print("\nMinimal results:")
    for name, value in minimal_results.items():
        print(f"  {name}: {value:.4f}")

    # Comprehensive metrics
    comprehensive_metrics = build_metrics(comprehensive_config, prefix="val")
    comprehensive_metrics.update(probs, labels)
    comprehensive_results = comprehensive_metrics.compute()

    print("\nComprehensive results:")
    for name, value in comprehensive_results.items():
        print(f"  {name}: {value:.4f}")

    print("\n💡 Use minimal for fast iteration, comprehensive for final evaluation")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SLICES Evaluation Module Examples")
    print("=" * 60 + "\n")

    example_binary_classification()
    example_multiclass_classification()
    example_batch_accumulation()
    example_configuration_validation()
    example_minimal_vs_comprehensive()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
