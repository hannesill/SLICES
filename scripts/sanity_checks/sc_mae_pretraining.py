"""Overfitting sanity check for MAE pretraining pipeline.

Loads a small subset of real data and trains MAE until reconstruction loss approaches zero.
Uses the SAME data for both training and validation to verify the pipeline works correctly.

Key checks:
1. Train masked loss should approach 0 (directly optimized)
2. Train visible loss should approach 0 (side effect of good representations)
3. Val masked loss should approach 0 (same data, model memorized it)
4. Val visible loss should approach 0 (same data, model memorized it)

If any of these fail to converge, something is broken in the pipeline.

Prerequisites:
    Run prepare_dataset.py first to generate normalization_stats.yaml

Usage:
    uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv-demo
    uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from slices.models.encoders import build_encoder
from slices.models.pretraining import build_ssl_objective
from slices.models.pretraining.mae import MAEConfig
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


@dataclass
class SanityCheckResult:
    """Results from sanity check."""

    passed: bool
    final_train_masked: float
    final_train_visible: float
    final_val_masked: float
    final_val_visible: float
    epochs_to_converge: int


def load_subset(
    processed_dir: Path,
    n_samples: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Load a small subset using prepared normalization stats.

    Args:
        processed_dir: Path to processed data directory.
        n_samples: Number of samples to load.

    Returns:
        Tuple of (timeseries, masks, n_features, seq_length).
    """
    print(f"Loading {n_samples} samples from {processed_dir}...")

    # Load metadata
    with open(processed_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    feature_names = metadata["feature_names"]
    seq_length = metadata["seq_length_hours"]
    n_features = len(feature_names)

    # Load normalization stats if available
    stats_path = processed_dir / "normalization_stats.yaml"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = yaml.safe_load(f)
        means = torch.tensor(stats["feature_means"], dtype=torch.float32)
        stds = torch.tensor(stats["feature_stds"], dtype=torch.float32)
        # Prevent division by zero
        stds = torch.clamp(stds, min=1e-6)
        print("  Using prepared normalization stats")
    else:
        means = torch.zeros(n_features)
        stds = torch.ones(n_features)
        print("  Warning: No normalization stats found, using raw values")

    # Load data
    ts_df = pl.read_parquet(processed_dir / "timeseries.parquet").head(n_samples)

    # Convert to tensors
    timeseries_list = []
    mask_list = []

    for row in ts_df.iter_rows(named=True):
        ts_data = row["timeseries"]
        mask_data = row["mask"]

        ts_tensor = torch.zeros(seq_length, n_features)
        mask_tensor = torch.zeros(seq_length, n_features, dtype=torch.bool)

        for t in range(min(len(ts_data), seq_length)):
            for f in range(n_features):
                val = ts_data[t][f]
                if val is not None and not np.isnan(val):
                    ts_tensor[t, f] = val
                mask_tensor[t, f] = bool(mask_data[t][f])

        # Zero imputation then normalize
        ts_tensor = torch.nan_to_num(ts_tensor, nan=0.0)
        ts_tensor = (ts_tensor - means) / stds

        timeseries_list.append(ts_tensor)
        mask_list.append(mask_tensor)

    timeseries = torch.stack(timeseries_list)
    masks = torch.stack(mask_list)

    # Compute observation statistics
    obs_ratio = masks.float().mean().item()
    print(f"  Shape: {timeseries.shape}")
    print(f"  Observation ratio: {obs_ratio:.1%}")

    return timeseries, masks, n_features, seq_length


def evaluate_mae(
    mae: nn.Module,
    timeseries: torch.Tensor,
    masks: torch.Tensor,
    use_random_mask: bool = True,
) -> Dict[str, float]:
    """Evaluate MAE on data.

    Args:
        mae: MAE objective module.
        timeseries: Input tensor (B, T, D).
        masks: Observation mask (B, T, D).
        use_random_mask: If True, use random SSL masks. If False, use deterministic.

    Returns:
        Dictionary of metrics.
    """
    mae.eval()
    with torch.no_grad():
        # Temporarily set training mode for mask generation if needed
        original_training = mae.training
        if use_random_mask:
            mae.train()  # Random masks

        loss, metrics = mae(timeseries, masks)

        mae.train(original_training)

    return {
        "loss": loss.item(),
        "masked": metrics["mae_recon_loss_masked"].item(),
        "visible": metrics["mae_recon_loss_visible"].item(),
        "mask_ratio": metrics["mae_mask_ratio_actual"],
    }


def run_sanity_check(
    processed_dir: str,
    n_samples: int = 64,
    max_epochs: int = 500,
    target_loss: float = 0.01,
    log_every: int = 50,
    mask_ratio: float = 0.15,
    mask_strategy: str = "random",
) -> SanityCheckResult:
    """Run MAE overfitting sanity check.

    Args:
        processed_dir: Path to processed data directory.
        n_samples: Number of samples to use.
        max_epochs: Maximum training epochs.
        target_loss: Target loss to achieve (for all metrics).
        log_every: Log every N epochs.
        mask_ratio: Fraction of input to mask.
        mask_strategy: Masking strategy (random, block, timestep, feature).

    Returns:
        SanityCheckResult with pass/fail and final metrics.
    """
    print("=" * 70)
    print("MAE Pretraining Sanity Check")
    print("=" * 70)
    print()
    print("This test verifies the MAE pipeline can overfit on a small batch.")
    print("All four metrics (train/val masked/visible) should approach zero.")
    print()

    L.seed_everything(SEED, workers=True)

    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist")
        return SanityCheckResult(
            passed=False,
            final_train_masked=float("inf"),
            final_train_visible=float("inf"),
            final_val_masked=float("inf"),
            final_val_visible=float("inf"),
            epochs_to_converge=-1,
        )

    # Load small subset
    timeseries, masks, n_features, seq_length = load_subset(processed_dir, n_samples)

    # Create dataloader (same data for train and val!)
    dataset = TensorDataset(timeseries, masks)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

    # Build encoder with NO dropout (for overfitting)
    d_model = 64
    encoder = build_encoder(
        "transformer",
        {
            "d_input": n_features,
            "d_model": d_model,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": d_model * 2,
            "dropout": 0.0,  # Critical: no dropout for overfitting
            "max_seq_length": seq_length,
            "pooling": "none",  # MAE requires per-timestep output
        },
    )

    # Build MAE objective with NO dropout
    mae_config = MAEConfig(
        name="mae",
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        min_block_size=3,
        max_block_size=10,
        mask_value=0.0,
        decoder_d_model=32,
        decoder_n_layers=1,
        decoder_n_heads=4,
        decoder_d_ff=64,
        decoder_dropout=0.0,  # Critical: no dropout for overfitting
        loss_on_observed_only=True,
        norm_target=False,
    )
    mae = build_ssl_objective(encoder, mae_config)

    # Count parameters
    total_params = sum(p.numel() for p in mae.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Mask ratio: {mask_ratio:.0%}, strategy: {mask_strategy}")

    optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

    print(f"\nTraining on {n_samples} samples for up to {max_epochs} epochs...")
    print(f"Target: all losses < {target_loss}")
    print()
    print(
        f"{'Epoch':>6} | {'Train Masked':>12} | {'Train Visible':>13} | "
        f"{'Val Masked':>10} | {'Val Visible':>11}"
    )
    print("-" * 70)

    # Get single batch (we use all data)
    x, obs_mask = next(iter(loader))

    # Training loop
    converged_epoch = -1

    for epoch in range(max_epochs):
        # Training step (random masks)
        mae.train()
        optimizer.zero_grad()
        loss, train_metrics = mae(x, obs_mask)
        loss.backward()
        optimizer.step()

        train_masked = train_metrics["mae_recon_loss_masked"].item()
        train_visible = train_metrics["mae_recon_loss_visible"].item()

        # Validation step (same data, but eval mode with deterministic masks)
        mae.eval()
        with torch.no_grad():
            _, val_metrics = mae(x, obs_mask)

        val_masked = val_metrics["mae_recon_loss_masked"].item()
        val_visible = val_metrics["mae_recon_loss_visible"].item()

        # Log progress
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"{epoch+1:>6} | {train_masked:>12.6f} | {train_visible:>13.6f} | "
                f"{val_masked:>10.6f} | {val_visible:>11.6f}"
            )

        # Check convergence (all metrics below target)
        all_below_target = (
            train_masked < target_loss
            and train_visible < target_loss
            and val_masked < target_loss
            and val_visible < target_loss
        )

        if all_below_target and converged_epoch < 0:
            converged_epoch = epoch + 1
            # Continue training to verify stability
            if epoch >= max_epochs // 2:
                break

    # Final evaluation
    mae.eval()
    with torch.no_grad():
        _, final_train = mae(x, obs_mask)
        mae.train()
        _, final_train_random = mae(x, obs_mask)

    final_train_masked = final_train_random["mae_recon_loss_masked"].item()
    final_train_visible = final_train_random["mae_recon_loss_visible"].item()

    mae.eval()
    with torch.no_grad():
        _, final_val = mae(x, obs_mask)
    final_val_masked = final_val["mae_recon_loss_masked"].item()
    final_val_visible = final_val["mae_recon_loss_visible"].item()

    # Print final results
    print("-" * 70)
    print(
        f"{'Final':>6} | {final_train_masked:>12.6f} | {final_train_visible:>13.6f} | "
        f"{final_val_masked:>10.6f} | {final_val_visible:>11.6f}"
    )
    print()

    # Determine pass/fail
    all_converged = (
        final_train_masked < target_loss
        and final_train_visible < target_loss
        and final_val_masked < target_loss
        and final_val_visible < target_loss
    )

    if all_converged:
        print(f"[PASS] All metrics converged below {target_loss}")
        if converged_epoch > 0:
            print(f"  Converged at epoch {converged_epoch}")
    else:
        print(f"[FAIL] Not all metrics converged after {max_epochs} epochs")
        print()
        print("Diagnosis:")
        if final_train_masked >= target_loss:
            print(f"  - Train masked loss too high: {final_train_masked:.6f}")
            print("    This is the primary optimization target - something is very wrong")
        if final_train_visible >= target_loss:
            print(f"  - Train visible loss too high: {final_train_visible:.6f}")
            print("    The encoder may not be learning good representations")
        if final_val_masked >= target_loss:
            print(f"  - Val masked loss too high: {final_val_masked:.6f}")
            print("    Since train/val use same data, this suggests mask handling issues")
        if final_val_visible >= target_loss:
            print(f"  - Val visible loss too high: {final_val_visible:.6f}")
            print("    Since train/val use same data, this suggests eval mode issues")

    return SanityCheckResult(
        passed=all_converged,
        final_train_masked=final_train_masked,
        final_train_visible=final_train_visible,
        final_val_masked=final_val_masked,
        final_val_visible=final_val_visible,
        epochs_to_converge=converged_epoch,
    )


def run_all_strategies(processed_dir: str) -> bool:
    """Run sanity check for all masking strategies.

    Args:
        processed_dir: Path to processed data directory.

    Returns:
        True if all strategies pass.
    """
    strategies = ["random", "block", "timestep", "feature"]
    results = {}

    for strategy in strategies:
        print()
        print("#" * 70)
        print(f"# Testing masking strategy: {strategy}")
        print("#" * 70)
        print()

        result = run_sanity_check(
            processed_dir,
            mask_strategy=strategy,
            max_epochs=300,
            log_every=50,
        )
        results[strategy] = result

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Strategy':<12} | {'Pass':<6} | {'Train Masked':>12} | {'Val Masked':>10}")
    print("-" * 50)

    all_passed = True
    for strategy, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{strategy:<12} | {status:<6} | {result.final_train_masked:>12.6f} | "
            f"{result.final_val_masked:>10.6f}"
        )
        if not result.passed:
            all_passed = False

    print()
    if all_passed:
        print("[PASS] All masking strategies work correctly")
    else:
        print("[FAIL] Some masking strategies failed")

    return all_passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python scripts/sanity_checks/sc_mae_pretraining.py <processed_dir>")
        print("  uv run python scripts/sanity_checks/sc_mae_pretraining.py <processed_dir> --all")
        print()
        print("Examples:")
        print(
            "  uv run python scripts/sanity_checks/sc_mae_pretraining.py "
            "data/processed/mimic-iv-demo"
        )
        print(
            "  uv run python scripts/sanity_checks/sc_mae_pretraining.py "
            "data/processed/mimic-iv --all"
        )
        sys.exit(1)

    processed_dir = sys.argv[1]
    run_all = "--all" in sys.argv

    if run_all:
        success = run_all_strategies(processed_dir)
    else:
        result = run_sanity_check(processed_dir)
        success = result.passed

    sys.exit(0 if success else 1)
