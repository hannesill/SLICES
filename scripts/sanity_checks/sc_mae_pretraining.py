"""Overfitting sanity check for MAE pretraining pipeline.

Verifies the MAE pipeline can overfit on a small batch of real data.
Uses the SAME data for both training and validation to isolate pipeline issues
from generalization concerns.

Two modes are available:
1. Fixed mask (--fixed-mask): Same mask positions every step. Masked loss should
   approach zero. This tests if the architecture CAN overfit.
2. Random masks (default): Different mask positions each step. Loss won't reach
   zero because model must learn generalizable reconstruction, not memorization.

Prerequisites:
    Run prepare_dataset.py first to generate normalization_stats.yaml

Usage:
    # Default: random masks (generalization test)
    uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv

    # Fixed mask (true overfitting test)
    uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv --fixed-mask

    # Test encoder-decoder without SSL masking
    uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv --autoencoder
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from slices.data.transforms import create_ssl_mask
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

    Applies forward-fill imputation to match the actual pipeline behavior.

    Args:
        processed_dir: Path to processed data directory.
        n_samples: Number of samples to load.

    Returns:
        Tuple of (timeseries, obs_mask, n_features, seq_length).
    """
    print(f"Loading {n_samples} samples from {processed_dir}...")

    with open(processed_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    feature_names = metadata["feature_names"]
    seq_length = metadata["seq_length_hours"]
    n_features = len(feature_names)

    # Load normalization stats
    stats_path = processed_dir / "normalization_stats.yaml"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = yaml.safe_load(f)
        means = torch.tensor(stats["feature_means"], dtype=torch.float32)
        stds = torch.tensor(stats["feature_stds"], dtype=torch.float32)
        stds = torch.clamp(stds, min=1e-6)
        print("  Using prepared normalization stats")
    else:
        means = torch.zeros(n_features)
        stds = torch.ones(n_features)
        print("  Warning: No normalization stats found, using raw values")

    ts_df = pl.read_parquet(processed_dir / "timeseries.parquet").head(n_samples)

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
                else:
                    ts_tensor[t, f] = float("nan")
                mask_tensor[t, f] = bool(mask_data[t][f])

        # Forward-fill imputation (matches actual pipeline)
        for f in range(n_features):
            last_valid = means[f].item()
            for t in range(seq_length):
                if not torch.isnan(ts_tensor[t, f]):
                    last_valid = ts_tensor[t, f].item()
                else:
                    ts_tensor[t, f] = last_valid

        ts_tensor = (ts_tensor - means) / stds

        timeseries_list.append(ts_tensor)
        mask_list.append(mask_tensor)

    timeseries = torch.stack(timeseries_list)
    masks = torch.stack(mask_list)

    obs_ratio = masks.float().mean().item()
    print(f"  Shape: {timeseries.shape}")
    print(f"  Observation ratio: {obs_ratio:.1%}")

    return timeseries, masks, n_features, seq_length


def run_sanity_check(
    processed_dir: str,
    n_samples: int = 64,
    max_epochs: int = 500,
    target_loss: float = 0.01,
    log_every: int = 50,
    mask_ratio: float = 0.15,
    mask_strategy: str = "random",
    learning_rate: float = 1e-3,
    use_fixed_mask: bool = False,
) -> SanityCheckResult:
    """Run MAE overfitting sanity check.

    Args:
        processed_dir: Path to processed data directory.
        n_samples: Number of samples to use.
        max_epochs: Maximum training epochs.
        target_loss: Target loss to achieve.
        log_every: Log every N epochs.
        mask_ratio: Fraction of observed positions to mask.
        mask_strategy: Masking strategy (random, block, timestep, feature).
        learning_rate: Learning rate for optimizer.
        use_fixed_mask: If True, use fixed masks (true overfitting test).

    Returns:
        SanityCheckResult with pass/fail and final metrics.
    """
    print("=" * 70)
    print("MAE Pretraining Sanity Check")
    print("=" * 70)
    print()

    if use_fixed_mask:
        print("MODE: Fixed mask (true overfitting test)")
        print("  - Same mask positions every step")
        print("  - Masked loss should approach zero")
    else:
        print("MODE: Random masks (generalization test)")
        print("  - Different mask positions each step")
        print("  - Model learns generalizable reconstruction")
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

    timeseries, masks, n_features, seq_length = load_subset(processed_dir, n_samples)

    dataset = TensorDataset(timeseries, masks)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

    # Build encoder with NO dropout for overfitting
    d_model = 64
    encoder = build_encoder(
        "transformer",
        {
            "d_input": n_features,
            "d_model": d_model,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": d_model * 2,
            "dropout": 0.0,
            "max_seq_length": seq_length,
            "pooling": "none",
        },
    )

    # Build MAE objective with NO dropout
    # Note: MAE now uses two-token system (MISSING_TOKEN and MASK_TOKEN)
    mae_config = MAEConfig(
        name="mae",
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        min_block_size=3,
        max_block_size=10,
        decoder_d_model=32,
        decoder_n_layers=1,
        decoder_n_heads=4,
        decoder_d_ff=64,
        decoder_dropout=0.0,
        norm_target=False,
    )
    mae = build_ssl_objective(encoder, mae_config)

    total_params = sum(p.numel() for p in mae.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Mask ratio: {mask_ratio:.0%}, strategy: {mask_strategy}")

    optimizer = torch.optim.Adam(mae.parameters(), lr=learning_rate)

    print(f"\nTraining on {n_samples} samples for up to {max_epochs} epochs...")
    print(f"Target: masked loss < {target_loss}")
    print()
    print(
        f"{'Epoch':>6} | {'Train Masked':>12} | {'Train Visible':>13} | "
        f"{'Val Masked':>10} | {'Val Visible':>11}"
    )
    print("-" * 70)

    x, obs_mask = next(iter(loader))

    # Create fixed mask if requested
    fixed_ssl_mask = None
    if use_fixed_mask:
        fixed_ssl_mask = create_ssl_mask(
            shape=x.shape,
            mask_ratio=mask_ratio,
            strategy=mask_strategy,
            obs_mask=obs_mask,
            device=x.device,
            generator=torch.Generator().manual_seed(SEED),
        )

    converged_epoch = -1

    for epoch in range(max_epochs):
        mae.train()
        optimizer.zero_grad()

        if use_fixed_mask and fixed_ssl_mask is not None:
            # Manual forward with fixed mask using two-token system
            B, T, D = x.shape

            # Step 1: Replace missing positions with MISSING_TOKEN
            x_input = torch.where(obs_mask, x, mae.missing_token.expand(B, T, D))

            # Step 2: Replace SSL-masked positions with MASK_TOKEN
            ssl_masked_positions = obs_mask & ~fixed_ssl_mask
            x_input = torch.where(ssl_masked_positions, mae.mask_token.expand(B, T, D), x_input)

            # Step 3: Encode and decode
            encoder_output = mae.encoder(x_input, mask=None, padding_mask=None)
            x_recon = mae.decoder(encoder_output)

            # Loss only on masked AND observed positions (MASK_TOKEN positions)
            squared_error = (x_recon - x) ** 2
            loss_mask = (~fixed_ssl_mask) & obs_mask
            loss = (squared_error * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

            # Metrics for logging
            visible_mask = fixed_ssl_mask & obs_mask
            visible_loss = (
                squared_error * visible_mask.float()
            ).sum() / visible_mask.float().sum().clamp(min=1)

            train_metrics = {
                "mae_recon_loss_masked": loss.detach(),
                "mae_recon_loss_visible": visible_loss.detach(),
            }
        else:
            loss, train_metrics = mae(x, obs_mask)

        loss.backward()
        optimizer.step()

        train_masked = train_metrics["mae_recon_loss_masked"].item()
        train_visible = train_metrics["mae_recon_loss_visible"].item()

        # Validation (same data, deterministic masks)
        mae.eval()
        with torch.no_grad():
            _, val_metrics = mae(x, obs_mask)

        val_masked = val_metrics["mae_recon_loss_masked"].item()
        val_visible = val_metrics["mae_recon_loss_visible"].item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"{epoch+1:>6} | {train_masked:>12.6f} | {train_visible:>13.6f} | "
                f"{val_masked:>10.6f} | {val_visible:>11.6f}"
            )

        # Check convergence
        if use_fixed_mask:
            all_below_target = train_masked < target_loss
        else:
            random_mask_target = max(target_loss, 0.10)
            all_below_target = train_masked < random_mask_target

        if all_below_target and converged_epoch < 0:
            converged_epoch = epoch + 1
            if epoch >= max_epochs // 2:
                break

    # Final evaluation
    mae.eval()
    with torch.no_grad():
        _, final_val = mae(x, obs_mask)

    mae.train()
    _, final_train = mae(x, obs_mask)

    final_train_masked = final_train["mae_recon_loss_masked"].item()
    final_train_visible = final_train["mae_recon_loss_visible"].item()
    final_val_masked = final_val["mae_recon_loss_masked"].item()
    final_val_visible = final_val["mae_recon_loss_visible"].item()

    print("-" * 70)
    print(
        f"{'Final':>6} | {final_train_masked:>12.6f} | {final_train_visible:>13.6f} | "
        f"{final_val_masked:>10.6f} | {final_val_visible:>11.6f}"
    )
    print()

    # Determine pass/fail
    if use_fixed_mask:
        passed = final_val_masked < target_loss
        target_desc = f"masked loss < {target_loss}"
    else:
        random_mask_target = max(target_loss, 0.10)
        passed = final_val_masked < random_mask_target
        target_desc = f"masked loss < {random_mask_target}"

    if passed:
        print(f"[PASS] Achieved {target_desc}")
        if converged_epoch > 0:
            print(f"  Converged at epoch {converged_epoch}")
    else:
        print(f"[FAIL] Did not achieve {target_desc}")
        print(f"  Final val masked loss: {final_val_masked:.6f}")
        if use_fixed_mask:
            print("  With fixed masks, the model should memorize the positions.")
        else:
            print("  Try --fixed-mask to test if architecture can overfit.")

    return SanityCheckResult(
        passed=passed,
        final_train_masked=final_train_masked,
        final_train_visible=final_train_visible,
        final_val_masked=final_val_masked,
        final_val_visible=final_val_visible,
        epochs_to_converge=converged_epoch,
    )


def run_autoencoder_test(processed_dir: str) -> bool:
    """Test encoder-decoder reconstruction WITHOUT SSL masking.

    Isolates architecture issues from masking/SSL logic.

    Args:
        processed_dir: Path to processed data directory.

    Returns:
        True if test passes.
    """
    print("=" * 70)
    print("Autoencoder Test (No SSL Masking)")
    print("=" * 70)
    print()
    print("Testing if encoder-decoder can learn identity reconstruction.")
    print()

    L.seed_everything(SEED, workers=True)
    processed_dir_path = Path(processed_dir)

    timeseries, masks, n_features, seq_length = load_subset(processed_dir_path, n_samples=32)

    d_model = 64
    encoder = build_encoder(
        "transformer",
        {
            "d_input": n_features,
            "d_model": d_model,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": d_model * 2,
            "dropout": 0.0,
            "max_seq_length": seq_length,
            "pooling": "none",
        },
    )

    decoder = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, n_features),
    )

    class SimpleAutoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def forward(self, x, obs_mask):
            h = self.encoder(x, mask=obs_mask)
            return self.decoder(h)

    model = SimpleAutoencoder(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("Training for 200 epochs...")
    print()
    print(f"{'Epoch':>6} | {'MSE (all)':>12} | {'MSE (observed)':>15}")
    print("-" * 45)

    x, obs_mask = timeseries, masks

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        x_recon = model(x, obs_mask)
        loss_all = ((x_recon - x) ** 2).mean()
        loss_obs = ((x_recon - x) ** 2 * obs_mask.float()).sum() / obs_mask.float().sum().clamp(
            min=1
        )

        loss_obs.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"{epoch+1:>6} | {loss_all.item():>12.6f} | {loss_obs.item():>15.6f}")

    print("-" * 45)

    passed = loss_obs.item() < 0.01
    if passed:
        print("\n[PASS] Autoencoder converged")
        print("  Encoder-decoder architecture works correctly.")
    else:
        print("\n[FAIL] Autoencoder did NOT converge")
        print(f"  Final observed loss: {loss_obs.item():.6f}")
        print("  This suggests a fundamental architecture issue.")

    return passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MAE pretraining sanity check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test generalization learning (random masks)
  uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv

  # Test architecture can overfit (fixed mask)
  uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv --fixed-mask

  # Test encoder-decoder without SSL
  uv run python scripts/sanity_checks/sc_mae_pretraining.py data/processed/mimic-iv --autoencoder
        """,
    )
    parser.add_argument("processed_dir", help="Path to processed data directory")
    parser.add_argument(
        "--autoencoder",
        action="store_true",
        help="Test encoder-decoder without SSL masking",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs (default: 500)")
    parser.add_argument(
        "--fixed-mask",
        action="store_true",
        help="Use fixed mask (true overfitting test)",
    )

    args = parser.parse_args()

    if args.autoencoder:
        success = run_autoencoder_test(args.processed_dir)
    else:
        result = run_sanity_check(
            args.processed_dir,
            max_epochs=args.epochs,
            learning_rate=args.lr,
            use_fixed_mask=args.fixed_mask,
        )
        success = result.passed

    sys.exit(0 if success else 1)
