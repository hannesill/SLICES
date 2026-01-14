"""Overfitting sanity check for the training pipeline.

Loads a small subset of real data and trains until loss approaches zero.
If the model can't overfit, something is broken.

Prerequisites:
    Run prepare_dataset.py first to generate splits.yaml and normalization_stats.yaml

Usage:
    uv run python scripts/sanity_checks/sc_supervised_learning.py data/processed/mimic-iv-demo
"""

import sys
from pathlib import Path

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from slices.models.encoders import build_encoder
from slices.models.heads import TaskHeadConfig, build_task_head
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


def load_subset(
    processed_dir: Path,
    n_samples: int = 20,
    task_name: str = "mortality_24h",
):
    """Load a small subset using prepared normalization stats."""
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
        print("  Using prepared normalization stats")
    else:
        means = torch.zeros(n_features)
        stds = torch.ones(n_features)
        print("  Warning: No normalization stats found, using raw values")

    # Load data
    ts_df = pl.read_parquet(processed_dir / "timeseries.parquet").head(n_samples)
    labels_df = pl.read_parquet(processed_dir / "labels.parquet").head(n_samples)

    # Convert to tensors
    timeseries_list = []
    mask_list = []

    for row in ts_df.iter_rows(named=True):
        ts_data = row["timeseries"]
        mask_data = row["mask"]

        ts_tensor = torch.zeros(seq_length, n_features)
        mask_tensor = torch.zeros(seq_length, n_features)

        for t in range(min(len(ts_data), seq_length)):
            for f in range(n_features):
                val = ts_data[t][f]
                if val is not None and not np.isnan(val):
                    ts_tensor[t, f] = val
                mask_tensor[t, f] = float(mask_data[t][f])

        # Zero imputation then normalize
        ts_tensor = torch.nan_to_num(ts_tensor, nan=0.0)
        ts_tensor = (ts_tensor - means) / stds

        timeseries_list.append(ts_tensor)
        mask_list.append(mask_tensor)

    timeseries = torch.stack(timeseries_list)
    masks = torch.stack(mask_list)

    # Get labels
    labels = torch.tensor(labels_df[task_name].to_list(), dtype=torch.long)

    print(f"  Shape: {timeseries.shape}")
    print(f"  Labels: {labels.sum().item()} positive, {(labels == 0).sum().item()} negative")

    return timeseries, masks, labels, n_features, seq_length


def run_sanity_check(processed_dir: str, max_epochs: int = 300, target_loss: float = 0.001):
    """Run overfitting sanity check."""
    print("=" * 60)
    print("Overfitting Sanity Check")
    print("=" * 60)

    L.seed_everything(SEED, workers=True)

    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist")
        return False

    # Load small subset
    timeseries, masks, labels, n_features, seq_length = load_subset(processed_dir)
    n_samples = len(labels)

    # Create dataloader
    dataset = TensorDataset(timeseries, masks, labels)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

    # Build model
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
            "pooling": "mean",
        },
    )

    task_head = build_task_head(
        TaskHeadConfig(
            name="mlp",
            task_name="sanity_check",
            task_type="binary",
            input_dim=encoder.get_output_dim(),
            hidden_dims=[32],
            dropout=0.0,
        )
    )

    class Model(nn.Module):
        def __init__(self, enc, head):
            super().__init__()
            self.encoder = enc
            self.head = head

        def forward(self, x, mask):
            h = self.encoder(x, mask=mask)
            return self.head(h)["logits"]

    model = Model(encoder, task_head)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nTraining on {n_samples} samples for up to {max_epochs} epochs...")
    print(f"Target: loss < {target_loss}\n")

    # Training loop
    model.train()
    for epoch in range(max_epochs):
        for x, mask, y in loader:
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = model(x, mask).argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.6f}, accuracy={accuracy:.1%}")

        if loss.item() < target_loss:
            print(f"\n[PASS] Achieved target at epoch {epoch+1}")
            print(f"  Final loss: {loss.item():.6f}")
            print(f"  Final accuracy: {accuracy:.1%}")
            return True

    print(f"\n[FAIL] Could not overfit after {max_epochs} epochs")
    print(f"  Final loss: {loss.item():.6f}")
    print(f"  Final accuracy: {accuracy:.1%}")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/sanity_check.py <processed_dir>")
        print("Example: uv run python scripts/sanity_check.py data/processed/mimic-iv-demo")
        sys.exit(1)

    success = run_sanity_check(sys.argv[1])
    sys.exit(0 if success else 1)
