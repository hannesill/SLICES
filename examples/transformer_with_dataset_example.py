"""Example: Integrating Transformer Encoder with ICUDataset

This script demonstrates how to use the transformer encoder with real ICU data
loaded from the ICUDataset. It shows the complete pipeline from data loading
to embedding extraction.
"""

from pathlib import Path

import torch
from slices.models.encoders import TransformerConfig, TransformerEncoder


def check_data_availability() -> bool:
    """Check if extracted MIMIC-IV demo data is available."""
    data_dir = Path("data/processed/mimic-iv-demo")
    return data_dir.exists() and (data_dir / "timeseries.parquet").exists()


def basic_integration():
    """Basic integration: Load dataset and process with transformer."""
    print("=== Basic Integration with ICUDataset ===")

    if not check_data_availability():
        print("⚠️  MIMIC-IV demo data not found.")
        print("   Run: uv run python scripts/setup_mimic_iv.py")
        print("   This example requires extracted ICU data.\n")
        return False

    from slices.data import ICUDataset

    # Load dataset
    dataset = ICUDataset(
        "data/processed/mimic-iv-demo",
        task_name="mortality_24h",
        normalize=True,
        impute_strategy="forward_fill",
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Features: {dataset.n_features}")
    print(f"Sequence length: {dataset.seq_length}")
    print(f"Available tasks: {dataset.task_names}")

    # Get a sample
    sample = dataset[0]
    print("\nSample structure:")
    print(f"  timeseries: {sample['timeseries'].shape}")
    print(f"  mask: {sample['mask'].shape}")
    print(f"  label: {sample['label'].item()}")
    print(f"  static: {list(sample['static'].keys())}")

    # Create transformer
    config = TransformerConfig(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        max_seq_length=dataset.seq_length,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)
    print(f"\nTransformer created: {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # Process single sample
    x = sample["timeseries"].unsqueeze(0)  # Add batch dimension
    mask = sample["mask"].unsqueeze(0)

    encoder.eval()
    with torch.no_grad():
        embedding = encoder(x, mask=mask)

    print(f"\nSingle sample embedding: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm().item():.4f}")

    return True


def batch_processing():
    """Process a batch of samples."""
    print("=== Batch Processing ===")

    if not check_data_availability():
        print("⚠️  MIMIC-IV demo data not found. Skipping.\n")
        return False

    from slices.data import ICUDataset
    from torch.utils.data import DataLoader

    # Load dataset
    dataset = ICUDataset(
        "data/processed/mimic-iv-demo",
        task_name="mortality_24h",
        normalize=True,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging, >0 for production
    )

    # Create transformer
    config = TransformerConfig(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)
    encoder.eval()

    # Process first batch
    batch = next(iter(dataloader))
    x = batch["timeseries"]  # (B, T, D)
    mask = batch["mask"]  # (B, T, D)
    labels = batch["label"]  # (B,)

    print(f"Batch shape: {x.shape}")
    print(f"Labels: {labels.tolist()}")

    with torch.no_grad():
        embeddings = encoder(x, mask=mask)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Mean embedding norm: {embeddings.norm(dim=1).mean().item():.4f}")

    return True


def with_datamodule():
    """Use with ICUDataModule (Lightning)."""
    print("=== Integration with ICUDataModule ===")

    if not check_data_availability():
        print("⚠️  MIMIC-IV demo data not found. Skipping.\n")
        return False

    from slices.data import ICUDataModule

    # Create datamodule
    datamodule = ICUDataModule(
        processed_dir="data/processed/mimic-iv-demo",
        task_name="mortality_24h",
        batch_size=32,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Setup splits
    datamodule.setup()

    # Get split information
    split_info = datamodule.get_split_info()
    print(f"Train stays: {split_info['train_stays']} ({split_info['train_patients']} patients)")
    print(f"Val stays: {split_info['val_stays']} ({split_info['val_patients']} patients)")
    print(f"Test stays: {split_info['test_stays']} ({split_info['test_patients']} patients)")

    # Get train dataloader
    train_loader = datamodule.train_dataloader()

    # Create transformer
    config = TransformerConfig(
        d_input=datamodule.get_feature_dim(),
        d_model=128,
        n_layers=4,
        n_heads=8,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)

    # Simulate training loop
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

    batch = next(iter(train_loader))
    x = batch["timeseries"]
    mask = batch["mask"]
    _labels = batch["label"]  # Available for task heads

    # Forward pass
    embeddings = encoder(x, mask=mask)

    # Dummy loss (normally you'd have a task head)
    loss = embeddings.pow(2).mean()  # Just for demonstration

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("\nTraining step completed:")
    print(f"  Batch size: {x.size(0)}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Embeddings mean: {embeddings.mean().item():.4f}")

    return True


def different_sequence_lengths():
    """Handle sequences with different lengths (padding)."""
    print("=== Variable-Length Sequences ===")

    if not check_data_availability():
        print("⚠️  MIMIC-IV demo data not found. Skipping.\n")
        return False

    from slices.data import ICUDataset

    # Load dataset with shorter sequences
    dataset = ICUDataset(
        "data/processed/mimic-iv-demo",
        seq_length=24,  # Use only first 24 hours
        normalize=True,
    )

    print(f"Dataset with seq_length={dataset.seq_length}")

    # Create transformer (with longer max_seq_length)
    config = TransformerConfig(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=2,
        n_heads=8,
        max_seq_length=72,  # Can handle up to 72 hours
        pooling="mean",
    )
    encoder = TransformerEncoder(config)

    # Get sample
    sample = dataset[0]
    x = sample["timeseries"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    print(f"Input shape: {x.shape}")

    encoder.eval()
    with torch.no_grad():
        embedding = encoder(x, mask=mask)

    print(f"Output shape: {embedding.shape}")
    print("✓ Transformer handles variable sequence lengths correctly\n")

    return True


def main():
    """Run all examples."""
    print("=" * 70)
    print("Transformer + ICUDataset Integration Examples")
    print("=" * 70)
    print()

    # Track which examples ran successfully
    success = []

    success.append(basic_integration())
    success.append(batch_processing())
    success.append(with_datamodule())
    success.append(different_sequence_lengths())

    print("=" * 70)

    if any(success):
        print("✓ Examples completed successfully!")
    else:
        print("⚠️  No examples ran (MIMIC-IV demo data not found)")
        print("\nTo run these examples:")
        print("1. Download MIMIC-IV demo dataset")
        print("2. Run: uv run python scripts/setup_mimic_iv.py")

    print("=" * 70)


if __name__ == "__main__":
    main()
