"""Example: Using the Transformer Encoder for ICU Time-Series

This script demonstrates how to:
1. Create a transformer encoder from config
2. Process ICU time-series data with observation and padding masks
3. Use different pooling strategies
4. Integrate with PyTorch training loops
"""

import torch
import torch.nn as nn
from slices.models.encoders import TransformerConfig, TransformerEncoder


def basic_usage():
    """Basic usage: Create encoder and process time-series."""
    print("=== Basic Usage ===")

    # Create config
    config = TransformerConfig(
        d_input=35,  # Number of input features
        d_model=128,  # Model dimension
        n_layers=4,  # Number of transformer layers
        n_heads=8,  # Number of attention heads
        d_ff=512,  # Feedforward dimension
        max_seq_length=168,  # 7 days in hours
        pooling="mean",  # Mean pooling for sequence-level representation
    )

    # Create encoder
    encoder = TransformerEncoder(config)
    print(f"Created transformer with {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # Simulate ICU data: batch_size=16, seq_length=48 hours, features=35
    batch_size = 16
    seq_length = 48
    x = torch.randn(batch_size, seq_length, config.d_input)

    # Forward pass
    embeddings = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (mean pooling): {embeddings.shape}")
    print()


def with_observation_mask():
    """Using observation mask to handle missing values."""
    print("=== With Observation Mask (Missing Values) ===")

    config = TransformerConfig(
        d_input=35,
        d_model=128,
        n_layers=2,
        n_heads=8,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)

    batch_size = 8
    seq_length = 48
    x = torch.randn(batch_size, seq_length, config.d_input)

    # Create observation mask (True = observed, False = missing)
    # Simulate 30% missing values
    obs_mask = torch.rand(batch_size, seq_length, config.d_input) > 0.3

    print(f"Data shape: {x.shape}")
    print(
        f"Missing values: {(~obs_mask).sum().item()} / {obs_mask.numel()} "
        f"({100 * (~obs_mask).sum().item() / obs_mask.numel():.1f}%)"
    )

    # Note: Missing values should be imputed before passing to encoder
    # The mask is currently used for logging/analysis only
    embeddings = encoder(x, mask=obs_mask)
    print(f"Output embeddings: {embeddings.shape}")
    print()


def with_padding_mask():
    """Using padding mask for variable-length sequences."""
    print("=== With Padding Mask (Variable-Length Sequences) ===")

    config = TransformerConfig(
        d_input=35,
        d_model=128,
        n_layers=2,
        n_heads=8,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)

    batch_size = 8
    max_length = 72
    x = torch.randn(batch_size, max_length, config.d_input)

    # Create padding mask (True = valid, False = padding)
    # Simulate variable-length sequences
    padding_mask = torch.ones(batch_size, max_length, dtype=torch.bool)
    sequence_lengths = torch.randint(24, max_length, (batch_size,))

    for i, length in enumerate(sequence_lengths):
        padding_mask[i, length:] = False

    print(f"Data shape: {x.shape}")
    print(f"Sequence lengths: {sequence_lengths.tolist()}")

    # Forward pass with padding mask
    embeddings = encoder(x, padding_mask=padding_mask)
    print(f"Output embeddings: {embeddings.shape}")
    print("Note: Mean pooling correctly averages only over valid timesteps")
    print()


def different_pooling_strategies():
    """Comparing different pooling strategies."""
    print("=== Different Pooling Strategies ===")

    batch_size = 4
    seq_length = 48
    x = torch.randn(batch_size, seq_length, 35)

    pooling_strategies = ["mean", "max", "cls", "last", "none"]

    for pooling in pooling_strategies:
        config = TransformerConfig(
            d_input=35,
            d_model=128,
            n_layers=2,
            n_heads=8,
            pooling=pooling,
        )
        encoder = TransformerEncoder(config)

        embeddings = encoder(x)
        print(f"Pooling: {pooling:6s} -> Output shape: {embeddings.shape}")

    print()


def prenorm_vs_postnorm():
    """Comparing Pre-LN and Post-LN architectures."""
    print("=== Pre-LN vs Post-LN Transformer ===")

    x = torch.randn(8, 48, 35)

    # Pre-LN (modern, more stable)
    config_prenorm = TransformerConfig(
        d_input=35,
        d_model=128,
        n_layers=4,
        n_heads=8,
        prenorm=True,
    )
    encoder_prenorm = TransformerEncoder(config_prenorm)

    # Post-LN (original transformer)
    config_postnorm = TransformerConfig(
        d_input=35,
        d_model=128,
        n_layers=4,
        n_heads=8,
        prenorm=False,
    )
    encoder_postnorm = TransformerEncoder(config_postnorm)

    embeddings_prenorm = encoder_prenorm(x)
    embeddings_postnorm = encoder_postnorm(x)

    print(f"Pre-LN output shape: {embeddings_prenorm.shape}")
    print(f"Post-LN output shape: {embeddings_postnorm.shape}")
    print("Pre-LN is recommended for deeper models (more stable training)")
    print()


def training_loop_integration():
    """Integration with PyTorch training loop."""
    print("=== Training Loop Integration ===")

    # Setup
    config = TransformerConfig(
        d_input=35,
        d_model=128,
        n_layers=4,
        n_heads=8,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)

    # Simple downstream task: binary classification from embeddings
    classifier = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
    )

    # Optimizer
    model = nn.Sequential(encoder, classifier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Simulate training batch
    batch_size = 16
    x = torch.randn(batch_size, 48, 35)
    labels = torch.randint(0, 2, (batch_size, 1)).float()

    # Training step
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    print(f"Batch size: {batch_size}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def memory_efficient_inference():
    """Memory-efficient inference with torch.no_grad()."""
    print("=== Memory-Efficient Inference ===")

    config = TransformerConfig(
        d_input=35,
        d_model=128,
        n_layers=4,
        n_heads=8,
        pooling="mean",
    )
    encoder = TransformerEncoder(config)
    encoder.eval()  # Set to evaluation mode (disables dropout)

    # Large batch for inference
    batch_size = 128
    x = torch.randn(batch_size, 48, 35)

    # No gradient computation for inference
    with torch.no_grad():
        embeddings = encoder(x)

    print(f"Batch size: {batch_size}")
    print(f"Output shape: {embeddings.shape}")
    print("Using torch.no_grad() saves memory by not storing gradients")
    print()


def main():
    """Run all examples."""
    print("=" * 70)
    print("Transformer Encoder Examples for ICU Time-Series")
    print("=" * 70)
    print()

    basic_usage()
    with_observation_mask()
    with_padding_mask()
    different_pooling_strategies()
    prenorm_vs_postnorm()
    training_loop_integration()
    memory_efficient_inference()

    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
