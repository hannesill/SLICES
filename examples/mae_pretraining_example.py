"""Example: MAE pretraining with TransformerEncoder and ICUDataset.

This example demonstrates how to:
1. Load ICU data with the dataset
2. Configure a transformer encoder
3. Set up MAE (Masked Autoencoder) SSL objective
4. Run a simple training loop
5. Switch between different masking strategies

The MAE objective learns representations by:
- Masking portions of the input time-series
- Encoding the masked input
- Reconstructing the original values
- Minimizing reconstruction error on masked positions
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from slices.data.dataset import ICUDataset
from slices.models.encoders import TransformerEncoder, TransformerConfig
from slices.models.pretraining import MAEConfig, MAEObjective, build_ssl_objective


def create_mae_model(
    d_input: int,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 8,
    mask_ratio: float = 0.15,
    mask_strategy: str = "random",
) -> nn.Module:
    """Create MAE model with transformer encoder.
    
    Args:
        d_input: Input feature dimension.
        d_model: Transformer model dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        mask_ratio: Fraction of input to mask.
        mask_strategy: Masking strategy (random, block, timestep, feature).
    
    Returns:
        MAE objective module.
    """
    # Configure encoder (must use pooling='none' for MAE)
    encoder_config = TransformerConfig(
        d_input=d_input,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=4 * d_model,
        dropout=0.1,
        pooling="none",  # Required for MAE - need per-timestep outputs
        use_positional_encoding=True,
        prenorm=True,
    )
    encoder = TransformerEncoder(encoder_config)
    
    # Configure MAE objective
    mae_config = MAEConfig(
        name="mae",
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        min_block_size=3,
        max_block_size=10,
        mask_value=0.0,
        decoder_d_model=d_model // 2,  # Lighter decoder
        decoder_n_layers=2,
        decoder_n_heads=4,
        decoder_d_ff=2 * d_model,
        decoder_dropout=0.1,
        loss_on_observed_only=True,  # Only penalize on originally observed values
        norm_target=False,  # Whether to normalize reconstruction targets
    )
    
    # Build MAE objective
    mae = MAEObjective(encoder, mae_config)
    
    return mae


def train_mae_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one epoch of MAE training.
    
    Args:
        model: MAE objective module.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        device: Device to train on.
    
    Returns:
        Dict of average metrics for the epoch.
    """
    model.train()
    
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0
    
    for batch in dataloader:
        # Move batch to device
        timeseries = batch["timeseries"].to(device)  # (B, T, D)
        mask = batch["mask"].to(device)  # (B, T, D)
        
        # Forward pass
        loss, metrics = model(timeseries, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item() if torch.is_tensor(value) else value
        
        n_batches += 1
    
    # Average metrics
    avg_metrics = {
        "loss": total_loss / n_batches,
        **{k: v / n_batches for k, v in total_metrics.items()}
    }
    
    return avg_metrics


def main():
    """Main example demonstrating MAE pretraining."""
    
    # Configuration
    data_dir = Path("data/processed/mimic-iv-demo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("MAE Pretraining Example")
    print("=" * 80)
    
    # Check if data exists
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please run setup_mimic_iv.py first to extract data.")
        return
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_dir}...")
    dataset = ICUDataset(
        data_dir=data_dir,
        task_name=None,  # No task labels needed for pretraining
        impute_strategy="forward_fill",
        normalize=True,
    )
    
    print(f"   ✓ Loaded {len(dataset)} ICU stays")
    print(f"   ✓ Feature dimension: {dataset.n_features}")
    print(f"   ✓ Sequence length: {dataset.seq_length}")
    
    # Create dataloader
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"   ✓ Created dataloader with batch_size={batch_size}")
    
    # =========================================================================
    # Example 1: Random masking (BERT-style)
    # =========================================================================
    print("\n2. Creating MAE model with RANDOM masking...")
    mae_random = create_mae_model(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        mask_ratio=0.15,  # Mask 15% of positions
        mask_strategy="random",
    ).to(device)
    
    n_params = sum(p.numel() for p in mae_random.parameters())
    print(f"   ✓ Model created with {n_params:,} parameters")
    print(f"   ✓ Masking: 15% random positions")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(mae_random.parameters(), lr=1e-3)
    
    # Train for a few batches
    print("\n3. Training with random masking...")
    mae_random.train()
    
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just 3 batches for demo
            break
        
        timeseries = batch["timeseries"].to(device)
        mask = batch["mask"].to(device)
        
        loss, metrics = mae_random(timeseries, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {i+1}: loss={loss.item():.4f}, "
              f"masked_loss={metrics['mae_recon_loss_masked'].item():.4f}, "
              f"visible_loss={metrics['mae_recon_loss_visible'].item():.4f}")
    
    # =========================================================================
    # Example 2: Block masking (for temporal structure learning)
    # =========================================================================
    print("\n4. Creating MAE model with BLOCK masking...")
    mae_block = create_mae_model(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        mask_ratio=0.15,
        mask_strategy="block",  # Mask contiguous time blocks
    ).to(device)
    
    print(f"   ✓ Model created with block masking")
    print(f"   ✓ Masking: 15% in contiguous blocks (3-10 timesteps)")
    
    optimizer = torch.optim.AdamW(mae_block.parameters(), lr=1e-3)
    
    print("\n5. Training with block masking...")
    mae_block.train()
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        timeseries = batch["timeseries"].to(device)
        mask = batch["mask"].to(device)
        
        loss, metrics = mae_block(timeseries, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {i+1}: loss={loss.item():.4f}, "
              f"mask_ratio={metrics['mae_mask_ratio_actual']:.3f}")
    
    # =========================================================================
    # Example 3: Timestep masking (mask entire timesteps)
    # =========================================================================
    print("\n6. Creating MAE model with TIMESTEP masking...")
    mae_timestep = create_mae_model(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        mask_ratio=0.15,
        mask_strategy="timestep",  # Mask entire timesteps
    ).to(device)
    
    print(f"   ✓ Model created with timestep masking")
    print(f"   ✓ Masking: 15% of timesteps (all features)")
    
    optimizer = torch.optim.AdamW(mae_timestep.parameters(), lr=1e-3)
    
    print("\n7. Training with timestep masking...")
    mae_timestep.train()
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        timeseries = batch["timeseries"].to(device)
        mask = batch["mask"].to(device)
        
        loss, metrics = mae_timestep(timeseries, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {i+1}: loss={loss.item():.4f}, "
              f"obs_ratio={metrics['mae_obs_ratio']:.3f}")
    
    # =========================================================================
    # Example 4: Feature masking (mask entire features)
    # =========================================================================
    print("\n8. Creating MAE model with FEATURE masking...")
    mae_feature = create_mae_model(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        mask_ratio=0.15,
        mask_strategy="feature",  # Mask entire features
    ).to(device)
    
    print(f"   ✓ Model created with feature masking")
    print(f"   ✓ Masking: 15% of features (all timesteps)")
    
    optimizer = torch.optim.AdamW(mae_feature.parameters(), lr=1e-3)
    
    print("\n9. Training with feature masking...")
    mae_feature.train()
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        timeseries = batch["timeseries"].to(device)
        mask = batch["mask"].to(device)
        
        loss, metrics = mae_feature(timeseries, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {i+1}: loss={loss.item():.4f}")
    
    # =========================================================================
    # Example 5: Using factory function for easy switching
    # =========================================================================
    print("\n10. Using factory function for easy SSL objective switching...")
    
    # Create encoder
    encoder_config = TransformerConfig(
        d_input=dataset.n_features,
        d_model=128,
        n_layers=4,
        n_heads=8,
        pooling="none",
    )
    encoder = TransformerEncoder(encoder_config).to(device)
    
    # Configure MAE
    mae_config = MAEConfig(
        name="mae",
        mask_ratio=0.20,
        mask_strategy="block",
    )
    
    # Build using factory
    ssl_objective = build_ssl_objective(encoder, mae_config).to(device)
    
    print(f"   ✓ Created SSL objective: {mae_config.name}")
    print(f"   ✓ Strategy: {mae_config.mask_strategy}")
    print(f"   ✓ Mask ratio: {mae_config.mask_ratio}")
    
    # Train for one batch
    optimizer = torch.optim.AdamW(ssl_objective.parameters(), lr=1e-3)
    
    batch = next(iter(dataloader))
    timeseries = batch["timeseries"].to(device)
    mask = batch["mask"].to(device)
    
    loss, metrics = ssl_objective(timeseries, mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Trained one batch: loss={loss.item():.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\n✓ MAE objective implemented with 4 masking strategies:")
    print("  1. Random: Uniform random masking (BERT-style)")
    print("  2. Block: Contiguous time blocks (temporal structure)")
    print("  3. Timestep: Entire timesteps (all features)")
    print("  4. Feature: Entire features (all timesteps)")
    print("\n✓ Key features:")
    print("  - Respects observation mask (only masks observed values)")
    print("  - Configurable mask ratio and strategy")
    print("  - Lightweight decoder for reconstruction")
    print("  - Loss computed only on masked positions")
    print("  - Easy to switch objectives via factory pattern")
    print("\n✓ Next steps:")
    print("  - Implement other SSL objectives (contrastive, JEPA, etc.)")
    print("  - Add to scripts/pretrain.py for full training")
    print("  - Evaluate learned representations on downstream tasks")
    print("=" * 80)


if __name__ == "__main__":
    main()
