# Self-Supervised Learning Objectives

This module implements various SSL objectives for pretraining on unlabeled ICU time-series data.

## Available Objectives

### MAE (Masked Autoencoder)

Learns representations by masking portions of the input and reconstructing them.

**Key Features**:
- 4 masking strategies: random, block, timestep, feature
- Respects observation mask (ICU data-aware)
- Lightweight transformer decoder
- Configurable mask ratio and decoder architecture

**Usage**:
```python
from slices.models.pretraining import MAEConfig, MAEObjective

mae_config = MAEConfig(
    mask_ratio=0.15,
    mask_strategy="block",
    decoder_d_model=64,
    decoder_n_layers=2,
)
mae = MAEObjective(encoder, mae_config)

loss, metrics = mae(x, obs_mask)
```

**See**: `docs/MAE_IMPLEMENTATION.md` for detailed documentation.

## Factory Pattern

Use the factory for easy switching between objectives:

```python
from slices.models.pretraining import build_ssl_objective

# Build any SSL objective
ssl_obj = build_ssl_objective(encoder, config)
```

## Adding New Objectives

1. Create objective class extending `BaseSSLObjective`
2. Create config extending `SSLConfig`
3. Register in `factory.py`
4. Add tests in `tests/test_<objective>.py`
5. Add documentation in `docs/<OBJECTIVE>_IMPLEMENTATION.md`

## File Structure

```
pretraining/
├── __init__.py          # Exports
├── base.py              # Abstract base classes
├── factory.py           # Factory functions and registry
├── mae.py               # MAE implementation
└── README.md            # This file
```

## Requirements

All SSL objectives require:
- Encoder with `pooling="none"` (for per-timestep outputs)
- Input shape: `(B, T, D)`
- Observation mask shape: `(B, T, D)`

## Future Objectives

Planned implementations:
- **SimCLR**: Contrastive learning with augmentations
- **MoCo**: Momentum contrast
- **JEPA**: Joint-Embedding Predictive Architecture
- **TimeMAE**: Time-specific MAE variant
- **TS-TCC**: Time-series contrastive coding

## References

- **MAE**: He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners"
- **BERT**: Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
