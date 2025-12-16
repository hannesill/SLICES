# Transformer Encoder for ICU Time-Series

A modular, configurable transformer architecture designed for self-supervised learning on ICU time-series data with missing values and variable-length sequences.

## Features

- **Modular Architecture**: Clean separation between input projection, positional encoding, transformer layers, and pooling
- **Missing Value Support**: Handles observation masks for feature-level missingness
- **Variable-Length Sequences**: Proper padding mask handling for variable-length ICU stays
- **Multiple Pooling Strategies**: Mean, max, CLS token, last timestep, or no pooling
- **Modern Best Practices**: Pre-LN by default (more stable training)
- **Configurable**: All hyperparameters exposed via dataclass config

## Quick Start

```python
from slices.models.encoders import TransformerConfig, TransformerEncoder

# Create config
config = TransformerConfig(
    d_input=35,       # Number of input features
    d_model=128,      # Model dimension
    n_layers=4,       # Number of transformer layers
    n_heads=8,        # Number of attention heads
    d_ff=512,         # Feedforward dimension
    pooling="mean",   # Pooling strategy
)

# Create encoder
encoder = TransformerEncoder(config)

# Process ICU time-series
x = torch.randn(32, 48, 35)  # (batch, seq_len, features)
embeddings = encoder(x)       # (batch, d_model)
```

## Architecture

### Input Pipeline

1. **Input Projection**: Linear layer maps raw features to model dimension
   - Input: `(B, T, d_input)` → Output: `(B, T, d_model)`

2. **Positional Encoding**: Sinusoidal position embeddings (optional)
   - Provides temporal ordering information
   - Can be disabled with `use_positional_encoding=False`

3. **Optional CLS Token**: Prepended for BERT-style pooling
   - Only added when `pooling="cls"`

### Transformer Layers

Each layer consists of:
- **Multi-head Self-Attention**: Captures dependencies across timesteps
- **Feedforward Network**: Two-layer MLP with nonlinearity
- **Layer Normalization**: Pre-LN (default) or Post-LN
- **Residual Connections**: Around attention and feedforward

```
┌──────────────────────────┐
│   Input (B, T, d_model)  │
└──────────┬───────────────┘
           │
    ┌──────▼──────┐
    │  LayerNorm  │ (Pre-LN only)
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │  Multi-head     │
    │  Self-Attention │
    └──────┬──────────┘
           │
    ┌──────▼──────┐
    │  + Residual │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  LayerNorm  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Feedforward│
    │  (d_ff)     │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  + Residual │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  LayerNorm  │ (Pre-LN only)
    └──────┬──────┘
           │
  Output (B, T, d_model)
```

### Pooling Strategies

| Strategy | Output Shape | Use Case |
|----------|--------------|----------|
| `mean` | `(B, d_model)` | Average over valid timesteps (recommended for SSL) |
| `max` | `(B, d_model)` | Max pooling over valid timesteps |
| `cls` | `(B, d_model)` | Use [CLS] token representation (BERT-style) |
| `last` | `(B, d_model)` | Use last valid timestep |
| `none` | `(B, T, d_model)` | Return per-timestep embeddings |

## Configuration

### Basic Parameters

```python
@dataclass
class TransformerConfig(EncoderConfig):
    # Input/Output
    d_input: int = 35            # Number of input features
    d_model: int = 128           # Model dimension
    max_seq_length: int = 168    # Maximum sequence length

    # Architecture
    n_layers: int = 4            # Number of transformer layers
    n_heads: int = 8             # Number of attention heads
    d_ff: int = 512              # Feedforward dimension

    # Regularization
    dropout: float = 0.1         # Dropout probability

    # Options
    activation: str = "gelu"     # Activation: gelu, relu, silu
    prenorm: bool = True         # Pre-LN (True) or Post-LN (False)
    use_positional_encoding: bool = True
    pooling: str = "mean"        # Pooling strategy
```

### Common Configurations

**Small (for testing/debugging)**
```python
config = TransformerConfig(
    d_model=64, n_layers=2, n_heads=4, d_ff=256
)
# ~200K parameters
```

**Medium (default)**
```python
config = TransformerConfig(
    d_model=128, n_layers=4, n_heads=8, d_ff=512
)
# ~1.5M parameters
```

**Large (research-scale)**
```python
config = TransformerConfig(
    d_model=512, n_layers=8, n_heads=16, d_ff=2048
)
# ~25M parameters
```

## Handling Missing Values

ICU time-series data often has missing values. The encoder accepts an observation mask:

```python
# Create observation mask (True = observed, False = missing)
obs_mask = torch.rand(B, T, D) > 0.3  # 30% missing

# Important: Impute missing values BEFORE passing to encoder
x_imputed = impute(x, obs_mask)  # forward fill, mean, etc.

# Pass both data and mask
embeddings = encoder(x_imputed, mask=obs_mask)
```

**Note**: The observation mask is currently used for logging/analysis. Missing values should be imputed before encoding. Future SSL objectives may incorporate the mask explicitly (e.g., masked prediction).

## Handling Variable-Length Sequences

ICU stays have different lengths. Use padding masks to handle this:

```python
# Create padding mask (True = valid, False = padding)
padding_mask = torch.ones(B, T, dtype=torch.bool)
for i, length in enumerate(sequence_lengths):
    padding_mask[i, length:] = False

# Forward pass
embeddings = encoder(x, padding_mask=padding_mask)
```

**Important**: Pooling strategies (`mean`, `max`, `last`) correctly handle padding:
- `mean`: Averages only over valid timesteps
- `max`: Max only over valid timesteps
- `last`: Uses last valid timestep (not last position)

## Design Decisions

### Why Pre-LN (Pre-LayerNorm)?

Modern transformers use Pre-LN instead of the original Post-LN:
- **More stable training**: Gradients flow better, especially for deep models
- **Less sensitive to learning rate**: Easier to tune
- **Standard in recent work**: BERT variants, GPT-3, ViT all use Pre-LN

Set `prenorm=False` if you need original transformer behavior.

### Why Sinusoidal Positional Encoding?

- **Extrapolation**: Can handle sequences longer than seen during training
- **Deterministic**: No learned parameters, more interpretable
- **Standard**: Works well for time-series (timestamps are ordered)

Alternative: Learned positional embeddings (not yet implemented).

### Why Mean Pooling by Default?

For SSL pretraining, mean pooling is recommended:
- **Robust**: Less sensitive to outliers than max pooling
- **Uses all information**: Unlike CLS or last timestep
- **Simple**: No extra parameters

For supervised tasks, CLS pooling may work better (requires fine-tuning).

## Integration with SSL Objectives

The encoder is designed to work with SSL objectives:

```python
from slices.models.encoders import TransformerEncoder
from slices.models.pretraining import MaskedAutoencoderSSL  # TODO

# Create encoder
encoder = TransformerEncoder(config)

# Wrap in SSL objective
ssl_model = MaskedAutoencoderSSL(encoder, ssl_config)

# Pretraining
x, obs_mask = batch
loss, metrics = ssl_model(x, obs_mask)
loss.backward()
```

After pretraining, extract the encoder:
```python
pretrained_encoder = ssl_model.get_encoder()
```

## Performance Tips

### Memory Efficiency

- Use gradient checkpointing for very deep models (not yet implemented)
- Use `torch.no_grad()` for inference
- Reduce batch size if OOM

### Training Speed

- Use mixed precision training (AMP)
- Smaller models train faster (d_model=64-128 for initial experiments)
- Use `pooling="none"` only if you need per-timestep outputs

### Model Size vs. Performance

For ICU data (typical: 35 features, 48-hour windows):
- **Small models (d_model=64-128)**: Usually sufficient, faster training
- **Large models (d_model≥256)**: May overfit on small datasets
- **Deep models (n_layers≥6)**: Use Pre-LN and careful tuning

## Testing

Run the comprehensive test suite:
```bash
uv run pytest tests/test_transformer_encoder.py -v
```

Run the example script:
```bash
uv run python examples/transformer_encoder_example.py
```

## References

- **Attention Is All You Need** (Vaswani et al., 2017): Original transformer
- **On Layer Normalization in the Transformer Architecture** (Xiong et al., 2020): Pre-LN vs Post-LN
- **BERT** (Devlin et al., 2019): CLS token pooling
- **ricu** (GigaScience, 2023): ICU data concept dictionaries
- **YAIB** (ICLR, 2024): ICU benchmark tasks
