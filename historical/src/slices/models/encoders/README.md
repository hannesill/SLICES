# Encoder Architectures

This package contains the encoder backbones used across SSL pretraining,
downstream finetuning, and baseline experiments.

## Canonical Benchmark Encoder

The main benchmark uses `TransformerEncoder` from `transformer.py`.

Current canonical config:

- `d_model=64`
- `n_layers=2`
- `n_heads=4`
- `d_ff=256`
- `obs_aware=true`
- `pooling=mean` for downstream tasks
- `pooling=none` for SSL pretraining

The canonical source of truth for these defaults is
`configs/model/transformer.yaml`.

## Obs-Aware Timestep Tokenization

The benchmark transformer is not a plain linear projection over imputed values.
When `obs_aware=True`, each timestep token is built from:

```text
concat(values_t, obs_mask_t) -> MLP -> d_model
```

This gives the encoder direct access to both observed values and missingness
patterns. In the SSL path, `tokenize()` also marks fully unobserved hours as
invalid timestep tokens so they can be excluded from masking and loss logic.

## Core APIs

`TransformerEncoder` exposes three levels of interface:

1. `forward(x, mask, padding_mask)` for normal downstream use
2. `tokenize(x, obs_mask)` for timestep-token creation in SSL objectives
3. `encode(tokens, padding_mask)` for running transformer layers on prebuilt
   visible-token subsets

The SSL objectives use `tokenize()` and `encode()` directly so they can apply
their own masking strategies while sharing the same encoder body.

## Pooling

Supported pooling strategies for `TransformerEncoder`:

- `mean`
- `max`
- `cls`
- `last`
- `none`

Use `pooling="none"` only when the caller needs per-token outputs, which is the
case for MAE, JEPA, contrastive, TS2Vec, and SMART pretraining flows.

Example:

```python
from slices.models.encoders import TransformerConfig, TransformerEncoder

config = TransformerConfig(
    d_input=35,
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=256,
    obs_aware=True,
    pooling="mean",
)

encoder = TransformerEncoder(config)
embeddings = encoder(x, mask=obs_mask)
```

For SSL:

```python
ssl_config = TransformerConfig(
    d_input=35,
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=256,
    obs_aware=True,
    pooling="none",
)
```

## Other Encoders In This Package

- `ObservationTransformerEncoder` in `observation.py`
  - observation-level tokenization
  - kept as an alternate architecture, not the canonical benchmark encoder
- `SMARTEncoder` in `smart.py`
  - MART-style architecture required by the appendix SMART objective
- `GRUDEncoder` in `gru_d.py`
  - GRU-D baseline for contextual comparisons
- `LinearEncoder` in `linear.py`
  - simple baseline / utility encoder

## Which Encoder Goes With Which Objective

- `mae`, `jepa`, `contrastive`, `ts2vec`
  - use `TransformerEncoder`
  - require `obs_aware=True`
  - require `pooling="none"` during pretraining
- `smart`
  - requires `SMARTEncoder`
  - requires `pooling="none"`

The compatibility checks are enforced in `scripts/training/pretrain.py`.

## Notes On Missingness And Padding

- Observation masks are used directly by the obs-aware transformer path.
- Padding masks are handled at the sequence level and inverted internally to
  match PyTorch attention conventions.
- Fully unobserved hours are treated as invalid SSL timesteps in
  `TransformerEncoder.tokenize()`.

## Testing

Relevant test coverage lives in:

- `tests/test_transformer_encoder.py`
- `tests/test_smart_encoder.py`
- `tests/test_factories.py`

Example script:

```bash
uv run python examples/transformer_encoder_example.py
```
