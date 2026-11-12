# Self-Supervised Learning Objectives

This package contains the SSL objectives used by SLICES pretraining.

## Implemented Objectives

- `mae`
  - timestep-level masked autoencoding
  - reconstructs masked timestep features in input space
- `jepa`
  - timestep-level latent prediction with an EMA target encoder
- `contrastive`
  - instance-level or temporal NT-Xent over masked views
- `ts2vec`
  - temporal contrastive extension with noise augmentation and hierarchical loss
- `smart`
  - appendix-only external reference built around `SMARTEncoder`

The registry lives in `factory.py`.

## Encoder Requirements

The objectives do not all share the same encoder contract.

| Objective | Required Encoder | Key Requirements |
|---|---|---|
| `mae` | `TransformerEncoder` | `obs_aware=True`, `pooling="none"` |
| `jepa` | `TransformerEncoder` | `obs_aware=True`, `pooling="none"` |
| `contrastive` | `TransformerEncoder` | `obs_aware=True`, `pooling="none"` |
| `ts2vec` | `TransformerEncoder` | `obs_aware=True`, `pooling="none"` |
| `smart` | `SMARTEncoder` | `pooling="none"` |

These constraints are enforced in the objective constructors and by
`scripts/training/pretrain.py`.

## Benchmark Design

The controlled benchmark objectives share the same timestep-level obs-aware
transformer encoder and differ only in the SSL objective and masking logic.

Highlights:

- MAE reconstructs masked timestep features
- JEPA predicts masked latent targets from an EMA teacher
- Contrastive aligns sequence or temporal representations across masked views
- TS2Vec gives the contrastive family a stronger temporal objective
- SMART remains outside the controlled comparison because it swaps in MART

## Example

```python
from slices.models.pretraining import MAEConfig, build_ssl_objective

ssl_config = MAEConfig(mask_ratio=0.5)
ssl_objective = build_ssl_objective(encoder, ssl_config)

loss, metrics = ssl_objective(x, obs_mask)
```

## File Layout

```text
pretraining/
├── base.py
├── factory.py
├── masking.py
├── mae.py
├── jepa.py
├── contrastive.py
├── ts2vec.py
├── smart.py
└── README.md
```

## Objective Notes

### MAE

- operates on timestep tokens
- encoder sees only visible tokens
- decoder reconstructs full `(B, T, D)` values
- loss is computed only on observed features at masked timesteps

### JEPA

- uses the same timestep-token interface as MAE
- target encoder is an EMA copy kept in eval mode
- supports `mse` and `cosine` latent losses

### Contrastive

- supports `instance` and `temporal` modes
- standard benchmark setting uses complementary masked views in instance mode

### TS2Vec

- adds input-level Gaussian noise and optional cropping
- uses independent timestep masks
- applies a hierarchical temporal contrastive loss

### SMART

- uses element-wise masking, not timestep masking
- requires the MART-style `SMARTEncoder`
- included as an external appendix reference rather than the main benchmark

## Configuration

Hydra objective configs live in `configs/ssl/`:

- `configs/ssl/mae.yaml`
- `configs/ssl/jepa.yaml`
- `configs/ssl/contrastive.yaml`
- `configs/ssl/ts2vec.yaml`
- `configs/ssl/smart.yaml`

Switch objectives at the CLI with:

```bash
uv run python scripts/training/pretrain.py dataset=miiv ssl=jepa
```

## Extending

To add a new objective:

1. Add a config dataclass extending `SSLConfig`
2. Implement a `BaseSSLObjective` subclass
3. Register both in `factory.py`
4. Add objective-specific tests under `tests/`
5. Add or update the relevant Hydra config in `configs/ssl/`
