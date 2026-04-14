# Training Module

This package contains the Lightning modules and helper utilities used by the
training scripts under `scripts/training/`.

## Main Components

### `SSLPretrainModule`

Builds:

- an encoder from `config.encoder`
- an SSL objective from `config.ssl`
- optimizer and scheduler state from `config.optimizer` and `config.scheduler`

What it does:

- logs `train/loss` and `val/loss`
- logs objective-specific metrics under `train/*` and `val/*`
- tracks `train/gradient_steps`
- tracks `train/wall_clock_seconds`
- exposes `save_encoder()` for downstream reuse

The actual pretraining entrypoint is `scripts/training/pretrain.py`, which also
enforces SSL/encoder compatibility and saves both last-epoch and best-val
encoder checkpoints.

Example:

```bash
uv run python scripts/training/pretrain.py dataset=miiv ssl=mae
uv run python scripts/training/pretrain.py dataset=miiv ssl=jepa
uv run python scripts/training/pretrain.py dataset=miiv ssl=ts2vec
```

### `FineTuneModule`

Composes:

- an encoder
- an optional missing-token wrapper
- a task head
- metric collections for train, val, and test

Supported task types:

- `binary`
- `multiclass`
- `multilabel`
- `regression`

Checkpoint inputs:

- encoder checkpoint from `encoder.pt`
- full Lightning pretrain checkpoint via `pretrain_checkpoint`
- no checkpoint for supervised-from-scratch runs

What it handles:

- linear probing via `training.freeze_encoder=true`
- full finetuning via `training.freeze_encoder=false`
- gradual unfreezing via `training.unfreeze_epoch`
- task-specific losses and metrics
- optional projection heads for dimension-controlled evaluation

The main entrypoints are:

- `scripts/training/finetune.py`
- `scripts/training/supervised.py`

Examples:

```bash
uv run python scripts/training/finetune.py dataset=miiv checkpoint=outputs/.../encoder.pt
uv run python scripts/training/finetune.py dataset=miiv checkpoint=outputs/.../encoder.pt training.freeze_encoder=false
uv run python scripts/training/supervised.py dataset=miiv
```

## Configuration Layout

The training scripts use Hydra config groups:

- `configs/model/` for encoder architecture
- `configs/ssl/` for SSL objective selection
- `configs/tasks/` for downstream task heads and metrics
- `configs/pretrain.yaml`, `configs/finetune.yaml`, `configs/supervised.yaml`, and `configs/xgboost.yaml` for phase-specific orchestration

Note that custom encoders belong in `configs/model/`, not `configs/encoder/`.

## Checkpoint Formats

The downstream path supports two checkpoint styles:

1. `encoder.pt`
   - compact encoder export for reuse across downstream runs
   - includes encoder config and optional learned missing token
2. full Lightning `.ckpt`
   - used when reconstructing the entire pretraining module state

`scripts/training/finetune.py` records which checkpoint source was actually used
for final test evaluation so post-run fairness evaluation can reuse the same
provenance.

## Utilities

`utils.py` contains shared helpers for:

- optimizer and scheduler construction
- callback setup
- W&B logger setup
- checkpoint export
- label-support validation
- fairness evaluation hooks
- data prerequisite checks

## Extending

To add a new encoder:

1. implement it in `src/slices/models/encoders/`
2. register it in `src/slices/models/encoders/factory.py`
3. add a Hydra config under `configs/model/`
4. add tests

To add a new SSL objective:

1. implement it in `src/slices/models/pretraining/`
2. register it in `src/slices/models/pretraining/factory.py`
3. add a Hydra config under `configs/ssl/`
4. add tests

To add a new task head:

1. implement it in `src/slices/models/heads/`
2. register it in `src/slices/models/heads/factory.py`
3. reference it from the relevant task config

## Testing

Useful focused test targets:

```bash
uv run pytest tests/test_pretrain_module.py -v
uv run pytest tests/test_finetune_module.py -v
uv run pytest tests/test_training_utils.py -v
```

## Related Docs

- `src/slices/models/encoders/README.md`
- `src/slices/models/pretraining/README.md`
- `src/slices/eval/README.md`
