# Training Module

This module contains Lightning modules for training SSL and supervised models.

## Components

### `SSLPretrainModule`

Lightning module for SSL pretraining. This module is fully agnostic to:
- **Encoder architecture**: Works with any encoder (transformer, RNN, CNN, etc.)
- **SSL objective**: Works with any SSL objective (MAE, contrastive, JEPA, etc.)

All components are built from configuration using factory patterns.

#### Key Features

- **Configuration-driven**: All models built from YAML configs via Hydra
- **Flexible optimizers**: Supports Adam, AdamW, SGD with configurable parameters
- **Learning rate schedulers**: Cosine annealing, step decay, plateau, warmup+cosine
- **Automatic logging**: Logs all metrics from SSL objectives to tensorboard/wandb
- **Checkpoint management**: Saves encoder weights separately for downstream tasks
- **Distributed training**: Works with multi-GPU via Lightning

#### Usage

```python
from omegaconf import OmegaConf
from slices.training import SSLPretrainModule

# Load config (typically from Hydra)
config = OmegaConf.load("configs/pretrain.yaml")

# Create module
module = SSLPretrainModule(config)

# Train with Lightning Trainer
import lightning.pytorch as L
trainer = L.Trainer(max_epochs=100, devices=1)
trainer.fit(module, datamodule=datamodule)

# Save encoder for downstream tasks
module.save_encoder("pretrained_encoder.pt")
```

#### Config Structure

The module expects a config with three main sections:

```yaml
encoder:
  name: transformer  # Encoder architecture name
  d_input: 35       # Set automatically from data
  d_model: 128
  # ... encoder-specific params

ssl:
  name: mae         # SSL objective name
  mask_ratio: 0.15
  # ... objective-specific params

optimizer:
  name: adamw
  lr: 1.0e-3
  weight_decay: 0.01

scheduler:  # Optional
  name: warmup_cosine
  warmup_epochs: 10
  max_epochs: 100
```

#### Methods

- **`forward(timeseries, mask)`**: Forward pass through SSL objective
- **`training_step(batch, batch_idx)`**: Training step (called by Lightning)
- **`validation_step(batch, batch_idx)`**: Validation step (called by Lightning)
- **`configure_optimizers()`**: Configure optimizer and optional scheduler
- **`get_encoder()`**: Get encoder module (e.g., for inspection)
- **`save_encoder(path)`**: Save encoder weights to file

#### Logged Metrics

The module automatically logs:
- `train/loss`: Training loss (per step and per epoch)
- `val/loss`: Validation loss (per epoch)
- `train/{metric}`: All metrics returned by SSL objective (per epoch)
- `val/{metric}`: All validation metrics (per epoch)
- Learning rate (when using scheduler)

#### Supported Optimizers

- **adam**: Classic Adam optimizer
- **adamw**: Adam with decoupled weight decay (recommended)
- **sgd**: Stochastic gradient descent with momentum

#### Supported Schedulers

- **cosine**: Cosine annealing learning rate decay
- **step**: Step decay (reduce LR every N epochs)
- **plateau**: Reduce LR when validation loss plateaus
- **warmup_cosine**: Linear warmup followed by cosine decay (recommended for SSL)

#### Example: Different Optimizers

```yaml
# AdamW (recommended for transformers)
optimizer:
  name: adamw
  lr: 1.0e-3
  weight_decay: 0.01

# Adam (no weight decay)
optimizer:
  name: adam
  lr: 1.0e-3
  weight_decay: 0.0

# SGD with momentum
optimizer:
  name: sgd
  lr: 1.0e-2
  weight_decay: 1.0e-4
  momentum: 0.9
```

#### Example: Different Schedulers

```yaml
# Warmup + Cosine (recommended for SSL)
scheduler:
  name: warmup_cosine
  warmup_epochs: 10
  max_epochs: 100
  eta_min: 1.0e-6

# Cosine annealing only
scheduler:
  name: cosine
  T_max: 100
  eta_min: 0.0

# Step decay
scheduler:
  name: step
  step_size: 30
  gamma: 0.1

# Reduce on plateau
scheduler:
  name: plateau
  mode: min
  factor: 0.1
  patience: 10
```

## Adding New Components

### Adding a New Encoder

1. Create encoder class in `src/slices/models/encoders/`:

```python
from .base import BaseEncoder, EncoderConfig

@dataclass
class MyEncoderConfig(EncoderConfig):
    # Add encoder-specific params
    hidden_size: int = 256

class MyEncoder(BaseEncoder):
    def __init__(self, config: MyEncoderConfig):
        super().__init__(config)
        # ... build encoder

    def forward(self, x, mask=None, padding_mask=None):
        # ... encode input
        return encoded
```

2. Register in `src/slices/models/encoders/factory.py`:

```python
from .my_encoder import MyEncoder, MyEncoderConfig

ENCODER_REGISTRY["my_encoder"] = MyEncoder
ENCODER_CONFIG_REGISTRY["my_encoder"] = MyEncoderConfig
```

3. Create config file `configs/encoder/my_encoder.yaml`:

```yaml
name: my_encoder
hidden_size: 256
# ... other params
```

4. Use in pretraining:

```bash
uv run python scripts/pretrain.py encoder=my_encoder
```

### Adding a New SSL Objective

1. Create objective class in `src/slices/models/pretraining/`:

```python
from .base import BaseSSLObjective, SSLConfig

@dataclass
class MySSLConfig(SSLConfig):
    name: str = "my_ssl"
    # Add objective-specific params

class MySSLObjective(BaseSSLObjective):
    def __init__(self, encoder, config: MySSLConfig):
        super().__init__(encoder, config)
        # ... build objective-specific components

    def forward(self, x, mask):
        # ... compute SSL loss
        return loss, metrics
```

2. Register in `src/slices/models/pretraining/factory.py`:

```python
from .my_ssl import MySSLObjective, MySSLConfig

SSL_REGISTRY["my_ssl"] = MySSLObjective
CONFIG_REGISTRY["my_ssl"] = MySSLConfig
```

3. Create config file `configs/ssl/my_ssl.yaml`:

```yaml
name: my_ssl
# ... ssl-specific params
```

4. Use in pretraining:

```bash
uv run python scripts/pretrain.py ssl=my_ssl
```

### `FineTuneModule`

Lightning module for downstream task finetuning. This module composes a pretrained encoder with a task head for supervised learning on clinical prediction tasks.

#### Key Features

- **Flexible checkpoint loading**: Supports loading from encoder weights (`.pt`) or full pretrain checkpoints (`.ckpt`)
- **Freezing strategies**: Linear probing (frozen encoder), full finetuning, or gradual unfreezing
- **Task support**: Binary classification, multiclass, multilabel, and regression
- **Comprehensive metrics**: AUROC, AUPRC, accuracy, F1 score for classification tasks
- **Class imbalance handling**: Configurable loss functions and class weights

#### Usage

```python
from omegaconf import OmegaConf
from slices.training import FineTuneModule

# Load config (typically from Hydra)
config = OmegaConf.load("configs/finetune.yaml")

# Create module with pretrained encoder
module = FineTuneModule(
    config=config,
    checkpoint_path="outputs/encoder.pt",  # From pretraining
)

# Train with Lightning Trainer
import lightning.pytorch as L
trainer = L.Trainer(max_epochs=50, devices=1)
trainer.fit(module, datamodule=datamodule)
```

#### Config Structure

The module expects a config with encoder, task, and training sections:

```yaml
encoder:
  name: transformer
  d_model: 128
  # ... encoder config (should match pretrained encoder)

task:
  task_name: mortality_24h
  task_type: binary  # binary | multiclass | multilabel | regression
  head_type: mlp    # mlp | linear
  hidden_dims: [64]
  dropout: 0.1

training:
  freeze_encoder: true  # Linear probing (default)
  unfreeze_epoch: null  # Set to N for gradual unfreezing
  max_epochs: 50
  batch_size: 64

optimizer:
  name: adamw
  lr: 1.0e-4  # Lower LR for finetuning
  weight_decay: 0.01
```

#### Finetuning Strategies

1. **Linear Probing** (default): Freeze encoder, train only task head
   ```yaml
   training:
     freeze_encoder: true
   ```

2. **Full Finetuning**: Train both encoder and task head
   ```yaml
   training:
     freeze_encoder: false
   ```

3. **Gradual Unfreezing**: Start frozen, unfreeze after N epochs
   ```yaml
   training:
     freeze_encoder: true
     unfreeze_epoch: 5  # Unfreeze at epoch 5
   ```

#### Task Types

- **binary**: Binary classification (2 outputs for CrossEntropyLoss)
- **multiclass**: Multi-class classification (requires `n_classes`)
- **multilabel**: Multi-label classification (requires `n_classes`)
- **regression**: Continuous prediction (1 output, MSELoss)

#### Logged Metrics

For classification tasks:
- `train/loss`, `val/loss`, `test/loss`: Cross-entropy loss
- `train/accuracy`, `val/accuracy`, `test/accuracy`: Classification accuracy
- `val/auroc`, `test/auroc`: Area under ROC curve
- `val/auprc`, `test/auprc`: Area under PR curve
- `val/f1`, `test/f1`: F1 score

For regression tasks:
- `train/loss`, `val/loss`, `test/loss`: MSE loss

#### Methods

- **`forward(timeseries, mask)`**: Forward pass through encoder and task head
- **`training_step(batch, batch_idx)`**: Training step (called by Lightning)
- **`validation_step(batch, batch_idx)`**: Validation step (called by Lightning)
- **`test_step(batch, batch_idx)`**: Test step (called by Lightning)
- **`on_train_epoch_start()`**: Called at start of each epoch (handles gradual unfreezing)
- **`configure_optimizers()`**: Configure optimizer and optional scheduler
- **`get_encoder()`**: Get encoder module
- **`get_task_head()`**: Get task head module

#### Loading Checkpoints

The module supports two checkpoint formats:

1. **Encoder weights** (`.pt` file from `pretrain.py`):
   ```python
   module = FineTuneModule(
       config=config,
       checkpoint_path="outputs/encoder.pt",
   )
   ```

2. **Full pretrain checkpoint** (`.ckpt` file from Lightning):
   ```python
   module = FineTuneModule(
       config=config,
       pretrain_checkpoint_path="outputs/ssl-last.ckpt",
   )
   ```

#### Adding a New Task Head

1. Create task head class in `src/slices/models/heads/`:

```python
from .base import BaseTaskHead, TaskHeadConfig

class MyTaskHead(BaseTaskHead):
    def __init__(self, config: TaskHeadConfig):
        super().__init__(config)
        # ... build head architecture

    def forward(self, encoder_output):
        # ... compute predictions
        return {"logits": logits, "probs": probs}
```

2. Register in `src/slices/models/heads/factory.py`:

```python
from .my_head import MyTaskHead

TASK_HEAD_REGISTRY["my_head"] = MyTaskHead
```

3. Use in finetuning:

```yaml
task:
  head_type: my_head
  # ... other config
```

## Testing

Test the modules with pytest:

```bash
# Test pretraining module
uv run pytest tests/test_pretrain_module.py -v

# Test finetuning module
uv run pytest tests/test_finetune_module.py -v
```

## See Also

- [Pretraining Guide](../../../docs/PRETRAINING_GUIDE.md) - Full guide to using the pretraining pipeline
- [Encoder README](../models/encoders/README.md) - Details on encoder architectures
- [SSL Objectives README](../models/pretraining/README.md) - Details on SSL objectives
- [Task Heads README](../models/heads/README.md) - Details on task head architectures
