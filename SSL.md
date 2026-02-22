# Plan: Add JEPA and Contrastive SSL Objectives

## Context

The thesis requires comparing three SSL paradigms (MAE, JEPA, Contrastive) with the SSL objective as the **only** variable. The existing MAE already uses `ObservationTransformerEncoder` with a `tokenize()`/`encode()` split. JEPA and Contrastive must use the exact same encoder, same masking logic, and same hyperparameters — differing only in what they predict and how they compute loss.

The three paradigms form a clean comparison triangle:
- **MAE vs JEPA**: Same masking, same encoder input → tests input-space vs latent-space prediction
- **JEPA vs Contrastive**: Both in latent space → tests local positional prediction vs global invariance
- **MAE vs Contrastive**: Tests reconstruction vs discrimination

---

## Step 1: Create shared masking module

**Create** `src/slices/models/pretraining/masking.py`

Extract `_create_observation_mask` and `_extract_visible` from `mae.py:275-350` into standalone functions:
- `create_observation_mask(padding_mask, mask_ratio, device) → ssl_mask`
- `extract_visible(tokens, ssl_mask, padding_mask) → (visible_tokens, vis_padding)`

Same logic, just takes `mask_ratio` as an explicit parameter instead of `self.config.mask_ratio`.

## Step 2: Refactor MAE to use shared masking

**Modify** `src/slices/models/pretraining/mae.py`

- Import `create_observation_mask`, `extract_visible` from `.masking`
- In `forward()`, replace `self._create_observation_mask(...)` → `create_observation_mask(padding_mask, self.config.mask_ratio, device)`
- Replace `self._extract_visible(...)` → `extract_visible(tokens, ssl_mask, padding_mask)`
- Remove the two private methods from the class
- Run existing MAE tests to verify no regression

## Step 3: Implement JEPA objective

**Create** `src/slices/models/pretraining/jepa.py`

### JEPAConfig (dataclass extending SSLConfig)
- `mask_ratio: 0.75` (same as MAE)
- `predictor_d_model: 128`, `predictor_n_layers: 2`, `predictor_n_heads: 4`, `predictor_d_ff: 512`, `predictor_dropout: 0.1` (mirrors MAE decoder for fairness)
- `momentum_base: 0.996`, `momentum_final: 1.0`
- `loss_type: "mse"` (also supports `"cosine"`)

### JEPAPredictor (mirrors MAEDecoder architecture)
Same transformer-based design as `MAEDecoder` (`mae.py:45-177`):
- `encoder_proj`: Linear(d_encoder → d_predictor)
- `mask_token`: learnable Parameter(1, 1, d_predictor)
- `feature_embed` + `time_pe` for positional info
- Lightweight transformer (PyTorch TransformerEncoder, prenorm)
- `embed_dropout`
- **Key difference**: `output_proj` maps to `d_encoder` (not 1), producing representation vectors

`forward()` uses the same scatter logic as `MAEDecoder.forward()` (lines 132-175): fills mask_token at masked positions, scatters visible encoded tokens at their original positions, adds positional info, runs transformer.

### JEPAObjective (extends BaseSSLObjective)
- Validates encoder has `tokenize()`/`encode()` + `pooling='none'` (same as MAE)
- `missing_token = None` (observation encoder handles missingness intrinsically)
- Creates EMA `target_encoder = deepcopy(encoder)` with all params frozen (pattern from `smart.py:172-175`)
- Creates `JEPAPredictor`

**Forward pass:**
1. `self.encoder.tokenize(x, obs_mask)` → tokens, padding_mask, token_info
2. `create_observation_mask(padding_mask, mask_ratio, device)` → ssl_mask
3. `extract_visible(tokens, ssl_mask, padding_mask)` → visible_tokens, vis_padding
4. `self.encoder.encode(visible_tokens, vis_padding)` → context_repr (B, n_vis, d_model)
5. `self.target_encoder.tokenize(x, obs_mask)` then `self.target_encoder.encode(target_tokens, target_padding)` → target_repr (B, max_obs, d_model) [no grad] — target uses its own EMA-updated tokenization layers
6. `self.predictor(context_repr, ssl_mask, token_info, max_obs, padding_mask)` → predicted_repr (B, max_obs, d_model)
7. Loss: MSE (or cosine) between predicted_repr and target_repr at masked positions only

**momentum_update(progress)**: Same pattern as SMART (`smart.py:371-392`). Linear schedule base→final. Updates target encoder via EMA. Called by `SSLPretrainModule.on_train_batch_end` duck-typing hook.

**Metrics**: `jepa_loss`, `ssl_loss` (cross-paradigm alias), `jepa_mask_ratio_actual`, `jepa_n_tokens_per_sample`, `jepa_n_visible_per_sample`, `jepa_n_masked_per_sample`, `jepa_momentum`

## Step 4: Implement Contrastive objective

**Create** `src/slices/models/pretraining/contrastive.py`

### ContrastiveConfig (dataclass extending SSLConfig)
- `mask_ratio: 0.5` (same as MAE/JEPA)
- `proj_hidden_dim: 512`, `proj_output_dim: 128`
- `temperature: 0.1`

### ProjectionHead (nn.Module)
- `Linear(d_model, hidden_dim) → BatchNorm1d → ReLU → Linear(hidden_dim, proj_dim)`
- `forward()` applies the MLP then L2-normalizes the output

### ContrastiveObjective (extends BaseSSLObjective)
- Same encoder validation as MAE/JEPA
- `missing_token = None`
- No momentum encoder (SimCLR-style: simplest contrastive, one shared encoder)
- Creates `ProjectionHead`

**Forward pass:**
1. `self.encoder.tokenize(x, obs_mask)` → tokens, padding_mask, token_info
2. `create_observation_mask(...)` called **twice** → ssl_mask_1, ssl_mask_2 (different random masks)
3. View 1: `extract_visible → encode → mean_pool` → pooled_1 (B, d_model)
4. View 2: `extract_visible → encode → mean_pool` → pooled_2 (B, d_model)
5. `projection_head(pooled_1)` → z1, `projection_head(pooled_2)` → z2 (B, proj_dim)
6. NT-Xent loss: concatenate [z1; z2] → (2B, proj_dim), compute cosine similarity matrix / τ, mask diagonal, cross-entropy with positive pair labels

**Mean pooling**: `_mean_pool(encoded, padding_mask)` — sum over valid tokens, divide by count. Same as `apply_pooling(..., 'mean', ...)` from `common.py` but operates on the visible-only encoded output.

**Metrics**: `contrastive_loss`, `ssl_loss` (alias), `contrastive_accuracy` (top-1 retrieval accuracy), `contrastive_pos_similarity`, `contrastive_temperature`, token/view statistics

## Step 5: Register in factory and exports

**Modify** `src/slices/models/pretraining/factory.py`
- Import `JEPAConfig`, `JEPAObjective`, `ContrastiveConfig`, `ContrastiveObjective`
- Add to `SSL_REGISTRY`: `"jepa": JEPAObjective`, `"contrastive": ContrastiveObjective`
- Add to `CONFIG_REGISTRY`: `"jepa": JEPAConfig`, `"contrastive": ContrastiveConfig`

**Modify** `src/slices/models/pretraining/__init__.py`
- Add imports and `__all__` entries for the four new classes

## Step 6: Create config files

**Create** `configs/ssl/jepa.yaml` — JEPA SSL hyperparameters (mask_ratio, predictor dims, momentum, loss_type)

**Create** `configs/ssl/contrastive.yaml` — Contrastive SSL hyperparameters (mask_ratio, projection dims, temperature)

**Create** `configs/pretrain_jepa.yaml` — Entry config: defaults to `data: ricu`, `model: observation_transformer`, `ssl: jepa`. Same training params as `pretrain.yaml` (500 epochs, AdamW 1e-3, warmup_cosine, batch 256, encoder d_model=256).

**Create** `configs/pretrain_contrastive.yaml` — Same structure, `ssl: contrastive`.

## Step 7: Write tests

**Create** `tests/test_jepa_objective.py` — Following patterns from `test_mae_objective.py` and `test_smart_objective.py`:
- Predictor tests (output shape, mask_token learnable)
- Init validation (requires obs encoder, requires pooling=none, target frozen, target is copy)
- Forward (returns loss+metrics, backward works, encoder sees ~25% tokens)
- Momentum (update changes target, schedule works, EMA formula correct)
- Edge cases (very sparse data ~5%, single observation, varying sparsity)
- Gradient flow (to encoder+predictor, not to target)
- Training convergence (loss decreases over 30 steps)
- Factory integration (in registry, build works)
- Loss types (MSE and cosine both work)

**Create** `tests/test_contrastive_objective.py`:
- Projection head tests (output shape, L2 normalized)
- Init validation (requires obs encoder, no target encoder, missing_token=None)
- Forward (returns loss+metrics, backward works, two views encoded)
- NT-Xent tests (perfect alignment → low loss, accuracy metric in [0,1], temperature effect)
- Edge cases (sparse data, single observation)
- Gradient flow (to encoder+projection_head)
- Training convergence
- Factory integration

---

## Verification

1. `uv run pytest tests/test_mae_objective.py -v` — existing MAE tests still pass after masking refactor
2. `uv run pytest tests/test_jepa_objective.py -v` — all JEPA tests pass
3. `uv run pytest tests/test_contrastive_objective.py -v` — all contrastive tests pass
4. `uv run pytest tests/ -v` — full test suite passes
5. Smoke test each pretraining config (verify Hydra resolves correctly):
   - `uv run python scripts/training/pretrain.py --config-name pretrain_jepa training.overfit_batches=2 training.max_epochs=2 --cfg job`
   - `uv run python scripts/training/pretrain.py --config-name pretrain_contrastive training.overfit_batches=2 training.max_epochs=2 --cfg job`
