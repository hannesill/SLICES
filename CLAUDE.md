# SLICES Project Instructions

Guide AI agents here. Keep short: durable constraints, non-obvious experiment context, conventions not cheap to infer from code.

## Target Outcome

Maintain the reproducible research codebase and frozen result corpus behind the SLICES thesis: a controlled MAE vs JEPA vs Contrastive benchmark for sparse ICU time series, with traceable preprocessing, training, export, and evaluation artifacts suitable for thesis defense and paper adaptation.

When tradeoffs arise, preserve benchmark invariants, result traceability, and controlled comparison validity over new features or broader framework ambitions.

## Project Contract

SLICES compares SSL paradigms for sparse, irregular ICU time series under controlled conditions. Main thesis: MAE vs JEPA vs Contrastive. Controlled benchmark work, not open-ended architecture search.

Controlled SSL objectives (`mae`, `jepa`, `contrastive`) share:
- same RICU-derived data pipeline
- same canonical `TransformerEncoder`
- same obs-aware timestep tokenization
- same masking budget where configs intend parity
- same training budget unless ablation

`ts2vec` = temporal-contrastive extension. Extension/context, not core thesis vertex unless scope changed.

`smart` uses `SMARTEncoder` / MART contract + element-wise masking. Appendix/external reference or sanity check, not controlled thesis objective.

Supervised-from-scratch, GRU-D, linear, XGBoost = baselines/contextual comparisons, not SSL paradigms.

## Current Architecture Truth

Canonical controlled encoder: `TransformerEncoder` from `src/slices/models/encoders/transformer.py`, source config `configs/model/transformer.yaml`.

Don't call `ObservationTransformerEncoder` the shared thesis encoder. Alternate/legacy observation-level tokenization.

MAE/JEPA/Contrastive/TS2Vec pretraining:
- encoder must be `transformer`
- `obs_aware=true`
- `pooling=none`
- SSL objectives use `tokenize()` and `encode()` directly

Downstream/supervised:
- default pooling `mean`
- v3 encoder checkpoints rebuild from saved encoder config, then override pooling for finetune aggregation
- obs-aware `TransformerEncoder` handles missingness via obs projection, so missing-token wrapper skipped

## Data And Labels

RICU (R) owns cohort harmonization, unit conversion, concept extraction, hourly binning. Python reads RICU parquet, builds dense tensors, masks, labels.

Benchmark invariants live in `src/slices/constants.py`, not Hydra. Don't move fixed cohort/preprocessing/evaluation constants into sweep configs.

Core fixed choices:
- 24h observation window / min stay
- patient-level splits, 70/15/15
- normalization stats from train split only
- normalize then zero-fill
- preserve observation mask separately from values
- block model-input leakage from `los_hosp`, `los_icu`, `dur_var`

Default extracted labels: `mortality_24h`, `mortality_hospital`, `aki_kdigo`, `los_remaining`. `mortality` exists as optional ICU-mortality extension.

AKI needs creatinine in both baseline/prediction windows; missing labels expected, filter not paper over.

If dataset has `patient_id == stay_id`, patient-level split can't protect repeat-admission leakage; surface limitation.

## Execution Reality

Use `uv run` for Python commands. Never bare `python` or `pip install`.

Work split:
- Mac/local: RICU extraction + Python preprocessing; `data/processed/` parquet is deliverable
- GCP VM `slices-perm`: training experiments
- W&B: source of truth for experiment status/results
- Frozen thesis result corpus: `results/slices-final_thesis-v1/`; W&B project `slices-final`. Do not overwrite or mutate frozen result artifacts unless explicitly asked.

Local `outputs/` doesn't prove which experiments ran on VM. For run status, check W&B or ask user.

Before finalizing code changes, run targeted `uv run pytest ...` or state why verification was not run.

## Config Policy

Hydra configs: experiment knobs — dataset, task, SSL objective, model, training hyperparams, ablations.

`src/slices/constants.py`: benchmark invariants that shouldn't vary between runs.

Adding components, register code + config:
- encoder: `src/slices/models/encoders/factory.py` + `configs/model/`
- SSL objective: `src/slices/models/pretraining/factory.py` + `configs/ssl/`
- task head: `src/slices/models/heads/factory.py`
- task labels: `configs/tasks/` and label builder path if new label type

## Rulesync

Instruction source: `.rulesync/rules/project.md`. Generated adapters: `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/project.mdc`.

Run `.rulesync/generate.sh` after editing. Project intentionally doesn't use rulesync MCP generation.
