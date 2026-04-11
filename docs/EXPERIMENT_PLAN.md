# SLICES Experiment Plan

## Overview

This document defines the full experiment matrix for the SLICES benchmark. The goal is to answer:

> How do the three major SSL paradigm families — reconstruction, self-distillation, and contrastive learning — compare when applied to sparse, irregularly-sampled clinical time series under controlled conditions?

Secondary questions addressed through ablations:

1. Does SSL improve label efficiency over supervised baselines?
2. Do SSL representations transfer across hospital systems?
3. Do SSL paradigms differ in fairness properties?
4. Does increasing model capacity widen the SSL-supervised gap on low-label mortality prediction?
5. Does a temporal contrastive objective (TS2Vec) materially change the contrastive-family conclusion?

---

## 1. Experiment Matrix

### 1.1 Independent Variables

| Variable | Levels | Notes |
|----------|--------|-------|
| **Dataset** | MIMIC-IV, eICU, Combined | Combined = pooled pretraining corpus |
| **Paradigm** | MAE (reconstruction), JEPA (self-distillation), Contrastive, Supervised | Supervised = no pretraining baseline; final thesis reporting also includes GRU-D and XGBoost as contextual baselines |
| **Downstream Task** | mortality_24h, mortality_hospital, aki_kdigo, los_remaining | 3 classification + 1 regression |
| **Seed** | 42, 123, 456, 789, 1011 | Core matrix starts with 3 seeds; Sprint `10` extends the thesis corpus to 5 seeds, and Sprints `7p`, `11`, `12`, and `13` use the same 5-seed footing |

**Formal thesis scope**: The controlled comparison remains the three-way MAE/JEPA/Contrastive benchmark plus supervised. The final thesis matrix also includes GRU-D and XGBoost as contextual baselines, canonicalized in Sprint `11` rather than duplicated across the early SSL/supervised sprints. Two additional thesis experiments are now tracked explicitly: Sprint `7p` is a focused capacity study (larger MAE + supervised encoders on MIIV mortality_24h), and Sprint `13` is a temporal-contrastive extension (TS2Vec) run under the same downstream A/B protocol. Sprint `12` (SMART) remains appendix-only because it changes the encoder family.

### 1.2 Controlled Variables (Held Constant)

| Variable | Value | Rationale |
|----------|-------|-----------|
| Observation window | 48 hours | Standard in ICU benchmarks |
| AKI prediction window | Hours 48–72 (forward-looking) | Prevents leakage from creatinine values used in label construction |
| Min stay | 48 hours (72 hours for AKI) | Ensures full observation + prediction window |
| Splits | 70/15/15 train/val/test | Patient-level, no leakage |
| Imputation | Normalize-then-zero-fill | Eliminates imputation as confound |
| Finetuning head | MLP, hidden_dims=[64] | Same head architecture across all paradigms |
| Precision | fp32 | Avoids bf16 numerical issues with small model |
| Weight decay | 0.05 | Same regularization across all downstream runs |

### 1.3 Shared Encoder Architecture

All three SSL paradigms share the same encoder architecture, removing tokenization as a confounding variable:

| Paradigm | Tokenization | Encoder | Masking | Rationale |
|----------|-------------|---------|---------|-----------|
| MAE | Timestep-level (obs-aware) | Transformer (d=64, L=2, H=4, obs_aware=True) | Random timestep masking | Reconstruct masked timestep features |
| JEPA | Timestep-level (obs-aware) | Transformer (d=64, L=2, H=4, obs_aware=True) | Block masking (3 contiguous segments) | Predict in latent space; block masking prevents trivial interpolation |
| Contrastive | Timestep-level (obs-aware) | Transformer (d=64, L=2, H=4, obs_aware=True) | Two complementary masked views (instance mode, zero overlap) | Instance-level NT-Xent: mean-pool each view → align sequence embeddings |
| Supervised | Timestep-level (obs-aware) | Transformer (d=64, L=2, H=4, obs_aware=True) | N/A | Same encoder as SSL for fair comparison |

**Obs-aware tokenization**: Each timestep token is produced by an MLP projection of `concat(values, obs_mask)` → `d_model`. This encodes both the observed values and the missingness pattern, avoiding the "mostly zeros" problem of naively feeding sparse timestep vectors through a linear projection. With ~22 observed features per timestep on average, the MLP maps ~44 non-zero inputs (22 values + 22 mask bits) to 64-dim tokens — no information bottleneck.

**Design rationale**: An earlier observation-level design (1 token per observed value, using `ObservationTransformerEncoder`) produced ~1660 tokens per sample at 20% observation density, causing O(N²) attention to exceed L4 GPU memory. Timestep-level tokenization reduces this to 48 tokens (~35× reduction) while preserving all information through the obs-aware MLP. This also strengthens the controlled comparison by making all SSL paradigms share identical tokenization.

---

## 2. Training Protocol

### 2.1 Pretraining

| Setting | MAE | JEPA | Contrastive |
|---------|-----|------|-------------|
| Epochs | 100 | 100 | 100 |
| Batch size | 256 | 256 | 256 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Scheduler | Warmup cosine (10 warmup) | Warmup cosine (10 warmup) | Warmup cosine (10 warmup) |
| Mask ratio | 0.5 (random masking) | 0.5 (block masking) | 0.5 (complementary masks, zero overlap) |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Early stopping | None (fixed schedule) | None (fixed schedule) | None (fixed schedule) |
| Checkpoint | Last epoch + best val loss | Last epoch + best val loss | Last epoch + best val loss |

**Fixed-schedule training**: All paradigms train for the full epoch budget without early stopping, following the convention in self-distillation literature (I-JEPA, DINO, BYOL) where validation loss is unreliable as a stopping criterion — JEPA's val loss rises as representations get richer, not worse. Fixed schedules ensure identical gradient step budgets across paradigms. The cosine LR schedule decays to eta_min=1e-6, providing natural convergence. Both last-epoch and best-val-loss encoders are saved; the last-epoch encoder is used for all primary results, with best-val-loss as a robustness check in the appendix.

### 2.2 Downstream Evaluation (Two Protocols)

SSL encoders are evaluated under **two** protocols to answer different questions:

#### Protocol A: Linear Probing (primary SSL comparison)

Isolates representation quality — the only variable is the SSL pretraining objective.

| Setting | Value |
|---------|-------|
| Protocol | Linear probing (freeze_encoder=true) |
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Scheduler | Cosine decay (eta_min=1e-6) |
| Early stopping | Patience=10 on val AUPRC (classification) or val MAE (regression) |
| Head | MLP, hidden_dims=[64], dropout=0.3, ReLU |
| Gradient clipping | 1.0 |
| Label smoothing | 0.1 |
| Weight decay | 0.05 |
| Class weighting | sqrt(balanced) — square root of inverse frequency |

**Rationale**: Linear probing answers "Which SSL objective produces the best representations?" by preventing the encoder from adapting to the downstream task.

#### Protocol B: Full Finetuning (practical utility)

Measures how useful SSL pretraining is as weight initialization.

| Setting | Value |
|---------|-------|
| Protocol | Full finetuning (freeze_encoder=false) |
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Scheduler | Cosine decay (eta_min=1e-6) |
| Early stopping | Patience=10 on val AUPRC (classification) or val MAE (regression) |
| Head | MLP, hidden_dims=[64], dropout=0.3, ReLU |
| Gradient clipping | 1.0 |
| Label smoothing | 0.1 |
| Weight decay | 0.05 |
| Class weighting | sqrt(balanced) — square root of inverse frequency |

**Rationale**: Full finetuning answers "Which SSL objective is most useful in practice as initialization?" Results may diverge from linear probing — an encoder with mediocre frozen representations can still provide excellent initialization.

**Regularization rationale**: Early Sprint 1 runs showed severe overfitting across all paradigms (train AUROC ~0.98 by epoch 1–2, monotonically degrading val metrics). The regularization suite — reduced LR (1e-3 → 3e-4), increased dropout (0.1 → 0.3), higher weight decay (0.01 → 0.05), sqrt class weights, and label smoothing — was introduced to address this. These settings apply uniformly to all paradigms and the supervised baseline, preserving the controlled comparison.

### 2.3 Supervised Baseline

| Setting | Value |
|---------|-------|
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Scheduler | Cosine decay (eta_min=1e-6) |
| Early stopping | Patience=10 on val AUPRC (classification) or val MAE (regression) |
| Encoder | Transformer (d=64, L=2, H=4, obs_aware=True), trained end-to-end |
| Head | MLP, hidden_dims=[64], dropout=0.3, ReLU |
| Gradient clipping | 1.0 |
| Label smoothing | 0.1 |
| Weight decay | 0.05 |
| Class weighting | sqrt(balanced) — square root of inverse frequency |

**Same-architecture rationale**: The supervised baseline uses the identical encoder architecture and tokenization as the SSL paradigms to eliminate model capacity as a confounding variable. The only difference is the training procedure: supervised trains end-to-end on labeled data from random initialization, while SSL paradigms pretrain on unlabeled data and then evaluate via Protocol A (linear probe) and Protocol B (full finetune). Comparing supervised vs. Protocol B isolates the effect of SSL initialization; comparing SSL paradigms via Protocol A isolates representation quality.

---

## 3. Evaluation Metrics

### 3.1 Performance Metrics

| Task | Type | Primary Metric | Secondary Metrics |
|------|------|---------------|-------------------|
| mortality_24h | Binary classification | AUPRC | AUROC, Brier score, ECE |
| mortality_hospital | Binary classification | AUPRC | AUROC, Brier score, ECE |
| aki_kdigo | Binary classification | AUPRC | AUROC |
| los_remaining | Regression | MAE | MSE, R² |

**AUPRC as primary** for classification: ICU tasks are heavily imbalanced; AUPRC is more informative than AUROC in this regime.

### 3.2 Fairness Metrics

Computed on test set predictions for all classification tasks:

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| Per-group AUROC | AUROC computed per subgroup | Report gap (max - min) |
| Per-group AUPRC | AUPRC computed per subgroup | Report gap |
| Demographic parity difference | max\|P(Ŷ=1\|A=a) - P(Ŷ=1\|A=b)\| | Lower is better |
| Equalized odds difference | max(TPR gap, FPR gap) across groups | Lower is better |
| Disparate impact ratio | min_rate / max_rate | <0.8 = adverse impact |

### 3.3 Protected Attributes

| Attribute | Available In | Groups | Notes |
|-----------|-------------|--------|-------|
| **Sex** | MIMIC-IV, eICU | M, F | Primary; available in both datasets |
| **Age group** | MIMIC-IV, eICU | 18-44, 45-64, 65-79, 80+ | Primary; 4-bin split for granular fairness analysis |
| **Race/Ethnicity** | MIMIC-IV only | White, Black, Hispanic, Asian, Other | Secondary; MIMIC-only analysis |

Min subgroup size: 50 patients. Groups below threshold are excluded from fairness metrics.

### 3.4 Statistical Testing

- Report mean ± std across 5 seeds for all metrics
- **Paired Wilcoxon signed-rank test** between paradigms (paired across seeds and tasks)
- Bonferroni correction for multiple comparisons
- Report effect sizes (Cohen's d) for key comparisons

---

## 4. Main Experiments

### 4.1 Phase 1: Pretraining (9 runs × 3 seeds = 27 runs)

| ID | Dataset | Paradigm | Encoder | Config Override |
|----|---------|----------|---------|-----------------|
| P1 | MIMIC-IV | MAE | Transformer (obs_aware) | ssl=mae |
| P2 | MIMIC-IV | JEPA | Transformer (obs_aware) | ssl=jepa |
| P3 | MIMIC-IV | Contrastive | Transformer (obs_aware) | ssl=contrastive |
| P4 | eICU | MAE | Transformer (obs_aware) | ssl=mae dataset=eicu |
| P5 | eICU | JEPA | Transformer (obs_aware) | ssl=jepa dataset=eicu |
| P6 | eICU | Contrastive | Transformer (obs_aware) | ssl=contrastive dataset=eicu |
| P7 | Combined | MAE | Transformer (obs_aware) | ssl=mae dataset=combined |
| P8 | Combined | JEPA | Transformer (obs_aware) | ssl=jepa dataset=combined |
| P9 | Combined | Contrastive | Transformer (obs_aware) | ssl=contrastive dataset=combined |

### 4.2 Phase 2: Downstream Evaluation (84 runs × 3 seeds = 252 runs)

Each of the 9 pretrained encoders is evaluated on 4 downstream tasks under **both** protocols:
- Protocol A (linear probe): 9 encoders × 4 tasks = 36 runs
- Protocol B (full finetune): 9 encoders × 4 tasks = 36 runs
- Supervised baseline: 3 datasets × 4 tasks = 12 runs

**Result table template** (one per dataset, two sub-tables per dataset):

| | mortality_24h | mortality_hospital | aki_kdigo | los_remaining |
|---|---|---|---|---|
| **MAE** | AUPRC ± std | AUPRC ± std | AUPRC ± std | MAE ± std |
| **JEPA** | | | | |
| **Contrastive** | | | | |
| **Supervised** | (Protocol B only) | | | |

### 4.3 Phase 3: Fairness Evaluation (on all downstream runs)

No additional training. Compute fairness metrics on test predictions from Phase 2.

---

## 5. Ablation Experiments

### 5.1 Label Efficiency (Few-Shot Finetuning)

**Question**: Does SSL improve sample efficiency — how much labeled data is needed to match supervised performance?

**Design**: Evaluate SSL under both downstream protocols plus the supervised Transformer baseline at low label fractions {1%, 5%, 10%, 25%, 50%}. The 100% endpoints are inherited from the main full-label runs rather than relaunched here.

| Scope | Tasks | Label Fractions | Runs |
|-------|-------|----------------|------|
| Full sweep | mortality_24h (anchor task) | 1%, 5%, 10%, 25%, 50% | (3 SSL paradigms × 2 protocols + supervised) × 3 datasets × 5 fractions = 105 runs/seed |
| Trend check | Remaining 3 tasks | 10% | (3 SSL paradigms × 2 protocols + supervised) × 3 datasets × 3 tasks = 63 runs/seed |
| **Total** | | | **168 runs/seed × 3 seeds = 504 runs** |

**Visualization**: Learning curves (AUPRC vs label fraction) per paradigm, one plot per dataset.

### 5.2 Cross-Dataset Transfer

**Question**: Do SSL representations generalize across hospital systems?

**Design**: Pretrain on source dataset, finetune on target dataset.

| Source → Target | Paradigms | Tasks | Runs |
|-----------------|-----------|-------|------|
| MIMIC-IV → eICU | MAE, JEPA, Contrastive | All 4 | 12 |
| eICU → MIMIC-IV | MAE, JEPA, Contrastive | All 4 | 12 |
| **Total** | | | **24 runs × 3 seeds = 72 runs** |

**Baselines** (from main experiments, no extra runs): in-domain pretraining, supervised from scratch.

### 5.3 Learning Rate Sensitivity

**Question**: Are paradigm rankings robust to the shared learning rate?

**Design**: Pretrain with LR ∈ {2e-4, 5e-4, 1e-3, 2e-3} on MIMIC-IV, finetune on mortality_24h. LR=1e-3 reused from main experiments.

**Total**: 9 new pretraining + 9 new finetuning = 18 runs × 3 seeds = 54 runs

### 5.4 Mask Ratio Sensitivity

**Question**: How sensitive are SSL paradigms to the masking ratio?

**Design**: Pretrain with {0.3, 0.5, 0.75} mask ratios on MIMIC-IV, finetune on mortality_24h. 0.5 reused from main experiments.

**Total**: 6 new pretraining + 6 new finetuning = 12 runs × 3 seeds = 36 runs

### 5.5 Focused Capacity Study (Sprint 7p)

**Question**: Are the modest SSL gains on low-label mortality prediction capacity-limited, or does supervised scale similarly when model size increases?

**Design**: MIIV only, mortality_24h only, MAE + supervised only. Compare two larger encoder sizes (128d/4L and 256d/4L) against the inherited default-size baseline (64d/2L) from Sprints `6` and `10`, at label fractions {0.01, 0.1, 0.5}, across 5 seeds.

**Total**: 2 sizes × [1 pretrain + 3 Protocol B finetunes + 3 Protocol A probes + 3 supervised] = 20 runs/seed × 5 seeds = **100 runs**

### 5.6 Temporal Contrastive Extension (Sprint 13)

**Question**: Does giving the contrastive family its natural temporal objective and augmentations overturn the conclusion from the instance-level contrastive baseline?

**Design**: TS2Vec on all 3 datasets, all 4 downstream tasks, both Protocol A and Protocol B, 5 seeds. Same base encoder family and matched training budget as the controlled comparison.

**Total**: 1 paradigm × 3 datasets × 5 seeds = 15 pretrains + 120 downstream probe/finetune runs = **135 runs**

---

## 6. Run Summary

| Phase | Description | Unit Runs | Planned Runs | Cumulative |
|-------|-------------|-----------|--------------|------------|
| Phase 1 | Pretraining | 9 | 27 | 27 |
| Phase 2 | Downstream (linear probe + full finetune + supervised) | 84 | 252 | 279 |
| Phase 3 | Fairness evaluation | 0 (eval only) | 0 | 279 |
| Ablation 5.1 | Label efficiency | 168 | 504 | 783 |
| Ablation 5.2 | Cross-dataset transfer | 24 | 72 | 855 |
| Ablation 5.3 | LR sensitivity | 18 | 54 | 909 |
| Ablation 5.4 | Mask ratio sensitivity | 12 | 36 | 945 |
| Sprint 7p | Focused capacity study (MAE + supervised, 5 seeds) | — | 100 | 1045 |
| Sprint 10 | Seeds 789, 1011 for Sprints 1–8 scope (SSL + supervised) | — | 630 | 1675 |
| Sprint 11 | Classical baselines (XGBoost + GRU-D), 5 seeds | — | 360 | 2035 |
| Sprint 13 | TS2Vec temporal contrastive extension, 5 seeds | — | 135 | 2170 |
| Sprint 12 | SMART external SSL reference, 5 seeds | — | 135 | 2305 |
| **Total** | | | **2305** | |

**Planning note**: The formal thesis corpus through Sprint `13` totals **2170** runs. Sprint `12` adds **135** appendix-only SMART reference runs, bringing the full run budget to **2305**.

---

## 7. Sprint Execution Order

| Sprint | Description | Runs | Status |
|--------|-------------|------|--------|
| 1 | Pipeline validation (MIMIC, Protocol B + supervised) | 19 | COMPLETE |
| 1b | LR sensitivity (MIMIC, mortality_24h) | 18 | COMPLETE |
| 1c | Mask ratio sensitivity (MIMIC, mortality_24h) | 12 | COMPLETE |
| 2 | MIMIC Protocol A (completes MIMIC results table) | 12 | COMPLETE |
| 3 | eICU (both protocols + supervised) | 31 | COMPLETE |
| 4 | Combined (both protocols + supervised) | 31 | COMPLETE |
| 5 | Seeds 123, 456 for Sprints 1–4 | 186 | COMPLETE |
| 6 | Label efficiency ablation | 504 | COMPLETE |
| 7 | Cross-dataset transfer | 72 | COMPLETE |
| 8 | LR + mask ratio extra seeds | 60 | COMPLETE |
| 10 | Seeds 789, 1011 for Sprints 1–8 scope (SSL + supervised) | 630 | TODO |
| 7p | Focused capacity study (MIIV mortality_24h, MAE + supervised, 5 seeds) | 100 | TODO |
| 11 | Classical baselines (XGBoost + GRU-D), 5 seeds | 360 | TODO |
| 13 | TS2Vec temporal contrastive extension, both protocols, 5 seeds | 135 | TODO |
| 12 | SMART external SSL reference, 5 seeds | 135 | TODO |
| 9 | Fairness (eval only, runs last after all training) | 0 | TODO |

For detailed sprint outcomes, see `EXPERIMENT_RESULTS.md`.

### Upcoming: Sprint 10 — Extra Seeds 789, 1011 (630 runs)

**Motivation**: Go from 3 seeds to 5 seeds for all SSL and supervised experiments, improving statistical power (4 d.f. instead of 2 for variance estimates).

**Scope**: Seeds 789 and 1011 for the full Sprints 1–8 scope. Classical baselines are not duplicated here; Sprint `11` remains their single canonical family.

| Scope | Equivalent Sprint | Runs |
|-------|-------------------|------|
| Core experiments (pretrain + probe + finetune + supervised) | Sprints 1–4 via Sprint 5 | 186 |
| Label efficiency ablation (SSL + supervised) | Sprint 6 | 336 |
| Cross-dataset transfer | Sprint 7 | 48 |
| LR + mask ratio HP ablations | Sprint 8 | 60 |
| **Total** | | **630** |

**Breakdown**: 48 pretrain (~30 min each, the bottleneck) + 510 finetune/probe (~3–5 min each) + 72 supervised (~3–5 min each).

**Execution**: `uv run python scripts/run_experiments.py run --sprint 10 --parallel 4` (limited by GPU-bound pretraining)

### Upcoming: Sprint 7p — Focused Capacity Study (100 runs)

**Motivation**: The main benchmark suggests SSL gains are real but modest. Sprint `7p` tests whether that gap widens once model capacity increases, while keeping the question narrow enough to stay interpretable: MAE vs supervised, MIIV only, mortality_24h only, and low-label fractions where representation quality matters most.

**Matrix**: 5 seeds × 2 model sizes × [1 pretrain + 3 Protocol A probes + 3 Protocol B finetunes + 3 supervised baselines] = **100 runs**

**Comparison set**: Default-size baselines are inherited from Sprint `6` (seeds 42/123/456) and Sprint `10` (seeds 789/1011), so Sprint `7p` only launches the larger-capacity runs.

**Execution**: `uv run python scripts/run_experiments.py run --sprint 7p --parallel 4`

### Upcoming: Sprint 11 — Classical Baselines (360 runs)

**Motivation**: Provide non-neural (XGBoost) and missingness-native neural (GRU-D) reference points. Without these, SSL AUROC numbers exist in a vacuum. XGBoost with engineered features is the perennially competitive clinical baseline (BAT 2025); GRU-D (Che et al. 2018) handles missing data natively via learnable decay.

**Canonicalization note**: These baselines are part of the final thesis matrix, but they are launched only here. Earlier sprints intentionally contain only the controlled SSL + supervised benchmark so the classical baselines are reported once as a separate contextual comparison family.

**Matrix**: 2 baselines × 3 datasets × 4 tasks × 5 seeds = 120 full-label runs + 240 label-efficiency runs = **360 runs**

**XGBoost**: 8 summary statistics per input feature (mean, std, min, max, first, last, obs_count, obs_fraction) → 896 tabular features. Uses `ICUDataModule` for identical data splits.

**GRU-D**: d_model=64, 1 GRU layer, same downstream training protocol as supervised Transformer (lr=3e-4, wd=0.05, patience=10, label smoothing, class weights). Trains on GPU via `supervised.py --config-name gru_d`.

**Key comparisons enabled**:
1. XGBoost/GRU-D vs SSL Protocol B — absolute performance context
2. XGBoost/GRU-D vs supervised Transformer — is the Transformer architecture justified?
3. XGBoost cannot transfer across datasets — highlights SSL's cross-dataset transfer capability (Sprint 7)
4. XGBoost/GRU-D on label efficiency curves — direct comparison with SSL learning curves

**Execution**: `uv run python scripts/run_experiments.py run --sprint 11 --parallel 12` (all runs independent, no pretraining dependency)

### Upcoming: Sprint 13 — TS2Vec Temporal Contrastive Extension (135 runs)

**Motivation**: Sprint `13` addresses the strongest foreseeable reviewer objection to the contrastive-family result: "instance-level contrastive was set up to fail." TS2Vec gives contrastive learning its natural temporal objective and augmentations while preserving the rest of the benchmark machinery.

**Scope**: 3 datasets × 4 tasks × 2 protocols × 5 seeds, with matched downstream Protocol A/B evaluation and the same base encoder family.

**Matrix**: 15 pretrains + 120 downstream probe/finetune runs = **135 runs**

**Interpretation**: Sprint `13` is a formal thesis extension, not a replacement for the controlled MAE/JEPA/Contrastive triangle. It tests whether a better-instantiated contrastive family changes the conclusion.

**Execution**: `uv run python scripts/run_experiments.py run --sprint 13 --parallel 4`

### Upcoming: Sprint 12 — SMART External SSL Reference (135 runs)

**Motivation**: SMART (NeurIPS 2024) is the most recent published SSL method for ICU time series. Including it provides an external SOTA reference point beyond the controlled three-way comparison. Addresses reviewer concern #7 ("no comparison to published clinical SSL methods").

**Important**: SMART uses a different encoder architecture (MART, d_model=32, per-variable query tokens, element-wise masking) and is therefore NOT part of the controlled comparison. Results go in the appendix as an external reference, not in the main results table.

**Matrix**: 1 paradigm × 3 datasets × 4 tasks × 2 protocols × 5 seeds = 120 finetune + 15 pretrain = **135 runs**

**Execution**: `uv run python scripts/run_experiments.py run --sprint 12 --parallel 4`

### Upcoming: Sprint 9 — Fairness Analysis (0 extra runs)

**Runs last**, after all training sprints are complete. Zero additional training — pure evaluation on existing test predictions.

1. Compute fairness metrics on all finetune/supervised test predictions from Sprints 1–5, 7p, 10, 12, and 13
2. Enable `eval.fairness.enabled=true` and rerun evaluation
3. Generate fairness tables and disparity plots
4. Protected attributes: sex (all datasets), age group (all), race/ethnicity (MIMIC-IV only)
5. Sprint `11` classical baselines are not part of the default standalone fairness sweep because `evaluate_fairness.py` currently targets `finetune` and `supervised` phases only

---

## 8. W&B Tracking

### Project Structure

Legacy exploratory runs remain in W&B project `slices`. Final thesis reruns should log to a separate project such as `slices-thesis` and use a single revision tag such as `thesis-v1`.

### Run Naming Convention

```
{phase}_{dataset}_{paradigm}_{task}_seed{seed}
```

Examples:
- `pretrain_mimic_mae_seed42`
- `finetune_eicu_jepa_mortality24h_seed123`
- `supervised_combined_aki_kdigo_seed456`
- `ablation-label_mimic_mae_mortality24h_frac10_seed42`
- `ablation-transfer_mimic2eicu_contrastive_aki_seed42`

### Tags

| Tag | Purpose |
|-----|---------|
| `phase:pretrain` / `phase:finetune` / `phase:supervised` | Training phase |
| `dataset:mimic` / `dataset:eicu` / `dataset:combined` | Dataset |
| `paradigm:mae` / `paradigm:jepa` / `paradigm:contrastive` / `paradigm:supervised` / `paradigm:xgboost` / `paradigm:gru_d` | Paradigm |
| `task:mortality_24h` / etc. | Downstream task |
| `ablation:label-efficiency` / `ablation:transfer` | Ablation type |
| `sprint:N` | Execution sprint |
| `seed:42` / `seed:123` / `seed:456` / `seed:789` / `seed:1011` | Random seed |

### Key Metrics to Log

| Phase | Metrics |
|-------|---------|
| Pretraining | train_loss, val_loss, gradient_steps, wall_clock_time |
| Finetuning | val_auroc, val_auprc, test_auroc, test_auprc, test_brier, test_ece |
| Regression | val_mae, test_mae, test_mse, test_r2 |
| Fairness | fairness/auroc_gap_sex, fairness/dem_parity_diff, fairness/eq_odds_diff |

### Baseline Inheritance Across Sprints

Later sprints reuse runs from earlier sprints as comparison baselines. To enable filtering all relevant runs for a sprint in a single W&B query, baseline runs are tagged with later sprint tags using `run_experiments.py tag --sprint N`.

| Sprint | Inherits From | What's Inherited |
|--------|--------------|------------------|
| **1** | — | Self-contained (root sprint) |
| **1b** | 1 | Default-LR pretrain+finetune (mortality_24h) + supervised baseline |
| **1c** | 1 | Default-mask-ratio pretrain+finetune (mortality_24h) + supervised baseline |
| **2** | 1 | All Sprint 1 runs — completes the MIMIC results table |
| **3** | — | Self-contained (eICU has own pretrains + supervised) |
| **4** | — | Self-contained (Combined has own pretrains + supervised) |
| **5** | 1, 2, 3, 4 | All seed=42 runs (to aggregate with seeds 123, 456) |
| **6** | 1, 2, 3, 4, 5 | All fraction=1.0 finetune + supervised (learning curve right endpoints) |
| **7** | 1, 3, 5 | In-domain finetunes + supervised for MIMIC and eICU |
| **7p** | 6, 10 | Default-size label-efficiency baselines for seeds 42/123/456/789/1011 |
| **8** | 1, 1b, 1c, 5 | Seed=42 ablation runs + default-HP data points for seeds 123, 456 |
| **10** | — | Self-contained (new pretrains for seeds 789, 1011) |
| **11** | — | Self-contained (classical baselines, 5 seeds) |
| **13** | — | Self-contained (TS2Vec pretrains + both protocols, 5 seeds) |
| **12** | — | Self-contained (SMART pretrains + finetune, 5 seeds) |
| **9** | All | Evaluates test predictions from all training sprints (runs last) |

**Usage**: `uv run python scripts/run_experiments.py tag --sprint N` (idempotent).

---

## 9. Expected Outputs

### Tables for Paper

1. **Table 1**: Main results — AUPRC (mean ± std) for each paradigm × dataset × task
2. **Table 2**: Statistical significance — p-values for pairwise paradigm comparisons
3. **Table 3**: Fairness — AUROC gap and equalized odds difference per paradigm per protected attribute
4. **Table 4**: Cross-dataset transfer — AUPRC for source→target vs in-domain vs supervised

### Figures for Paper

1. **Figure 1**: Architecture diagram showing paradigm-intrinsic tokenization differences
2. **Figure 2**: Learning curves — AUPRC vs label fraction per paradigm (one subplot per dataset)
3. **Figure 3**: Radar/spider chart — paradigm comparison across all tasks (one per dataset)
4. **Figure 4**: Fairness disparity heatmap — paradigm × protected attribute × task
5. **Figure 5**: Transfer gap — bar chart of (transfer AUPRC - supervised AUPRC) per paradigm

### Appendix

- Per-seed results for transparency
- Training curves (loss vs steps) for all pretraining runs
- Hyperparameter sensitivity (if reviewer requests)
- Literature comparison table (see `docs/LIT_COMP.md`)

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| eICU lacks race/ethnicity data | Fairness analysis for race is MIMIC-only; document as limitation |
| Unequal training budgets across paradigms | Track gradient steps; normalize if >2x difference |
| Class imbalance affects metrics | Use AUPRC as primary metric; sqrt(balanced) class weighting; early stopping on val AUPRC (not val loss); report class ratios |
| Downstream overfitting | Addressed with regularization suite: dropout 0.3, LR 3e-4, weight decay 0.05, sqrt class weights, label smoothing 0.1 — applied uniformly to all paradigms |
| Compute budget exceeded | Sprint ordering ensures usable results at each checkpoint |
| Shared hyperparameters unfair to one paradigm | LR sensitivity (Sprint 1b) and mask ratio sensitivity (Sprint 1c) validate that rankings are robust to shared hyperparameter choices |
| Reviewer requests more seeds | Sprint `10` expands the main matrix to 5 seeds, and Sprints `7p`/`11`/`12`/`13` keep the same 4 d.f. footing for variance estimates |
| AKI label leakage | Fixed: forward-looking prediction window (hours 48–72) prevents model from seeing creatinine values used for KDIGO label construction |
