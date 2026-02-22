# SLICES Experiment Plan

## Overview

This document defines the full experiment matrix for the SLICES benchmark. The goal is to answer:

> How do the three major SSL paradigm families — reconstruction, self-distillation, and contrastive learning — compare when applied to sparse, irregularly-sampled clinical time series under controlled conditions?

Secondary questions addressed through ablations:

1. Does SSL improve label efficiency over supervised baselines?
2. Do SSL representations transfer across hospital systems?
3. Do SSL paradigms differ in fairness properties?

---

## 1. Experiment Matrix

### 1.1 Independent Variables

| Variable | Levels | Notes |
|----------|--------|-------|
| **Dataset** | MIMIC-IV, eICU, Combined | Combined = pooled pretraining corpus |
| **Paradigm** | MAE (reconstruction), JEPA (self-distillation), Contrastive, Supervised | Supervised = no pretraining baseline |
| **Downstream Task** | mortality_24h, mortality_hospital, aki_kdigo, los_remaining | 3 classification + 1 regression |
| **Seed** | 42, 123, 456 | 3 seeds for statistical significance |

### 1.2 Controlled Variables (Held Constant)

| Variable | Value | Rationale |
|----------|-------|-----------|
| Observation window | 48 hours | Standard in ICU benchmarks |
| Min stay | 48 hours | Ensures full observation window |
| Splits | 70/15/15 train/val/test | Patient-level, no leakage |
| Imputation | Normalize-then-zero-fill | Eliminates imputation as confound |
| Finetuning head | MLP, hidden_dims=[64] | Same head architecture across all paradigms |
| Finetuning LR | 1e-4 | Same optimization for all downstream runs |
| Finetuning epochs | 50 (patience=10) | Same budget |
| Weight decay | 0.01 | Same regularization |

### 1.3 Paradigm-Intrinsic Differences (Documented, Not Controlled)

These differences are inherent to each paradigm and are themselves a contribution:

| Paradigm | Tokenization | Encoder | Rationale |
|----------|-------------|---------|-----------|
| MAE | Observation-level (1 token per observed value) | ObservationTransformer (d=128, L=4, H=8) | Avoids "mostly zeros" on sparse data |
| JEPA | Observation-level (1 token per observed value) | ObservationTransformer (d=128, L=4, H=8) | Predicts in latent space; shares encoder with MAE |
| Contrastive | Observation-level (1 token per observed value) | ObservationTransformer (d=128, L=4, H=8) | Two masked views; shares encoder with MAE |
| Supervised | Timestep-level | Transformer (d=64, L=2, H=4) | Different encoder (no paradigm constraints) |

---

## 2. Training Protocol

### 2.1 Pretraining

| Setting | MAE | JEPA | Contrastive |
|---------|-----|------|-------------|
| Epochs | 500 | 500 | 500 |
| Batch size | 256 | 256 | 256 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Scheduler | Warmup cosine (50 warmup) | Warmup cosine (50 warmup) | Warmup cosine (50 warmup) |
| Mask ratio | 0.5 (random masking) | 0.5 (random masking) | 0.5 (two random masked views) |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Early stopping | Patience=10 on val loss | Patience=10 on val loss | Patience=10 on val loss |

**Budget equalization**: Raw epoch counts differ because paradigms converge at different rates. For fair comparison, report **total gradient steps** and **wall-clock time** alongside epochs. If step counts diverge significantly (>2x), normalize by training the slower paradigm longer.

### 2.2 Downstream Finetuning

All paradigms use identical finetuning protocol:

| Setting | Value |
|---------|-------|
| Protocol | Linear probing (freeze_encoder=true) |
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Scheduler | Cosine decay (eta_min=1e-6) |
| Early stopping | Patience=10 on val metric (AUPRC for classification, MSE for regression) |
| Head | MLP, hidden_dims=[64], dropout=0.1, ReLU |
| Class weighting | None (report as-is; sensitivity analysis if needed) |

**Linear probing rationale**: Isolates representation quality from finetuning optimization. Full finetuning results can be added as an appendix ablation.

### 2.3 Supervised Baseline

| Setting | Value |
|---------|-------|
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Scheduler | Cosine decay |
| Early stopping | Patience=20 on val metric (AUPRC for classification, MSE for regression) |
| Encoder | Transformer (d=64, L=2, H=4), trained end-to-end |
| Head | MLP, hidden_dims=[64], dropout=0.1, ReLU |

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

- Report mean ± std across 3 seeds for all metrics
- **Paired Wilcoxon signed-rank test** between paradigms (paired across seeds and tasks)
- Bonferroni correction for multiple comparisons
- Report effect sizes (Cohen's d) for key comparisons

---

## 4. Main Experiments

### 4.1 Phase 1: Pretraining (9 runs × 3 seeds = 27 runs)

| ID | Dataset | Paradigm | Encoder | Config Override |
|----|---------|----------|---------|-----------------|
| P1 | MIMIC-IV | MAE | ObservationTransformer | ssl=mae model=observation_transformer |
| P2 | MIMIC-IV | JEPA | ObservationTransformer | ssl=jepa model=observation_transformer |
| P3 | MIMIC-IV | Contrastive | ObservationTransformer | ssl=contrastive model=observation_transformer |
| P4 | eICU | MAE | ObservationTransformer | ssl=mae model=observation_transformer |
| P5 | eICU | JEPA | ObservationTransformer | ssl=jepa model=observation_transformer |
| P6 | eICU | Contrastive | ObservationTransformer | ssl=contrastive model=observation_transformer |
| P7 | Combined | MAE | ObservationTransformer | ssl=mae model=observation_transformer |
| P8 | Combined | JEPA | ObservationTransformer | ssl=jepa model=observation_transformer |
| P9 | Combined | Contrastive | ObservationTransformer | ssl=contrastive model=observation_transformer |

### 4.2 Phase 2: Downstream Evaluation (48 runs × 3 seeds = 144 runs)

Each of the 9 pretrained encoders is finetuned on 4 downstream tasks (36 SSL finetuning runs).
Each of the 3 datasets gets a supervised baseline on 4 tasks (12 supervised runs).

**Result Tables** (one per dataset):

#### MIMIC-IV Results

| | mortality_24h | mortality_hospital | aki_kdigo | los_remaining |
|---|---|---|---|---|
| **MAE** | AUPRC ± std | AUPRC ± std | AUPRC ± std | MAE ± std |
| **JEPA** | | | | |
| **Contrastive** | | | | |
| **Supervised** | | | | |

Same table structure for eICU and Combined.

### 4.3 Phase 3: Fairness Evaluation (on all 48 downstream runs)

No additional training. Compute fairness metrics on test predictions from Phase 2.

**Fairness Result Tables** (one per task, per dataset):

#### Example: mortality_24h, MIMIC-IV

| | AUROC gap (sex) | AUROC gap (age) | AUROC gap (race) | Dem. parity diff | Eq. odds diff |
|---|---|---|---|---|---|
| **MAE** | | | | | |
| **JEPA** | | | | | |
| **Contrastive** | | | | | |
| **Supervised** | | | | | |

---

## 5. Ablation Experiments

### 5.1 Label Efficiency (Few-Shot Finetuning)

**Question**: Does SSL improve sample efficiency — how much labeled data is needed to match supervised performance?

**Design**: Finetune with {1%, 5%, 10%, 25%, 50%, 100%} of labeled training data.

| Scope | Tasks | Label Fractions | Runs |
|-------|-------|----------------|------|
| Full sweep | mortality_24h (anchor task) | 1%, 5%, 10%, 25%, 50%, 100% | 4 paradigms × 3 datasets × 6 fractions = 72 |
| Trend check | Remaining 3 tasks | 10%, 100% | 4 paradigms × 3 datasets × 3 tasks × 2 fractions = 72 |
| **Total** | | | **144 runs × 3 seeds = 432 runs** |

**Expected finding**: SSL paradigms should outperform supervised at ≤10% labels. The label fraction at which supervised catches up (crossover point) is a key metric.

**Visualization**: Learning curves (AUPRC vs label fraction) per paradigm, one plot per dataset.

### 5.2 Cross-Dataset Transfer

**Question**: Do SSL representations generalize across hospital systems?

**Design**: Pretrain on source dataset, finetune on target dataset.

| Source → Target | Paradigms | Tasks | Runs |
|-----------------|-----------|-------|------|
| MIMIC-IV → eICU | MAE, JEPA, Contrastive | All 4 | 12 |
| eICU → MIMIC-IV | MAE, JEPA, Contrastive | All 4 | 12 |
| **Total** | | | **24 runs × 3 seeds = 72 runs** |

**Baselines for comparison** (from main experiments, no extra runs):
- In-domain pretraining (pretrain and finetune on same dataset)
- Supervised (no transfer, trained from scratch on target)

**Expected finding**: SSL transfer should outperform supervised-from-scratch on target, especially in low-data regimes. Combined pretraining (Phase 2) should perform best.

**Note**: Supervised has no transfer story — this asymmetry is itself a finding.

### 5.3 Mask Ratio Sensitivity

**Question**: How sensitive are SSL paradigms to the masking ratio?

**Design**: Pretrain with {0.3, 0.5, 0.75} mask ratios on MIMIC-IV, finetune on mortality_24h.

| Mask Ratio | Paradigms | Dataset | Task | Runs |
|------------|-----------|---------|------|------|
| 0.3 | MAE, JEPA, Contrastive | MIMIC-IV | mortality_24h | 3 |
| 0.5 | MAE, JEPA, Contrastive | MIMIC-IV | mortality_24h | 3 (from main experiments) |
| 0.75 | MAE, JEPA, Contrastive | MIMIC-IV | mortality_24h | 3 |
| **Total** | | | | **6 new pretraining + 6 new finetuning = 12 runs × 3 seeds = 36 runs** |

**Note**: The 0.5 runs are reused from the main experiments (P1–P3). Only 0.3 and 0.75 require additional pretraining.

**Visualization**: Bar chart or line plot of AUPRC vs mask ratio, one line per paradigm.

---

## 6. Run Summary

| Phase | Description | Runs | × 3 seeds | Cumulative |
|-------|-------------|------|-----------|------------|
| Phase 1 | Pretraining | 9 | 27 | 27 |
| Phase 2 | Downstream finetuning | 48 | 144 | 171 |
| Phase 3 | Fairness evaluation | 0 (eval only) | 0 | 171 |
| Ablation 5.1 | Label efficiency | 144 | 432 | 603 |
| Ablation 5.2 | Cross-dataset transfer | 24 | 72 | 675 |
| Ablation 5.3 | Mask ratio sensitivity | 12 | 36 | 711 |
| **Total** | | **237** | **711** | |

---

## 7. Execution Order

Priority ordering for incremental results and early debugging:

### Sprint 1: Sanity Check (4 runs)
1. MIMIC-IV, all 4 paradigms (MAE, JEPA, Contrastive, Supervised), mortality_24h only, seed=42
2. Verify: training converges, metrics are reasonable, pipeline end-to-end works
3. Check gradient step counts across paradigms — adjust epoch budgets if needed

### Sprint 2: First Full Dataset (16 runs)
4. MIMIC-IV, all 4 paradigms × 4 tasks, seed=42
5. Produces first complete results table for one dataset
6. Write preliminary results section

### Sprint 3: Generalization (16 runs)
7. eICU, all 4 paradigms × 4 tasks, seed=42
8. Cross-dataset comparison possible

### Sprint 4: Scaling (16 runs)
9. Combined dataset, all 4 paradigms × 4 tasks, seed=42
10. All three main result tables complete (single seed)

### Sprint 5: Statistical Robustness (96 runs)
11. Seeds 123 and 456 for all Sprint 2–4 runs
12. Compute mean ± std, run statistical tests

### Sprint 6: Label Efficiency Ablation (432 runs)
13. Full sweep on mortality_24h anchor
14. Trend check on remaining tasks
15. Generate learning curve plots

### Sprint 7: Transfer Ablation (72 runs)
16. Cross-dataset transfer experiments
17. Compare against in-domain baselines

### Sprint 8: Fairness Analysis (0 extra runs)
18. Compute fairness metrics on all Phase 2 test predictions
19. Enable `eval.fairness.enabled=true` and rerun evaluation
20. Generate fairness tables and disparity plots

---

## 8. W&B Tracking

### Project Structure

All runs logged to W&B project `slices`.

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
| `paradigm:mae` / `paradigm:jepa` / `paradigm:contrastive` / `paradigm:supervised` | SSL family |
| `task:mortality_24h` / etc. | Downstream task |
| `ablation:label-efficiency` / `ablation:transfer` | Ablation type |
| `sprint:N` | Execution sprint |
| `seed:42` / `seed:123` / `seed:456` | Random seed |

### Key Metrics to Log

| Phase | Metrics |
|-------|---------|
| Pretraining | train_loss, val_loss, gradient_steps, wall_clock_time |
| Finetuning | val_auroc, val_auprc, test_auroc, test_auprc, test_brier, test_ece |
| Regression | val_mae, test_mae, test_mse, test_r2 |
| Fairness | fairness/auroc_gap_sex, fairness/dem_parity_diff, fairness/eq_odds_diff |

---

## 9. Expected Outputs

### Tables for Paper

1. **Table 1**: Main results — AUPRC (mean ± std) for each paradigm × dataset × task (3 tables or 1 large table)
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

- Full finetuning results (unfreeze encoder) if time permits
- Per-seed results for transparency
- Training curves (loss vs steps) for all pretraining runs
- Hyperparameter sensitivity (if reviewer requests)

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Contrastive paradigm not yet implemented | Implement before Sprint 1; if delayed, run MAE + JEPA + Supervised first |
| eICU lacks race/ethnicity data | Fairness analysis for race is MIMIC-only; document as limitation |
| Phenotyping task is MIMIC-only | Excluded from main matrix; can add as MIMIC-specific appendix |
| Unequal training budgets across paradigms | Track gradient steps; normalize if >2x difference |
| Class imbalance affects metrics | Use AUPRC as primary; report class ratios; sensitivity analysis with class_weight="balanced" |
| Compute budget exceeded | Sprint ordering ensures usable results at each checkpoint |
| Reviewer requests 5 seeds | Add 2 seeds only to contested comparisons (minor revision) |
