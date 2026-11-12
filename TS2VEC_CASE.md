# A complete TS2Vec-style extension case

This case connects an existing non-core objective to configuration, pretraining,
encoder export, both downstream protocols, comparison checks, and reporting.
It demonstrates reuse of SLICES components. It does not introduce a new model or
reproduce original TS2Vec. The scientific results already exist in the frozen
export; the new execution uses only synthetic inputs.

## Run from a fresh extracted supplement

Prerequisites: `uv` on PATH, Python 3.12 obtainable by uv, and internet or a populated
package cache for installation. Run every command below from `slices-supplement/`,
the directory containing this guide. No Git checkout, data, account, or API key
is needed. Tested platforms and measured times are recorded with the paper's
validation receipts; these are agent checks, not independent human reuse.

```sh
uv run --no-project python audit.py
uv sync --project framework --frozen --extra viz --python 3.12
uv run --project framework python extension.py --out extension-output
```

The last command has a 120-second process timeout and one CPU thread. It refuses
an existing output directory. Use a new name for another run. It performs two SSL
epochs, then two epochs for each of four downstream runs: binary classification
and regression under frozen linear probing and full fine-tuning. There are 80
synthetic patients, partitioned 56/12/12 by the shared patient-split function.
The shared normalizer fits only observed training values and zero-fills missing
cells afterward. Labels are artificial index-derived numbers. Neither clinical
cohort inclusion nor endpoint construction is validated by this fixture.

The command uses `SSLPretrainModule`, its v3 encoder export, `FineTuneModule`, and
Lightning training and test loops. It asserts an exact encoder transfer, unchanged
frozen weights, changed fine-tuned weights, and an exact reload of the downstream
checkpoint selected by validation average precision or absolute error. Each test
partition is evaluated once. A missing best checkpoint fails; there is no silent
final-model fallback. No synthetic model weights are distributed in the supplement.

## Components changed by an objective extension

The TS2Vec implementation and its registry/configuration entries already existed
before this case. The new runner validates those entries and composes their YAML;
it does not bypass them by constructing a private replacement trainer.

| Component | TS2Vec-style change | Shared component retained |
| --- | --- | --- |
| Objective code | `framework/src/slices/models/pretraining/ts2vec.py`: `TS2VecObjective`, `TS2VecConfig` | `BaseSSLObjective.forward(values, mask)` and `get_encoder()` |
| Factory registration | `SSL_REGISTRY['ts2vec']` and `CONFIG_REGISTRY['ts2vec']` in `pretraining/factory.py` | `build_ssl_objective` and configuration validation |
| Launch compatibility | `'ts2vec': {'transformer'}` in `framework/scripts/training/pretrain.py` | Canonical obs-aware Transformer, sequence outputs during SSL |
| Objective configuration | `framework/configs/ssl/ts2vec.yaml` | Hydra model, task, optimizer, and protocol configuration groups |
| Views and loss | Independent masks, observed-value noise .01, no crop, temporal pairs, four pooling scales, temperature .05, projection 256/64 | Values and observation mask stay separate; empty hours excluded |
| Transfer | Discard temporal projector; export common encoder | Same 159,808 parameters and saved encoder configuration; downstream valid-hour mean pooling |
| Evaluation | Select `protocol=linear_probe` or `protocol=full_finetune` | Shared heads, losses, primary metrics, and validation-selected downstream testing |
| New evidence | `experiment_class=ts2vec_extension`, a fresh revision and output | Matched cell keys and traceable source/config/checkpoint identities |

Unlike original TS2Vec's dilated convolutional encoder and separate temporal and
instance objectives, this implementation uses SLICES's Transformer and combines
temporal and cross-instance negatives in one denominator. The original method is
described in [Yue et al., AAAI 2022](https://cdn.aaai.org/ojs/20881/20881-13-24894-1-2-20220628.pdf).
The scientific scope is this adaptation, not original TS2Vec's performance.

## Fixed scientific contract and declared synthetic departures

An intended scientific extension changes only `objective` and `objective_config`
in `benchmark.json`. `scientific-declaration.json` is that intended specification,
not a claim that the short execution used the full benchmark schedule.
Before shortening the run, the case checks composed encoder, pretraining, and
adaptation settings against the declaration. A configuration drift fails early.

Fixed for scientific comparison: database versions, cohort rules, ordered 84
features, 24-hour input and clinical label horizons, patient splits, train-only
normalization, observation masks, encoder architecture, final-epoch SSL transfer,
100-epoch pretraining schedule, both distinct adaptation schedules, downstream
selection rules, primary metrics, and five matched seeds. Objective modules,
augmentations, trainable parameter totals, encoder passes, runtime, and tuning
effort can differ. Matching declarations do not establish compute parity or prove
that private data and split manifests match.

`execution-declaration.json` records the actual short synthetic run. It changes
the data/label source, cohort applicability, seed coverage, batch sizes, epoch
caps, warmup, and stopping schedule. Its comparison with the scientific
declaration must fail. The run also rejects negative controls that reorder
features, shorten pretraining, or switch to best-SSL-loss transfer. This prevents
a passing software test from being labeled a comparable scientific result.

## Inspect and report results

To reproduce just the scientific comparison without installing numerical packages
or running the synthetic experiment:

```sh
uv run --no-project python extension_report.py --out extension-report
```

| Output inside `extension-output/` | Meaning |
| --- | --- |
| `synthetic-results.json` | Four execution records, transfer/update/reload checks, counts, versions, timings, source/config/checkpoint and synthetic input/split/normalization hashes |
| `pretrain-config.json` and task/protocol `config.json` | Resolved executed configuration, including every shortened schedule |
| `encoder.pt` and task/protocol `best.ckpt` | Locally generated synthetic weights only; retained to inspect transfer and selection |
| `scientific-declaration.json`, `execution-declaration.json` | Intended scientific contract versus actual test declaration |
| `frozen-extension.json` | Existing scientific results: 24 dataset/task/protocol cells, five seeds each, means, sample SD, paired differences, and scoped winner |
| `frozen-pairs.csv` | 120 matched-seed differences with anonymous scientific IDs joining both methods to `results/scientific_configs.json` |

The frozen reporting path checks exact seed coverage, successful best-downstream
checkpoint loading, and config-to-cell identity. It rejects differing sample or
positive counts when both are available. All 120 frozen TS2Vec rows lack these
counts, so test-support comparability is explicitly unknown. Two missing values
never count as a verified match. Counts alone would not prove private split
identity. The report yields the
existing 8/12 fine-tuning wins for TS2Vec-style and 12/12 probe wins for core
contrastive. AUPRC is higher-better; LOS absolute error is lower-better. Paired
differences are always TS2Vec minus contrastive. No pooled significance claim or
synthetic score is admitted into these results.

## Continue to a credentialed scientific experiment

Only after obtaining authorized inputs and following `TRAINING.md`, print the
full-budget commands from the supplement root:

```sh
uv run --no-project python training_recipe.py --tree framework --objective ts2vec --dataset miiv --task mortality_hospital --seed 42 --protocol linear_probe
uv run --no-project python training_recipe.py --tree framework --objective ts2vec --dataset miiv --task mortality_hospital --seed 42 --protocol full_finetune
```

These commands print recipes only. Run the emitted commands from `framework/`.
Pretrain once and reuse its final encoder in both protocols; the second recipe's
pretraining line is a duplicate, not an instruction to train twice. Append
`--cfg job` to each emitted Python training command to check configuration without
reading clinical data or training. Repeat approved scientific runs for all five
seeds and the intended tasks/regimes. Save them in a fresh revision. For exact
historical settings, use the keyed frozen scientific configurations and the
`historical/` tree as explained in `TRAINING.md`.

This revision fixes the recipe printer's missing TS2Vec option and incorrect core
classification for extension recipes. It adds the shared-trainer workflow and
paired result report. It does not change frozen evidence, benchmark constants,
clinical preprocessing, objective computation, or the thesis.
