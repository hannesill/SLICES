# Credentialed training reproduction

Training is separate from the public aggregate audit. Obtain MIMIC-IV v2.2 and
eICU-CRD v2.0 through PhysioNet with the required credentialing, training, and data
use agreements. Use your own authorized local copies. This package provides no
records, patient splits, normalization statistics, or trained patient-data weights.

`historical/` contains source and configs from the frozen launch revision.
`framework/` is the current refactored code plus review tooling. Do not attribute
new framework checks or fixes to historical training, and do not call a rerun an
identical result merely because it uses the same seed. Historical execution was
not bitwise deterministic. The source trees are inventoried by SHA256; identifying
owner text is redacted in the review copies, including the copyright name, while
the existing MIT permission text is preserved. The owner identity is withheld for
double-blind review.

## Prepare the authorized inputs

From `historical/`, install `uv sync --frozen --python 3.12`. The R extraction requires RICU
0.6.3 and its R dependencies. `Rscript scripts/preprocessing/extract_with_ricu.R --help`
documents the source directory and output options. Use the MIMIC-IV v2.2 RICU
configuration, eICU v2.0, and retain the 48-hour export needed for forward labels.
Do not substitute a later database version. Produce `data/ricu_output/{miiv,eicu}`,
then run the Python extraction and preparation paths in `scripts/preprocessing/`.
For example, after installing the R dependencies and obtaining authorized data:

```sh
Rscript scripts/preprocessing/extract_with_ricu.R --dataset miiv --raw_data_dir /path/to/mimic-iv-2.2 --output_dir data/ricu_output/miiv --raw_export_horizon_hours 48
Rscript scripts/preprocessing/extract_with_ricu.R --dataset eicu --raw_data_dir /path/to/eicu-2.0 --output_dir data/ricu_output/eicu --raw_export_horizon_hours 48
```

Where available, the current convenience orchestrator prints the exact sequence:

```sh
uv run --project ../framework python ../framework/scripts/preprocessing/build_processed_data.py --datasets miiv eicu --combined --dry-run
```

Run the printed extraction, preparation, and combined-dataset commands from the
chosen source tree. The scripts write only your local data directories. Validate
74,829 MIMIC-IV stays, 132,686 eICU stays, and 207,515 pooled stays; feature order
must exactly match `../benchmark.json`. A mismatch requires diagnosis and a new
analysis identity, not changing these reference counts to obtain a pass.

RICU handles concept-specific units and hourly aggregation. Python retains the
first 24 hours, removes empty stays and blocklisted label/future-derived features,
and builds values plus a separate observation mask. Full-cohort patient splits
precede task-specific label filtering. For each seed, the normalizer fits observed
training cells only, using sample SD, a minimum SD floor, and zero fill after
normalization. Low-label subsets use the full training-partition normalizer.

## Reproduce one core cell before expanding

From the supplement root, print a recipe:

```sh
uv run --no-project python training_recipe.py --dataset miiv --objective mae --task mortality_hospital --seed 42 --protocol linear_probe
```

Run the printed commands from `historical/`. They force CPU execution and disable
W&B. Full training is substantially larger than the smoke test and is not launched
by any audit command. The first command trains for 100 epochs; the second loads
the final `encoder.pt`. Repeat for the five seeds, three regimes and objectives,
four tasks, and both protocols. Reuse each pretrained encoder across its tasks
and protocols. Supervised, GRU-D and XGBoost entry points are in `scripts/training/`.
For every historical evaluated run, `results/scientific_configs.json` retains
its data semantics, encoder, head, optimizer, scheduler and training settings.
Anonymous scientific IDs join each record to exactly one result row. The
`scientific_cell` object retains subtype, source dataset and upstream learning
rate/mask ratio even when they are absent from the downstream Hydra config.
It also covers reduced labels, capacity, hyperparameters and transfer extensions.
Use these records to reproduce a particular extension, not current defaults.

Classification uses weighted, label-smoothed two-logit cross entropy. LOS uses
**squared-error training loss**, while validation selection and reported primary
evaluation use **absolute error in days**. Pretraining transfers final epochs;
all 420 exported core runs record successful best-validation-checkpoint loading.
The historical trainer fails on a recorded checkpoint that cannot load, unless
fallback is explicitly enabled; if no best path exists, it tests the final model.
That historical exception is not a recommended comparison policy. New exports
must retain and check the actual checkpoint source. Record source, config, data/split, normalization and checkpoint
hashes, runtime, seed, actual epoch and test counts in a new output directory.

Patient-level preprocessing and full historical training have not been rerun for
this supplement. The source and config recipe, synthetic execution, and public
result derivation were checked. Numerical retraining equivalence is unverified.
