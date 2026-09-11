# Add an objective without changing the evaluation contract

The fixed declaration is `benchmark.json`. MAE, JEPA, and contrastive learning are
reference implementations. The registry accepts additional objectives; it does
not certify their scientific comparability automatically.

Start with `TS2VEC_CASE.md` for a complete runnable case using the existing
TS2Vec-style objective, shared trainers, both evaluation protocols, and a paired
frozen-result report. The steps below describe how to add a different objective.

1. Subclass `BaseSSLObjective` in `framework/src/slices/models/pretraining/`.
   Implement `forward(x, obs_mask) -> (scalar_loss, metrics)` for tensors of shape
   `[batch, 24, 84]`. `obs_mask=True` means observed. Missing cells never become
   reconstruction targets. The encoder provides `tokenize()` and `encode()`;
   controlled SSL requires `obs_aware=True` and `pooling='none'`.
2. Register the objective and its dataclass in `SSL_REGISTRY` and `CONFIG_REGISTRY`.
   Add its Hydra configuration to `configs/ssl/` and its Transformer compatibility
   entry to `scripts/training/pretrain.py`. `smoke.py` demonstrates registry use
   with a toy objective and no patient data.
3. Return the common encoder through `get_encoder()`. Only encoder parameters and
   saved encoder configuration transfer. Auxiliary decoders, projectors, predictors,
   and target copies do not. Downstream loading rebuilds the saved architecture,
   changes `pooling='none'` to `'mean'`, and rejects incompatible CLS pooling.
4. Copy `benchmark.json` into the new experiment declaration. Change only
   `objective` and `objective_config` for a core-compatible objective comparison.
   Use `slices.benchmark.contract.compare_contracts(reference, candidate)`;
   an empty difference list means the declarations match. For actual cell pairing,
   also record the same dataset, seed, feature-order, cohort, split, normalization,
   source/config/checkpoint hashes and validation-selected test artifact. Private
   patient/split manifests stay local. A declaration match cannot prove data use.
5. Run the synthetic smoke and targeted objective tests before credentialed runs.
   Export to a fresh revision with source/config/checkpoint provenance. Changing
   a cohort, encoder, preprocessing, adaptation schedule, or checkpoint policy
   creates a separate analysis. Never append new runs to the frozen result CSV.

The package does not impose compute parity: record total parameters, encoder
passes, measured runtime, and tuning effort for each extension. Equal epochs are
equal data passes within a regime, not equal computational cost.
