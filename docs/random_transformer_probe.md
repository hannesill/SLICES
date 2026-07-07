# Random Transformer Probe Diagnostic

This diagnostic estimates the frozen random-encoder linear-probe baseline for
the default SLICES Transformer family. It caches random-encoder embeddings once
per dataset/task/seed and trains the linear head on those cached embeddings.
It is not core thesis evidence.

Run a dry-run check:

```sh
uv run python scripts/diagnostics/run_cached_random_transformer_probe.py --dry-run --limit 2
```

Run the full local grid sequentially:

```sh
uv run python scripts/diagnostics/run_cached_random_transformer_probe.py \
  --datasets miiv eicu combined \
  --tasks mortality_24h mortality_hospital aki_kdigo los_remaining \
  --seeds 42 123 456 789 1011 \
  --accelerator auto \
  --devices 1 \
  --no-wandb \
  --skip-existing
```

Run a one-job smoke with a one-epoch cap:

```sh
uv run python scripts/diagnostics/run_cached_random_transformer_probe.py \
  --datasets combined \
  --tasks mortality_24h \
  --seeds 42 \
  --accelerator auto \
  --devices 1 \
  --no-wandb \
  --max-epochs 1
```

The cached runner accepts `--wandb`/`--wandb-project` for CLI compatibility,
but currently forces `logging.use_wandb=false`. Do not log these runs to
`slices-final`.

Diagnostic identity:

- `experiment_class=random_probe_diagnostic`
- `experiment_subtype=random_transformer_cached_linear_probe`
- `revision=random-probe-cached-v1`
- `paradigm=random_transformer`
- `protocol=linear_probe_cached_embeddings`
- `label_fraction=1.0`
- `model_size=default`

Outputs:

- Run outputs: `outputs/diagnostics/random_transformer_probe_cached/...`
- Embedding cache per run: `embeddings.pt`
- Test metrics per run: `test_results.json`
- Launch manifests: `outputs/diagnostics/random_transformer_probe_cached/manifests/*.jsonl`
- Completion markers for `--skip-existing`: `diagnostic_complete.json`

To compare against frozen SSL linear probes, export the diagnostic runs to a
separate artifact such as `results/random-probe-diagnostic-v1/`. Do not merge
with `results/slices-thesis-v1/`. Join diagnostic rows to frozen rows from
`results/slices-thesis-v1/per_seed_results.parquet` on:

- `dataset`
- `task`
- `seed`
- `protocol=linear_probe`
- `label_fraction=1.0`
- `model_size=default`
- metric column: classification `test/auprc`, LOS `test/mae`

Keep the diagnostic row labelled `paradigm=random_transformer`. Treat any
difference from MAE, JEPA, or contrastive rows as a local diagnostic comparison,
not a frozen thesis result.
