# SLICES Test Suite

This directory contains the automated regression suite for data preparation,
task labeling, model components, training flows, evaluation, and export logic.

## What The Suite Covers

At a high level, the tests exercise:

- preprocessing and extractor behavior
- dataset loading, caching, normalization, and split logic
- task-label builders for mortality, AKI, and LOS
- encoder implementations and factory registration
- SSL objectives including MAE, JEPA, contrastive, TS2Vec, and SMART
- finetuning and supervised training modules
- metrics, fairness evaluation, and statistical utilities
- experiment-integrity regressions and result export paths

The suite size changes frequently. For the current inventory and collected test
count, use `pytest --collect-only` instead of relying on this README.

## Common Commands

Run the full suite:

```bash
uv run pytest tests/ -v
```

Run with coverage:

```bash
uv run pytest tests/ --cov=slices --cov-report=html --cov-report=term
```

Inspect the current collected tests:

```bash
uv run pytest --collect-only -q tests
```

Run a focused file:

```bash
uv run pytest tests/test_fairness_evaluator.py -v
```

Run a focused subset:

```bash
uv run pytest tests/ -k "jepa or ts2vec" -v
```

## Testing Philosophy

The suite is biased toward behavior and regression protection rather than
snapshotting implementation details.

Typical expectations for new work:

- add focused unit coverage for new logic
- add an integration or regression test when behavior spans modules
- prefer explicit failure modes for experiment-integrity issues
- keep tests deterministic across seeds and platforms where practical

## Writing New Tests

General conventions:

- test files start with `test_`
- test functions start with `test_`
- shared fixtures live in `conftest.py`
- behavior-specific assertions are preferred over broad smoke tests

When a change affects experiment correctness, add a regression test even if unit
coverage already exists elsewhere. This repo depends heavily on config-driven
orchestration, so the failure mode often matters as much as the happy path.
