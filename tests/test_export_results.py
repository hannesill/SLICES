"""Tests for the results export pipeline."""

import importlib
import sys

import pandas as pd


def test_build_statistical_tests_df_produces_pairwise_significance_rows():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for paradigm, phase, offset in [
        ("mae", "finetune", 0.10),
        ("supervised", "supervised", 0.0),
    ]:
        for seed in [42, 123, 456]:
            for task, base in [("mortality_24h", 0.30), ("aki_kdigo", 0.25)]:
                rows.append(
                    {
                        "experiment_type": "core",
                        "sprint": "1",
                        "paradigm": paradigm,
                        "dataset": "miiv",
                        "task": task,
                        "seed": seed,
                        "protocol": "A",
                        "label_fraction": 1.0,
                        "model_size": "default",
                        "source_dataset": None,
                        "phase": phase,
                        "test/auprc": base + offset + (seed / 100000.0),
                    }
                )

    per_seed_df = pd.DataFrame(rows)
    stats_df = mod.build_statistical_tests_df(per_seed_df)

    assert len(stats_df) == 1
    row = stats_df.iloc[0]
    assert row["primary_metric_name"] == "test/auprc"
    assert row["paradigm_a"] == "mae"
    assert row["paradigm_b"] == "supervised"
    assert row["n_pairs"] == 6
    assert row["n_tasks"] == 2
    assert row["better_paradigm"] == "mae"
    assert row["p_value_bonferroni"] >= row["p_value"]


def test_build_statistical_tests_df_adds_classical_context_rows():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for experiment_type, paradigm, phase, offset in [
        ("core", "mae", "finetune", 0.10),
        ("core", "supervised", "supervised", 0.0),
        ("classical_baselines", "xgboost", "baseline", 0.03),
        ("classical_baselines", "gru_d", "baseline", 0.05),
    ]:
        for seed in [42, 123]:
            for task, base in [("mortality_24h", 0.30), ("aki_kdigo", 0.25)]:
                rows.append(
                    {
                        "experiment_type": experiment_type,
                        "sprint": "11" if experiment_type == "classical_baselines" else "1",
                        "paradigm": paradigm,
                        "dataset": "miiv",
                        "task": task,
                        "seed": seed,
                        "protocol": "B",
                        "label_fraction": 1.0,
                        "model_size": "default",
                        "source_dataset": None,
                        "phase": phase,
                        "test/auprc": base + offset + (seed / 100000.0),
                    }
                )

    stats_df = mod.build_statistical_tests_df(pd.DataFrame(rows))
    contextual = stats_df[stats_df["experiment_type"] == "classical_context_full"]
    pairs = {
        tuple(sorted((row["paradigm_a"], row["paradigm_b"]))) for _, row in contextual.iterrows()
    }

    assert ("supervised", "xgboost") in pairs
    assert ("mae", "xgboost") in pairs
    assert ("gru_d", "supervised") in pairs
    assert ("gru_d", "mae") in pairs
    assert ("gru_d", "xgboost") not in pairs
    assert ("mae", "supervised") not in pairs


def test_parse_args_uses_revision_env_when_cli_omits_it(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setenv("REVISION", "thesis-v1")
    monkeypatch.delenv("WANDB_REVISION", raising=False)
    monkeypatch.setattr(sys, "argv", ["export_results.py"])

    args = mod.parse_args()

    assert args.revision == ["thesis-v1"]


def test_parse_args_requires_revision_without_cli_or_env(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.delenv("REVISION", raising=False)
    monkeypatch.delenv("WANDB_REVISION", raising=False)
    monkeypatch.setattr(sys, "argv", ["export_results.py"])

    try:
        mod.parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args() should require an explicit revision scope")
