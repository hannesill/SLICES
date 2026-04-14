"""Tests for the results export pipeline."""

import importlib

import pandas as pd


def test_build_statistical_tests_df_produces_pairwise_significance_rows():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for paradigm, offset in [("mae", 0.10), ("supervised", 0.0)]:
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
                        "phase": "finetune",
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
