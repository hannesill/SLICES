"""Tests for the class-based results export pipeline."""

import importlib
import sys

import pandas as pd
import pytest


class DummyRun:
    def __init__(self, config, tags, summary=None, name="dummy"):
        self.config = config
        self.tags = tags
        self.summary_metrics = summary or {}
        self.id = "dummy-id"
        self.url = "https://wandb.test/dummy-id"
        self.name = name
        self.group = None
        self.created_at = "2026-04-21T00:00:00"


def _row(experiment_class, paradigm, seed, task="mortality_24h", offset=0.0):
    return {
        "experiment_class": experiment_class,
        "experiment_type": experiment_class,
        "experiment_subtype": None,
        "paradigm": paradigm,
        "dataset": "miiv",
        "task": task,
        "seed": seed,
        "protocol": "B",
        "label_fraction": 1.0,
        "model_size": "default",
        "source_dataset": None,
        "phase": "baseline" if paradigm in {"gru_d", "xgboost"} else "finetune",
        "upstream_pretrain_lr": None,
        "upstream_pretrain_mask_ratio": None,
        "test/auprc": 0.3 + offset + (seed / 100000.0),
    }


def test_build_statistical_tests_df_produces_pairwise_significance_rows():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for paradigm, offset in [("mae", 0.10), ("supervised", 0.0)]:
        for seed in [42, 123, 456]:
            for task, base in [("mortality_24h", 0.30), ("aki_kdigo", 0.25)]:
                row = _row("core_ssl_benchmark", paradigm, seed, task, offset)
                row["test/auprc"] = base + offset + (seed / 100000.0)
                rows.append(row)

    stats_df = mod.build_statistical_tests_df(pd.DataFrame(rows))

    assert len(stats_df) == 1
    row = stats_df.iloc[0]
    assert row["primary_metric_name"] == "test/auprc"
    assert row["paradigm_a"] == "mae"
    assert row["paradigm_b"] == "supervised"
    assert row["n_pairs"] == 6
    assert row["better_paradigm"] == "mae"
    assert row["n_shared_task_seed_pairs"] == 6


def test_extract_run_requires_experiment_class():
    mod = importlib.import_module("scripts.export_results")

    run = DummyRun(
        config={
            "dataset": "miiv",
            "paradigm": "mae",
            "seed": 42,
            "training": {"freeze_encoder": False},
            "task": {"task_name": "mortality_24h"},
        },
        tags=["phase:finetune"],
    )

    with pytest.raises(RuntimeError, match="no experiment_class"):
        mod.extract_run(run, [])


def test_extract_run_uses_class_metadata_from_config_or_tags():
    mod = importlib.import_module("scripts.export_results")

    config_run = DummyRun(
        config={
            "experiment_class": "core_ssl_benchmark",
            "dataset": "miiv",
            "paradigm": "mae",
            "seed": 42,
            "training": {"freeze_encoder": False},
            "task": {"task_name": "mortality_24h"},
            "encoder": {"d_model": 64, "n_layers": 2},
        },
        tags=["phase:finetune", "revision:thesis-v1"],
    )
    tag_run = DummyRun(
        config={
            "dataset": "miiv",
            "paradigm": "xgboost",
            "seed": 42,
            "task": {"task_name": "mortality_24h"},
        },
        tags=["experiment_class:classical_baselines", "phase:baseline"],
    )

    assert mod.extract_run(config_run, [])["experiment_class"] == "core_ssl_benchmark"
    tag_row = mod.extract_run(tag_run, [])
    assert tag_row["experiment_class"] == "classical_baselines"
    assert tag_row["protocol"] == "B"


def test_fetch_all_runs_single_class_adds_server_side_filter(monkeypatch):
    mod = importlib.import_module("scripts.export_results")
    captured = {}

    class DummyApi:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def runs(self, path, filters, order):
            captured["path"] = path
            captured["filters"] = filters
            captured["order"] = order
            return []

    monkeypatch.setattr(mod.wandb, "Api", DummyApi)

    mod.fetch_all_runs(
        project="proj",
        entity="entity",
        experiment_class=["core_ssl_benchmark"],
        paradigm=["mae"],
        dataset=["miiv"],
        phase=["finetune"],
        revision=["thesis-v1"],
    )

    assert captured["path"] == "entity/proj"
    assert captured["filters"]["tags"]["$all"] == [
        "experiment_class:core_ssl_benchmark",
        "paradigm:mae",
        "dataset:miiv",
        "phase:finetune",
        "revision:thesis-v1",
    ]


def test_build_per_seed_df_keeps_hp_robustness_rows_out_of_derived_views():
    mod = importlib.import_module("scripts.export_results")

    runs = []
    for run_id, lr_code in [("hp-lr-2e4", "00002"), ("hp-lr-5e4", "00005")]:
        run = DummyRun(
            config={
                "experiment_class": "hp_robustness",
                "experiment_subtype": "lr_sensitivity",
                "dataset": "miiv",
                "paradigm": "mae",
                "seed": 42,
                "output_dir": (
                    "outputs/hp_robustness/"
                    f"finetune_mae_mortality_24h_miiv_seed42_lr{lr_code}"
                ),
                "encoder": {"d_model": 64, "n_layers": 2},
                "training": {"freeze_encoder": False},
                "task": {"task_name": "mortality_24h"},
                "label_fraction": 1.0,
            },
            tags=["experiment_class:hp_robustness", "phase:finetune"],
            name=f"finetune_mae_mortality_24h_miiv_seed42_lr{lr_code}",
        )
        run.id = run_id
        run.url = f"https://wandb.test/{run_id}"
        runs.append(run)

    df = mod.build_per_seed_df(runs)

    assert len(df) == 2
    assert set(df["experiment_type"]) == {"hp_robustness"}
    assert mod.build_label_efficiency_curve_df(df).empty
    assert mod.build_capacity_study_comparison_df(df).empty
    assert sorted(df["upstream_pretrain_lr"].tolist()) == [2e-4, 5e-4]


def test_build_per_seed_df_fails_closed_on_extraction_errors():
    mod = importlib.import_module("scripts.export_results")

    with pytest.raises(RuntimeError, match="failed extraction"):
        mod.build_per_seed_df([object()])

    assert mod.build_per_seed_df([object()], allow_extraction_failures=True).empty


def test_label_efficiency_view_adds_core_full_label_endpoint():
    mod = importlib.import_module("scripts.export_results")

    rows = [
        _row("core_ssl_benchmark", "mae", 42, offset=0.1),
        {**_row("label_efficiency", "mae", 42, offset=0.0), "label_fraction": 0.1},
    ]

    view = mod.build_label_efficiency_curve_df(pd.DataFrame(rows))

    assert set(view["source_experiment_class"]) == {"core_ssl_benchmark", "label_efficiency"}
    assert set(view["experiment_class"]) == {"label_efficiency"}
    assert set(view["comparison_view"]) == {"label_efficiency_with_core_endpoint"}


def test_build_statistical_tests_df_adds_classical_context_rows():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for experiment_class, paradigm, offset in [
        ("core_ssl_benchmark", "mae", 0.10),
        ("core_ssl_benchmark", "supervised", 0.0),
        ("classical_baselines", "xgboost", 0.03),
        ("classical_baselines", "gru_d", 0.05),
    ]:
        for seed in [42, 123]:
            for task in ["mortality_24h", "aki_kdigo"]:
                rows.append(_row(experiment_class, paradigm, seed, task, offset))

    stats_df = mod.build_statistical_tests_df(pd.DataFrame(rows))
    contextual = stats_df[stats_df["experiment_class"] == "classical_context_full"]
    pairs = {
        tuple(sorted((row["paradigm_a"], row["paradigm_b"]))) for _, row in contextual.iterrows()
    }

    assert ("mae", "xgboost") in pairs
    assert ("gru_d", "supervised") in pairs
    assert ("gru_d", "xgboost") not in pairs


def test_build_ts2vec_vs_core_contrastive_df_compares_across_classes():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for experiment_class, paradigm, offset in [
        ("ts2vec_extension", "ts2vec", 0.04),
        ("core_ssl_benchmark", "contrastive", 0.0),
    ]:
        for seed in [42, 123]:
            for task in ["mortality_24h", "aki_kdigo"]:
                rows.append(_row(experiment_class, paradigm, seed, task, offset))

    comparison_df = mod.build_ts2vec_vs_core_contrastive_df(pd.DataFrame(rows))

    assert len(comparison_df) == 1
    row = comparison_df.iloc[0]
    assert row["comparison_type"] == "ts2vec_vs_core_contrastive"
    assert row["paradigm_a"] == "ts2vec"
    assert row["paradigm_b"] == "contrastive"
    assert row["n_pairs"] == 4


def test_validate_flags_wrong_fixed_seed_set_with_correct_count():
    mod = importlib.import_module("scripts.export_results")

    aggregated_df = pd.DataFrame(
        [
            {
                "experiment_class": "core_ssl_benchmark",
                "paradigm": "mae",
                "dataset": "miiv",
                "task": "mortality_24h",
                "protocol": "B",
                "n_seeds": 5,
                "seed_list": "[1, 2, 3, 4, 5]",
            }
        ]
    )
    warnings = mod.validate(pd.DataFrame([{"test/auprc": 0.5}]), aggregated_df)

    assert any("expected seed set" in warning for warning in warnings)
    assert any("unexpected=[1, 2, 3, 4, 5]" in warning for warning in warnings)


def test_build_aggregated_df_records_per_metric_non_null_counts():
    mod = importlib.import_module("scripts.export_results")

    rows = [
        {**_row("core_ssl_benchmark", "mae", 42), "test/auroc": 0.7, "wandb_run_id": "run-42"},
        {
            **_row("core_ssl_benchmark", "mae", 123),
            "test/auprc": None,
            "test/auroc": 0.8,
            "wandb_run_id": "run-123",
        },
    ]

    aggregated = mod.build_aggregated_df(pd.DataFrame(rows))

    assert aggregated.iloc[0]["n_seeds"] == 2
    assert aggregated.iloc[0]["test/auprc/n"] == 1
    assert aggregated.iloc[0]["test/auroc/n"] == 2


def test_validate_warns_when_evaluation_row_missing_primary_metric():
    mod = importlib.import_module("scripts.export_results")

    warnings = mod.validate(
        pd.DataFrame(
            [
                {
                    "wandb_run_id": "run-1",
                    "paradigm": "mae",
                    "dataset": "miiv",
                    "task": "mortality_24h",
                    "seed": 42,
                    "phase": "finetune",
                    "test/auprc": None,
                }
            ]
        ),
        pd.DataFrame(),
    )

    assert any("missing their primary test metric" in warning for warning in warnings)


def test_extract_run_exports_los_regression_fairness_keys():
    mod = importlib.import_module("scripts.export_results")

    run = DummyRun(
        config={
            "experiment_class": "core_ssl_benchmark",
            "dataset": "miiv",
            "paradigm": "mae",
            "seed": 42,
            "encoder": {"d_model": 64, "n_layers": 2},
            "training": {"freeze_encoder": False},
            "task": {"task_name": "los_remaining", "task_type": "regression"},
        },
        tags=["experiment_class:core_ssl_benchmark", "phase:finetune"],
        summary={
            "test/mae": 1.4,
            "fairness/gender/worst_group_mse": 2.5,
            "fairness/gender/worst_group_mae": 1.2,
            "fairness/gender/mse_gap": 0.7,
            "fairness/gender/mae_gap": 0.3,
            "fairness/gender/per_group_mse/F": 1.8,
            "fairness/gender/per_group_mse/M": 2.5,
            "fairness/gender/per_group_mae/F": 0.9,
            "fairness/gender/per_group_mae/M": 1.2,
        },
        name="los_fairness",
    )

    row = mod.extract_run(run, mod.ALL_METRICS)
    assert row["fairness/gender/worst_group_mse"] == pytest.approx(2.5)
    assert row["fairness/gender/mse_gap"] == pytest.approx(0.7)
    assert row["fairness/gender/per_group_mse/M"] == pytest.approx(2.5)

    agg = mod.build_aggregated_df(pd.DataFrame([row]))
    assert agg.iloc[0]["fairness/gender/worst_group_mse/mean"] == pytest.approx(2.5)
    assert agg.iloc[0]["fairness/gender/per_group_mae/F/mean"] == pytest.approx(0.9)


def test_validate_warns_on_missing_or_failed_checkpoint_provenance():
    mod = importlib.import_module("scripts.export_results")

    per_seed_df = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-missing",
                "paradigm": "mae",
                "dataset": "miiv",
                "task": "mortality_24h",
                "seed": 42,
                "phase": "finetune",
                "test/auprc": 0.4,
            },
            {
                "wandb_run_id": "run-failed",
                "paradigm": "jepa",
                "dataset": "miiv",
                "task": "mortality_24h",
                "seed": 123,
                "phase": "finetune",
                "test/auprc": 0.4,
                "_eval_checkpoint_source": "failed",
                "_best_ckpt_error": "no best checkpoint",
            },
            {
                "wandb_run_id": "run-bad-load",
                "paradigm": "contrastive",
                "dataset": "miiv",
                "task": "mortality_24h",
                "seed": 456,
                "phase": "finetune",
                "test/auprc": 0.4,
                "_eval_checkpoint_source": "best",
                "_best_ckpt_path": "outputs/run/checkpoints/best.ckpt",
                "_best_ckpt_load_ok": False,
            },
            {
                "wandb_run_id": "run-xgb",
                "paradigm": "xgboost",
                "dataset": "miiv",
                "task": "mortality_24h",
                "seed": 42,
                "phase": "baseline",
                "test/auprc": 0.4,
            },
        ]
    )

    warnings = mod.validate(per_seed_df, pd.DataFrame())
    joined = "\n".join(warnings)

    assert "checkpoint provenance" in joined
    assert "run-missing" in joined
    assert "missing _eval_checkpoint_source" in joined
    assert "run-failed" in joined
    assert "run-bad-load" in joined
    assert "run-xgb" not in joined


def test_filter_thesis_tasks_excludes_optional_mortality_by_default():
    mod = importlib.import_module("scripts.export_results")

    df = pd.DataFrame(
        [
            {"task": "mortality_24h", "wandb_run_id": "main"},
            {"task": "mortality", "wandb_run_id": "extension"},
        ]
    )

    filtered = mod.filter_thesis_tasks(df)
    unfiltered = mod.filter_thesis_tasks(df, include_extension_tasks=True)

    assert filtered["wandb_run_id"].tolist() == ["main"]
    assert unfiltered["wandb_run_id"].tolist() == ["main", "extension"]


def test_parse_args_exposes_extension_task_escape_hatch(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_results.py", "--revision", "thesis-v1", "--include-extension-tasks"],
    )

    args = mod.parse_args()

    assert args.include_extension_tasks is True


def test_parse_args_uses_revision_env_when_cli_omits_it(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setenv("REVISION", "thesis-v1")
    monkeypatch.delenv("WANDB_REVISION", raising=False)
    monkeypatch.setattr(sys, "argv", ["export_results.py"])

    args = mod.parse_args()

    assert args.revision == ["thesis-v1"]


def test_parse_args_exposes_allow_incomplete_escape_hatch(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_results.py", "--revision", "thesis-v1", "--allow-incomplete"],
    )

    assert mod.parse_args().allow_incomplete is True


def test_parse_args_exposes_extraction_failure_escape_hatch(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_results.py", "--revision", "thesis-v1", "--allow-extraction-failures"],
    )

    assert mod.parse_args().allow_extraction_failures is True


def test_main_exits_nonzero_when_no_runs_match(monkeypatch, tmp_path):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_results.py", "--revision", "thesis-v1", "--output-dir", str(tmp_path)],
    )
    monkeypatch.setattr(mod, "fetch_all_runs", lambda **_: [])

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1


def test_parse_args_requires_revision_without_cli_or_env(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.delenv("REVISION", raising=False)
    monkeypatch.delenv("WANDB_REVISION", raising=False)
    monkeypatch.setattr(sys, "argv", ["export_results.py"])

    with pytest.raises(SystemExit) as excinfo:
        mod.parse_args()

    assert excinfo.value.code == 2
