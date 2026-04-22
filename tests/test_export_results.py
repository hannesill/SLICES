"""Tests for the results export pipeline."""

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
    assert row["n_shared_task_seed_pairs"] == 6
    assert row["n_union_task_seed_pairs"] == 6


def test_extract_run_uses_requested_inherited_sprint_tag_for_family():
    mod = importlib.import_module("scripts.export_results")

    run = DummyRun(
        config={
            "sprint": "6",
            "dataset": "miiv",
            "paradigm": "mae",
            "seed": 42,
            "encoder": {"d_model": 64, "n_layers": 2},
            "training": {"freeze_encoder": False},
            "task": {"task_name": "mortality_24h"},
            "label_fraction": 0.1,
        },
        tags=["sprint:6", "sprint:7p", "phase:finetune"],
        name="s6_inherited_capacity_baseline",
    )

    default_row = mod.extract_run(run, [])
    requested_row = mod.extract_run(run, [], requested_sprints=["7p"])

    assert default_row["experiment_type"] == "label_efficiency"
    assert requested_row["experiment_type"] == "capacity_pilot"
    assert requested_row["sprint"] == "7p"
    assert requested_row["config_sprint"] == "6"


def test_build_per_seed_df_adds_revision_wide_capacity_membership_for_inherited_row():
    mod = importlib.import_module("scripts.export_results")

    run = DummyRun(
        config={
            "sprint": "6",
            "dataset": "miiv",
            "paradigm": "mae",
            "seed": 42,
            "encoder": {"d_model": 64, "n_layers": 2},
            "training": {"freeze_encoder": False},
            "task": {"task_name": "mortality_24h"},
            "label_fraction": 0.1,
        },
        tags=["sprint:6", "sprint:7p", "phase:finetune"],
        name="s6_inherited_capacity_baseline",
    )

    df = mod.build_per_seed_df([run])

    assert set(df["experiment_type"]) == {"label_efficiency", "capacity_pilot"}
    capacity = df[df["experiment_type"] == "capacity_pilot"].iloc[0]
    assert capacity["sprint"] == "7p"
    assert capacity["config_sprint"] == "6"
    assert capacity["wandb_run_id"] == "dummy-id"

    scoped_df = mod.build_per_seed_df([run], requested_sprints=["6", "7p"])
    assert set(scoped_df["experiment_type"]) == {"label_efficiency", "capacity_pilot"}


def test_build_per_seed_df_does_not_add_capacity_membership_for_irrelevant_tagged_rows():
    mod = importlib.import_module("scripts.export_results")

    run = DummyRun(
        config={
            "sprint": "6",
            "dataset": "eicu",
            "paradigm": "jepa",
            "seed": 42,
            "encoder": {"d_model": 64, "n_layers": 2},
            "training": {"freeze_encoder": False},
            "task": {"task_name": "mortality_hospital"},
            "label_fraction": 0.05,
        },
        tags=["sprint:6", "sprint:7p", "phase:finetune"],
        name="s6_coarsely_tagged_but_not_capacity_scope",
    )

    df = mod.build_per_seed_df([run])

    assert df["experiment_type"].tolist() == ["label_efficiency"]


def test_build_per_seed_df_adds_unscoped_inherited_memberships_for_thesis_families():
    mod = importlib.import_module("scripts.export_results")

    base_config = {
        "sprint": "1",
        "dataset": "miiv",
        "paradigm": "mae",
        "seed": 42,
        "encoder": {"d_model": 64, "n_layers": 2},
        "training": {"freeze_encoder": False},
        "task": {"task_name": "mortality_24h"},
        "label_fraction": 1.0,
    }
    runs = [
        DummyRun(base_config, ["sprint:1", "sprint:6", "phase:finetune"], name="core_for_s6"),
        DummyRun(base_config, ["sprint:1", "sprint:7", "phase:finetune"], name="core_for_s7"),
        DummyRun(base_config, ["sprint:1", "sprint:8", "phase:finetune"], name="core_for_s8"),
    ]
    for i, run in enumerate(runs):
        run.id = f"run-{i}"
        run.url = f"https://wandb.test/run-{i}"
        run.created_at = f"2026-04-21T00:00:0{i}"

    df = mod.build_per_seed_df(runs)
    memberships = set(zip(df["wandb_run_id"], df["sprint"], df["experiment_type"], strict=True))

    assert ("run-0", "6", "label_efficiency") in memberships
    assert ("run-1", "7", "transfer") in memberships
    assert ("run-2", "8", "hp_ablation") in memberships


def test_build_per_seed_df_rejects_hp_ablation_from_label_and_transfer_inheritance():
    mod = importlib.import_module("scripts.export_results")

    runs = []
    for run_id, lr_code in [("hp-lr-2e4", "00002"), ("hp-lr-5e4", "00005")]:
        run = DummyRun(
            config={
                "sprint": "10",
                "dataset": "miiv",
                "paradigm": "mae",
                "seed": 42,
                "output_dir": (
                    "outputs/sprint10/" f"finetune_mae_mortality_24h_miiv_seed42_lr{lr_code}"
                ),
                "encoder": {"d_model": 64, "n_layers": 2},
                "training": {"freeze_encoder": False},
                "task": {"task_name": "mortality_24h"},
                "label_fraction": 1.0,
            },
            tags=["sprint:10", "sprint:6", "sprint:7", "phase:finetune"],
            name=f"finetune_mae_mortality_24h_miiv_seed42_lr{lr_code}",
        )
        run.id = run_id
        run.url = f"https://wandb.test/{run_id}"
        runs.append(run)

    df = mod.build_per_seed_df(runs)

    assert len(df) == 2
    assert set(df["experiment_type"]) == {"hp_ablation"}
    assert set(df["sprint"]) == {"10"}
    assert sorted(df["upstream_pretrain_lr"].tolist()) == [2e-4, 5e-4]


def test_build_per_seed_df_fails_closed_on_extraction_errors():
    mod = importlib.import_module("scripts.export_results")

    bad_run = object()

    with pytest.raises(RuntimeError, match="failed extraction"):
        mod.build_per_seed_df([bad_run])

    df = mod.build_per_seed_df([bad_run], allow_extraction_failures=True)
    assert df.empty


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


def test_build_statistical_tests_df_reports_incomplete_pair_coverage():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for paradigm, seeds in [("mae", [42, 123]), ("supervised", [42])]:
        for seed in seeds:
            rows.append(
                {
                    "experiment_type": "core",
                    "sprint": "1",
                    "paradigm": paradigm,
                    "dataset": "miiv",
                    "task": "mortality_24h",
                    "seed": seed,
                    "protocol": "B",
                    "label_fraction": 1.0,
                    "model_size": "default",
                    "source_dataset": None,
                    "phase": "finetune",
                    "test/auprc": 0.3 + (0.1 if paradigm == "mae" else 0.0),
                }
            )

    stats_df = mod.build_statistical_tests_df(pd.DataFrame(rows))
    row = stats_df.iloc[0]

    assert row["n_pairs"] == 1
    assert row["n_task_seed_pairs_a"] == 2
    assert row["n_task_seed_pairs_b"] == 1
    assert row["n_shared_task_seed_pairs"] == 1
    assert row["n_union_task_seed_pairs"] == 2
    assert '"seed": 123' in row["missing_task_seed_pairs_b"]


def test_build_ts2vec_vs_core_contrastive_df_compares_across_experiment_types():
    mod = importlib.import_module("scripts.export_results")

    rows = []
    for experiment_type, paradigm, offset in [
        ("temporal_contrastive", "ts2vec", 0.04),
        ("core", "contrastive", 0.0),
    ]:
        for seed in [42, 123]:
            for task, base in [("mortality_24h", 0.30), ("aki_kdigo", 0.25)]:
                rows.append(
                    {
                        "experiment_type": experiment_type,
                        "sprint": "13" if paradigm == "ts2vec" else "1",
                        "paradigm": paradigm,
                        "dataset": "miiv",
                        "task": task,
                        "seed": seed,
                        "protocol": "B",
                        "label_fraction": 1.0,
                        "model_size": "default",
                        "source_dataset": None,
                        "phase": "finetune",
                        "test/auprc": base + offset + (seed / 100000.0),
                    }
                )

    comparison_df = mod.build_ts2vec_vs_core_contrastive_df(pd.DataFrame(rows))

    assert len(comparison_df) == 1
    row = comparison_df.iloc[0]
    assert row["comparison_type"] == "ts2vec_vs_core_contrastive"
    assert row["paradigm_a"] == "ts2vec"
    assert row["paradigm_b"] == "contrastive"
    assert row["n_pairs"] == 4
    assert row["better_paradigm"] == "ts2vec"


def test_validate_flags_wrong_fixed_seed_set_with_correct_count():
    mod = importlib.import_module("scripts.export_results")

    aggregated_df = pd.DataFrame(
        [
            {
                "experiment_type": "core",
                "paradigm": "mae",
                "dataset": "miiv",
                "task": "mortality_24h",
                "protocol": "B",
                "n_seeds": 5,
                "seed_list": "[1, 2, 3, 4, 5]",
            }
        ]
    )
    per_seed_df = pd.DataFrame([{"test/auprc": 0.5}])

    warnings = mod.validate(per_seed_df, aggregated_df)

    assert any("expected seed set" in warning for warning in warnings)
    assert any("unexpected=[1, 2, 3, 4, 5]" in warning for warning in warnings)


def test_build_aggregated_df_records_per_metric_non_null_counts():
    mod = importlib.import_module("scripts.export_results")

    per_seed_df = pd.DataFrame(
        [
            {
                "experiment_type": "core",
                "sprint": "1",
                "paradigm": "mae",
                "dataset": "miiv",
                "task": "mortality_24h",
                "seed": 42,
                "protocol": "B",
                "label_fraction": 1.0,
                "model_size": "default",
                "source_dataset": None,
                "phase": "finetune",
                "test/auprc": 0.4,
                "test/auroc": 0.7,
                "wandb_run_id": "run-42",
            },
            {
                "experiment_type": "core",
                "sprint": "1",
                "paradigm": "mae",
                "dataset": "miiv",
                "task": "mortality_24h",
                "seed": 123,
                "protocol": "B",
                "label_fraction": 1.0,
                "model_size": "default",
                "source_dataset": None,
                "phase": "finetune",
                "test/auprc": None,
                "test/auroc": 0.8,
                "wandb_run_id": "run-123",
            },
        ]
    )

    aggregated = mod.build_aggregated_df(per_seed_df)

    assert aggregated.iloc[0]["n_seeds"] == 2
    assert aggregated.iloc[0]["test/auprc/n"] == 1
    assert aggregated.iloc[0]["test/auroc/n"] == 2


def test_validate_warns_when_evaluation_row_missing_primary_metric():
    mod = importlib.import_module("scripts.export_results")

    per_seed_df = pd.DataFrame(
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
    )
    aggregated_df = pd.DataFrame()

    warnings = mod.validate(per_seed_df, aggregated_df)

    assert any("missing their primary test metric" in warning for warning in warnings)


def test_extract_run_exports_los_regression_fairness_keys():
    mod = importlib.import_module("scripts.export_results")

    run = DummyRun(
        config={
            "sprint": "1",
            "dataset": "miiv",
            "paradigm": "mae",
            "seed": 42,
            "encoder": {"d_model": 64, "n_layers": 2},
            "training": {"freeze_encoder": False},
            "task": {"task_name": "los_remaining", "task_type": "regression"},
        },
        tags=["sprint:1", "phase:finetune"],
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

    args = mod.parse_args()

    assert args.allow_incomplete is True


def test_parse_args_exposes_extraction_failure_escape_hatch(monkeypatch):
    mod = importlib.import_module("scripts.export_results")

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_results.py", "--revision", "thesis-v1", "--allow-extraction-failures"],
    )

    args = mod.parse_args()

    assert args.allow_extraction_failures is True


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

    try:
        mod.parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args() should require an explicit revision scope")
