"""Focused tests for the standalone fairness rerun script."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def _binary_fairness_summary(attr: str, base: float = 0.7) -> dict[str, float]:
    prefix = f"fairness/{attr}/"
    return {
        f"{prefix}n_valid_groups": 2,
        f"{prefix}worst_group_auroc": base,
        f"{prefix}worst_group_auprc": base - 0.1,
        f"{prefix}auroc_gap": 0.05,
        f"{prefix}auprc_gap": 0.04,
        f"{prefix}demographic_parity_diff": 0.03,
        f"{prefix}equalized_odds_diff": 0.02,
        f"{prefix}disparate_impact_ratio": 0.9,
    }


def _regression_fairness_summary(attr: str) -> dict[str, float]:
    prefix = f"fairness/{attr}/"
    return {
        f"{prefix}n_valid_groups": 2,
        f"{prefix}worst_group_mse": 2.0,
        f"{prefix}worst_group_mae": 1.1,
        f"{prefix}mse_gap": 0.4,
        f"{prefix}mae_gap": 0.2,
    }


def test_default_phases_include_baseline():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert "baseline" in mod.DEFAULT_PHASES


def test_default_experiment_classes_include_classical_baselines():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert "classical_baselines" in mod.DEFAULT_EXPERIMENT_CLASSES


def test_default_experiment_classes_include_downstream_families():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert {"label_efficiency", "cross_dataset_transfer", "hp_robustness"}.issubset(
        set(mod.DEFAULT_EXPERIMENT_CLASSES)
    )


def test_parse_args_exposes_allow_incomplete_escape_hatch(monkeypatch):
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_fairness.py", "--revision", "thesis-v1", "--allow-incomplete"],
    )

    args = mod.parse_args()

    assert args.allow_incomplete is True


def test_main_exits_nonzero_when_no_runs_match(monkeypatch):
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    monkeypatch.setattr(sys, "argv", ["evaluate_fairness.py", "--revision", "thesis-v1"])
    monkeypatch.setattr(mod, "fetch_eval_runs", lambda **_: [])

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1


def test_resolve_evaluation_artifact_supports_xgboost(tmp_path):
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    run_dir = tmp_path / "run-xgb"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "xgboost_model.json"
    model_path.write_text("{}")

    run = SimpleNamespace(
        id="run-xgb",
        config={
            "paradigm": "xgboost",
            "output_dir": "outputs/run-xgb",
        },
        summary_metrics={},
    )

    artifact_path, source = mod.resolve_evaluation_artifact(
        run,
        outputs_root=str(tmp_path),
        task_type="binary",
    )

    assert artifact_path == model_path
    assert source == "xgboost_model"


def test_has_fairness_metrics_requires_requested_attribute_completeness():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    run = SimpleNamespace(
        config={"dataset": "miiv", "task": {"task_type": "binary"}},
        summary_metrics={
            **_binary_fairness_summary("gender", 0.71),
            **_binary_fairness_summary("age_group", 0.69),
        },
    )

    assert mod.has_fairness_metrics(run, ["gender", "age_group"]) is True
    assert mod.has_fairness_metrics(run, ["gender", "age_group", "race"]) is False


def test_has_fairness_metrics_ignores_race_for_eicu():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    run = SimpleNamespace(
        config={"dataset": "eicu", "task": {"task_type": "binary"}},
        summary_metrics={
            **_binary_fairness_summary("gender", 0.71),
            **_binary_fairness_summary("age_group", 0.69),
        },
    )

    assert mod.has_fairness_metrics(run, ["gender", "age_group", "race"]) is True


def test_has_fairness_metrics_rejects_partial_binary_summaries():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    run = SimpleNamespace(
        config={"dataset": "miiv", "task": {"task_type": "binary"}},
        summary_metrics={
            "fairness/gender/n_valid_groups": 2,
            "fairness/gender/worst_group_auroc": 0.71,
        },
    )

    assert mod.has_fairness_metrics(run, ["gender"]) is False


def test_has_fairness_metrics_requires_regression_metric_family():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    complete = SimpleNamespace(
        config={
            "dataset": "miiv",
            "task": {"task_name": "los_remaining", "task_type": "regression"},
        },
        summary_metrics={
            **_regression_fairness_summary("gender"),
            **_regression_fairness_summary("age_group"),
        },
    )
    partial = SimpleNamespace(
        config={
            "dataset": "miiv",
            "task": {"task_name": "los_remaining", "task_type": "regression"},
        },
        summary_metrics={
            "fairness/gender/n_valid_groups": 2,
            "fairness/gender/worst_group_mse": 2.0,
            **_regression_fairness_summary("age_group"),
        },
    )

    assert mod.has_fairness_metrics(complete, ["gender", "age_group"]) is True
    assert mod.has_fairness_metrics(partial, ["gender", "age_group"]) is False
