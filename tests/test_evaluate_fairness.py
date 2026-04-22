"""Focused tests for the standalone fairness rerun script."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_default_phases_include_baseline():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert "baseline" in mod.DEFAULT_PHASES


def test_default_core_sprints_include_sprint11():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert "11" in mod.CORE_SPRINTS


def test_default_core_sprints_include_ablation_and_transfer_sprints():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert {"6", "7", "8"}.issubset(set(mod.CORE_SPRINTS))


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
        config={"dataset": "miiv"},
        summary_metrics={
            "fairness/gender/n_valid_groups": 2,
            "fairness/gender/worst_group_auroc": 0.71,
            "fairness/age_group/n_valid_groups": 4,
            "fairness/age_group/worst_group_auroc": 0.69,
        },
    )

    assert mod.has_fairness_metrics(run, ["gender", "age_group"]) is True
    assert mod.has_fairness_metrics(run, ["gender", "age_group", "race"]) is False


def test_has_fairness_metrics_ignores_race_for_eicu():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    run = SimpleNamespace(
        config={"dataset": "eicu"},
        summary_metrics={
            "fairness/gender/n_valid_groups": 2,
            "fairness/gender/worst_group_auroc": 0.71,
            "fairness/age_group/n_valid_groups": 4,
            "fairness/age_group/worst_group_auroc": 0.69,
        },
    )

    assert mod.has_fairness_metrics(run, ["gender", "age_group", "race"]) is True
