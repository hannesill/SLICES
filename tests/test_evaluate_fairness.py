"""Focused tests for the standalone fairness rerun script."""

import importlib
from types import SimpleNamespace


def test_default_phases_include_baseline():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert "baseline" in mod.DEFAULT_PHASES


def test_default_core_sprints_include_sprint11():
    mod = importlib.import_module("scripts.eval.evaluate_fairness")

    assert "11" in mod.CORE_SPRINTS


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
