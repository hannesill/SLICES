"""Tests for label analysis script helpers."""

import polars as pl

from scripts.analyze_labels import analyze_task


def test_analyze_task_warns_when_missing_rate_exceeds_quality_threshold():
    labels_df = pl.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "aki_kdigo": [1, 0, None, None, None],
        }
    )

    stats = analyze_task(
        labels_df,
        "aki_kdigo",
        quality_checks={"max_missing_percentage": 50.0},
    )

    assert stats["missing_percentage"] == 60.0
    assert len(stats["quality_warnings"]) == 1
    assert "60.0%" in stats["quality_warnings"][0]
    assert "50.0%" in stats["quality_warnings"][0]


def test_analyze_task_has_no_quality_warning_below_threshold():
    labels_df = pl.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "aki_kdigo": [1, 0, 0, None, 1],
        }
    )

    stats = analyze_task(
        labels_df,
        "aki_kdigo",
        quality_checks={"max_missing_percentage": 28.0},
    )

    assert stats["missing_percentage"] == 20.0
    assert stats["quality_warnings"] == []


def test_analyze_task_summarizes_regression_labels_without_class_casting():
    labels_df = pl.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "los_remaining": [0.5, 2.0, None, 7.5],
        }
    )

    stats = analyze_task(labels_df, "los_remaining", task_type="regression")

    assert stats["task_type"] == "regression"
    assert stats["n_classes"] is None
    assert stats["class_distribution"] == {}
    assert stats["label_summary"]["mean"] == 3.3333
    assert stats["label_summary"]["median"] == 2.0
