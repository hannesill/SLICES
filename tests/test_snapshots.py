from pathlib import Path

import polars as pl
from slices.debug.snapshots import (
    LegacyPipelineStage,
    capture_dense_snapshot,
    capture_labels_snapshot,
    capture_stays_snapshot,
    export_snapshot,
)


def test_legacy_snapshot_helpers_use_valid_stages(tmp_path: Path) -> None:
    stays = pl.DataFrame({"stay_id": [1, 2], "subject_id": [10, 20]})
    labels = pl.DataFrame({"stay_id": [1, 2], "mortality_24h": [0, 1]})
    dense = pl.DataFrame(
        {
            "stay_id": [1],
            "timeseries": [[[1.0, 2.0], [3.0, 4.0]]],
            "mask": [[[True, False], [True, True]]],
        }
    )

    stays_snapshot = capture_stays_snapshot(stays, stay_ids=[1])
    labels_snapshot = capture_labels_snapshot(labels, stay_ids=[1])
    dense_snapshot = capture_dense_snapshot(
        dense,
        feature_names=["heart_rate", "spo2"],
        stay_ids=[1],
        flatten=True,
    )

    assert stays_snapshot.stage == LegacyPipelineStage.STAYS
    assert labels_snapshot.stage == LegacyPipelineStage.LABELS
    assert dense_snapshot.stage == LegacyPipelineStage.DENSE_TIMESERIES
    assert dense_snapshot.data.height == 4

    exported_path = export_snapshot(stays_snapshot, tmp_path)
    assert exported_path == tmp_path / "stays.csv"
    assert exported_path.exists()
    assert (tmp_path / "stays_metadata.yaml").exists()
