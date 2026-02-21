"""Tests for decompensation via SlidingWindowDataset with decompensation_pred_hours.

Verifies that per-window binary labels are computed correctly on-the-fly
from stay-level death_hours, and that windows where death occurs during
observation are properly excluded.
"""

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from slices.data.dataset import ICUDataset
from slices.data.sliding_window import SlidingWindowDataset


@pytest.fixture
def decompensation_data(tmp_path):
    """Create mock extracted data with death_hours labels for decompensation testing.

    Creates stays with seq_length=168h so sliding windows (48h) have room to slide.
    """

    def _create(
        stays_info: list[dict],
        seq_length: int = 168,
        n_features: int = 2,
    ):
        """Create test data.

        Args:
            stays_info: List of dicts with keys:
                - stay_id, patient_id, death_hours (float or inf)
            seq_length: Sequence length in hours.
            n_features: Number of features.
        """
        data_dir = tmp_path / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)

        feature_names = [f"feat_{i}" for i in range(n_features)]

        # Metadata
        metadata = {
            "dataset": "mock",
            "feature_set": "core",
            "feature_names": feature_names,
            "n_features": n_features,
            "seq_length_hours": seq_length,
            "min_stay_hours": 48,
            "task_names": ["decompensation"],
            "n_stays": len(stays_info),
        }
        with open(data_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Static
        static_df = pl.DataFrame(
            {
                "stay_id": [s["stay_id"] for s in stays_info],
                "patient_id": [s["patient_id"] for s in stays_info],
                "age": [65] * len(stays_info),
                "gender": ["M"] * len(stays_info),
                "los_days": [seq_length / 24.0] * len(stays_info),
            }
        )
        static_df.write_parquet(data_dir / "static.parquet")

        # Timeseries: random data
        np.random.seed(42)
        ts_data = []
        mask_data = []
        for _ in stays_info:
            ts = np.random.randn(seq_length, n_features).astype(np.float32).tolist()
            mask = np.ones((seq_length, n_features), dtype=bool).tolist()
            ts_data.append(ts)
            mask_data.append(mask)

        ts_df = pl.DataFrame(
            {
                "stay_id": [s["stay_id"] for s in stays_info],
                "timeseries": ts_data,
                "mask": mask_data,
            }
        )
        ts_df.write_parquet(data_dir / "timeseries.parquet")

        # Labels: decompensation column with death_hours (inf for survivors)
        labels_df = pl.DataFrame(
            {
                "stay_id": [s["stay_id"] for s in stays_info],
                "decompensation": [float(s["death_hours"]) for s in stays_info],
            }
        )
        labels_df.write_parquet(data_dir / "labels.parquet")

        return data_dir

    return _create


class TestDecompensationPerWindowLabels:
    """Test per-window binary label computation."""

    def test_survivor_all_labels_zero(self, decompensation_data):
        """Survivor (death_hours=inf) -> all window labels = 0."""
        data_dir = decompensation_data(
            [{"stay_id": 1, "patient_id": 1, "death_hours": float("inf")}]
        )
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=6,
            decompensation_pred_hours=24,
        )

        assert len(sw) > 0
        for i in range(len(sw)):
            sample = sw[i]
            assert sample["label"].item() == 0.0

    def test_death_in_prediction_window_label_1(self, decompensation_data):
        """Death at hour 80: window at t=8, obs_end=56, pred=[56,80) -> death at 80 not in -> 0.
        Window at t=30, obs_end=78, pred=[78,102) -> death at 80 in -> 1.
        """
        data_dir = decompensation_data([{"stay_id": 1, "patient_id": 1, "death_hours": 80.0}])
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=1,
            decompensation_pred_hours=24,
        )

        # Find window at t=30: obs_end=78, pred=[78,102), death=80 -> label=1
        found_t30 = False
        for i in range(len(sw)):
            sample = sw[i]
            if sample["window_start"] == 30:
                assert sample["label"].item() == 1.0, "Window t=30 should have label=1"
                found_t30 = True
                break
        assert found_t30, "Window at t=30 not found"

    def test_death_after_prediction_window_label_0(self, decompensation_data):
        """Death at hour 80: window at t=0, obs_end=48, pred=[48,72) -> label=0."""
        data_dir = decompensation_data([{"stay_id": 1, "patient_id": 1, "death_hours": 80.0}])
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=1,
            decompensation_pred_hours=24,
        )

        # Window at t=0: obs_end=48, pred=[48,72), death=80 -> label=0
        found_t0 = False
        for i in range(len(sw)):
            sample = sw[i]
            if sample["window_start"] == 0:
                assert sample["label"].item() == 0.0, "Window t=0 should have label=0"
                found_t0 = True
                break
        assert found_t0, "Window at t=0 not found"

    def test_death_exactly_at_obs_end(self, decompensation_data):
        """Death at hour 48, t=0: obs=[0,48), death NOT in obs. pred=[48,72), label=1."""
        data_dir = decompensation_data([{"stay_id": 1, "patient_id": 1, "death_hours": 48.0}])
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=48,
            decompensation_pred_hours=24,
        )

        # Window at t=0: death at 48, obs=[0,48), 48 < 48 is False -> not excluded
        # pred=[48,72), 48 <= 48 < 72 -> label=1
        found = False
        for i in range(len(sw)):
            sample = sw[i]
            if sample["window_start"] == 0:
                assert sample["label"].item() == 1.0
                found = True
                break
        assert found, "Window at t=0 should exist"

    def test_death_exactly_at_pred_end(self, decompensation_data):
        """Death at hour 72, t=0: pred=[48,72), death at 72 -> label=0 (half-open)."""
        data_dir = decompensation_data([{"stay_id": 1, "patient_id": 1, "death_hours": 72.0}])
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=48,
            decompensation_pred_hours=24,
        )

        found = False
        for i in range(len(sw)):
            sample = sw[i]
            if sample["window_start"] == 0:
                assert sample["label"].item() == 0.0
                found = True
                break
        assert found


class TestObservationDeathExclusion:
    """Test that windows where death occurs during observation are excluded."""

    def test_death_during_observation_excluded(self, decompensation_data):
        """Death at hour 30: windows where 30 is in [t, t+48) should be absent."""
        data_dir = decompensation_data([{"stay_id": 1, "patient_id": 1, "death_hours": 30.0}])
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=6,
            decompensation_pred_hours=24,
        )

        for i in range(len(sw)):
            sample = sw[i]
            t_start = sample["window_start"]
            obs_end = t_start + 48
            assert not (
                t_start <= 30 < obs_end
            ), f"Window at t={t_start} should be excluded (death at 30 in obs)"

    def test_death_at_window_start_excluded(self, decompensation_data):
        """Death at hour 0 -> window at t=0 is excluded (0 <= 0 < 48)."""
        data_dir = decompensation_data([{"stay_id": 1, "patient_id": 1, "death_hours": 0.0}])
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=48,
            decompensation_pred_hours=24,
        )

        # t=0: death at 0, 0 <= 0 < 48 -> excluded
        for i in range(len(sw)):
            assert sw[i]["window_start"] != 0


class TestWindowCountsDecompensation:
    """Test window count behavior in decompensation mode."""

    def test_stride_affects_window_count(self, decompensation_data):
        """Smaller stride produces more windows."""
        data_dir = decompensation_data(
            [{"stay_id": 1, "patient_id": 1, "death_hours": float("inf")}]
        )
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw_stride6 = SlidingWindowDataset(
            base,
            window_size=48,
            stride=6,
            decompensation_pred_hours=24,
        )
        sw_stride24 = SlidingWindowDataset(
            base,
            window_size=48,
            stride=24,
            decompensation_pred_hours=24,
        )

        assert len(sw_stride6) > len(sw_stride24)

    def test_output_shapes(self, decompensation_data):
        """Check timeseries and mask shapes match (window_size, n_features)."""
        data_dir = decompensation_data(
            [{"stay_id": 1, "patient_id": 1, "death_hours": float("inf")}],
            n_features=3,
        )
        base = ICUDataset(data_dir, task_name="decompensation", normalize=False, train_indices=[0])

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=24,
            decompensation_pred_hours=24,
        )

        sample = sw[0]
        assert sample["timeseries"].shape == (48, 3)
        assert sample["mask"].shape == (48, 3)
        assert sample["label"].dtype == torch.float32
        assert "stay_id" in sample
        assert "window_start" in sample


class TestMultipleStays:
    """Test with multiple stays."""

    def test_mixed_survivors_and_deceased(self, decompensation_data):
        """Mix of survivors and deceased produces correct per-window labels."""
        data_dir = decompensation_data(
            [
                {"stay_id": 1, "patient_id": 1, "death_hours": float("inf")},
                {"stay_id": 2, "patient_id": 2, "death_hours": 60.0},
            ]
        )
        base = ICUDataset(
            data_dir, task_name="decompensation", normalize=False, train_indices=[0, 1]
        )

        sw = SlidingWindowDataset(
            base,
            window_size=48,
            stride=6,
            decompensation_pred_hours=24,
        )

        # Survivor (stay 1): all labels should be 0
        stay1_labels = []
        stay2_labels = []
        for i in range(len(sw)):
            sample = sw[i]
            if sample["stay_id"] == 1:
                stay1_labels.append(sample["label"].item())
            elif sample["stay_id"] == 2:
                stay2_labels.append(sample["label"].item())

        assert all(l == 0.0 for l in stay1_labels), "Survivor should have all label=0"
        assert any(l == 1.0 for l in stay2_labels), "Deceased should have some label=1"
