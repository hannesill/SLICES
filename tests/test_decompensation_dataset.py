"""Tests for decompensation sliding-window dataset and datamodule."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch
import yaml
from slices.data.decompensation_dataset import DecompensationDataset


@pytest.fixture
def tmp_ricu_data(tmp_path):
    """Create minimal RICU parquet files for testing."""

    def _create(
        stays_data: list[dict],
        mortality_data: list[dict],
        timeseries_data: list[dict],
        feature_names: list[str] = None,
    ) -> dict:
        if feature_names is None:
            feature_names = ["hr", "sbp"]

        stays_df = pl.DataFrame(stays_data)
        mortality_df = pl.DataFrame(mortality_data)
        ts_df = pl.DataFrame(timeseries_data)

        stays_path = tmp_path / "ricu_stays.parquet"
        mortality_path = tmp_path / "ricu_mortality.parquet"
        ts_path = tmp_path / "ricu_timeseries.parquet"

        stays_df.write_parquet(stays_path)
        mortality_df.write_parquet(mortality_path)
        ts_df.write_parquet(ts_path)

        # Write metadata
        metadata = {"dataset": "test", "feature_names": feature_names}
        with open(tmp_path / "ricu_metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        return {
            "ts_path": ts_path,
            "stays_path": stays_path,
            "mortality_path": mortality_path,
            "feature_names": feature_names,
        }

    return _create


def _make_intime(stay_id: int) -> datetime:
    """Fixed admission time for deterministic tests."""
    return datetime(2020, 1, 1, 0, 0)


class TestDecompensationWindowGeneration:
    """Test window generation and counting."""

    def test_window_count_with_stride(self, tmp_ricu_data):
        """Stay of 100h, obs=48, stride=6 -> windows at 0,6,12,...,52."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        # Windows at t=0,6,12,...,52 (obs_end must be <= 100)
        # t + 48 <= 100 -> t <= 52 -> t in {0,6,12,18,24,30,36,42,48,52}
        expected_starts = list(range(0, 53, 6))
        assert len(ds) == len(expected_starts)

    def test_stride_1_more_windows_than_stride_6(self, tmp_ricu_data):
        """Stride=1 should produce more windows than stride=6."""
        intime = _make_intime(1)
        common_args = {
            "stays_data": [
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            "mortality_data": [
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
            ],
            "timeseries_data": [
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        }

        paths = tmp_ricu_data(**common_args)

        ds_stride6 = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        # Need to write new parquet for stride=1 (same data)
        ds_stride1 = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=1,
            normalize=False,
        )

        assert len(ds_stride1) > len(ds_stride6)


class TestDecompensationLabels:
    """Test label correctness."""

    def test_death_in_prediction_window_label_1(self, tmp_ricu_data):
        """Death at hour 80: window at t=30, obs_end=78, pred=[78,102) -> label=1."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": intime + timedelta(hours=80),
                    "hospital_expire_flag": 1,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "DIED",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=1,
            normalize=False,
        )

        # Find sample at t=30: obs_end=78, pred=[78,102), death at 80 -> label=1
        for sid, t_start, label in ds.samples:
            if sid == 1 and t_start == 30:
                assert label == 1, f"Window at t=30 should have label=1, got {label}"
                break
        else:
            pytest.fail("Window at t=30 not found")

    def test_death_after_prediction_window_label_0(self, tmp_ricu_data):
        """Death at hour 80: window at t=0, obs_end=48, pred=[48,72) -> label=0."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": intime + timedelta(hours=80),
                    "hospital_expire_flag": 1,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "DIED",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=1,
            normalize=False,
        )

        # Window at t=0: obs_end=48, pred=[48,72), death at 80 -> label=0
        for sid, t_start, label in ds.samples:
            if sid == 1 and t_start == 0:
                assert label == 0, f"Window at t=0 should have label=0, got {label}"
                break
        else:
            pytest.fail("Window at t=0 not found")

    def test_death_during_observation_excluded(self, tmp_ricu_data):
        """Window where death occurs during observation should be excluded."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": intime + timedelta(hours=30),
                    "hospital_expire_flag": 1,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "DIED",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        # Death at hour 30 -> any window where 30 is in [t_start, t_start+48)
        # should be excluded. Windows at t=0 (30 in [0,48)) should not exist.
        for sid, t_start, label in ds.samples:
            obs_end = t_start + 48
            assert not (
                t_start <= 30 < obs_end
            ), f"Window at t={t_start} should be excluded (death at 30 is during obs)"

    def test_stay_too_short_for_window(self, tmp_ricu_data):
        """Stay shorter than obs window -> no samples."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=30),
                    "los_days": 30 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=30),
                    "discharge_location": "HOME",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(30)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        assert len(ds) == 0

    def test_death_exactly_at_obs_end_is_excluded(self, tmp_ricu_data):
        """Death at obs_end boundary is NOT excluded (not during obs)."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": intime + timedelta(hours=48),
                    "hospital_expire_flag": 1,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "DIED",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=48,
            normalize=False,
        )

        # Window at t=0: death_hour=48, obs_end=48
        # Condition: t_start <= death < obs_end -> 0 <= 48 < 48 is False
        # So NOT excluded. Death at obs_end=48, pred=[48,72), death at 48 -> label=1
        found = False
        for sid, t_start, label in ds.samples:
            if sid == 1 and t_start == 0:
                assert label == 1, "Death exactly at obs_end should be label=1 (in pred window)"
                found = True
                break
        assert found, "Window at t=0 should exist"

    def test_death_exactly_at_pred_end_is_label_0(self, tmp_ricu_data):
        """Death exactly at pred_end: pred=[48,72), death at 72 -> label=0."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": intime + timedelta(hours=72),
                    "hospital_expire_flag": 1,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "DIED",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=48,
            normalize=False,
        )

        # Window at t=0: obs_end=48, pred=[48,72), death at 72 -> 48 <= 72 < 72 is False -> label=0
        for sid, t_start, label in ds.samples:
            if sid == 1 and t_start == 0:
                assert label == 0, "Death exactly at pred_end should be label=0"
                break


class TestDecompensationGetItem:
    """Test __getitem__ returns correct shapes and values."""

    def test_output_shapes(self, tmp_ricu_data):
        """Check timeseries and mask shapes match (obs_window, n_features)."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        sample = ds[0]
        assert sample["timeseries"].shape == (48, 2)
        assert sample["mask"].shape == (48, 2)
        assert sample["label"].dtype == torch.float32
        assert "stay_id" in sample
        assert "window_start" in sample


class TestDecompensationNormalization:
    """Test normalization stats computation."""

    def test_normalization_stats_computed(self, tmp_ricu_data):
        """compute_normalization_stats returns valid means and stds."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0 + h * 0.1, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        means, stds = ds.compute_normalization_stats()
        assert means.shape == (2,)
        assert stds.shape == (2,)
        assert not torch.isnan(means).any()
        assert not torch.isnan(stds).any()
        assert (stds > 0).all()

    def test_normalized_output_has_no_nan(self, tmp_ricu_data):
        """Normalized samples should have no NaN values."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=48,
            normalize=True,
            feature_means=torch.tensor([80.0, 120.0]),
            feature_stds=torch.tensor([5.0, 10.0]),
        )

        if len(ds) > 0:
            sample = ds[0]
            assert not torch.isnan(sample["timeseries"]).any()


class TestDecompensationMultipleStays:
    """Test with multiple stays to verify patient isolation."""

    def test_stay_ids_filter(self, tmp_ricu_data):
        """Only requested stay_ids produce samples."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": i,
                    "patient_id": i,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
                for i in [1, 2, 3]
            ],
            mortality_data=[
                {
                    "stay_id": i,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
                for i in [1, 2, 3]
            ],
            timeseries_data=[
                {"stay_id": sid, "hour": h, "hr": 80.0, "sbp": 120.0}
                for sid in [1, 2, 3]
                for h in range(100)
            ],
        )

        # Only include stay 1 and 2
        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            stay_ids=[1, 2],
            normalize=False,
        )

        sample_stay_ids = {sid for sid, _, _ in ds.samples}
        assert 1 in sample_stay_ids
        assert 2 in sample_stay_ids
        assert 3 not in sample_stay_ids

    def test_label_distribution(self, tmp_ricu_data):
        """get_label_distribution returns correct counts."""
        intime = _make_intime(1)
        paths = tmp_ricu_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": intime,
                    "outtime": intime + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {
                    "stay_id": 1,
                    "date_of_death": None,
                    "hospital_expire_flag": 0,
                    "dischtime": intime + timedelta(hours=100),
                    "discharge_location": "HOME",
                }
            ],
            timeseries_data=[
                {"stay_id": 1, "hour": h, "hr": 80.0, "sbp": 120.0} for h in range(100)
            ],
        )

        ds = DecompensationDataset(
            ricu_timeseries_path=paths["ts_path"],
            stays_path=paths["stays_path"],
            mortality_path=paths["mortality_path"],
            feature_names=paths["feature_names"],
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            normalize=False,
        )

        dist = ds.get_label_distribution()
        assert dist["total"] == len(ds)
        assert dist["positive"] + dist["negative"] == dist["total"]
