"""Tests for decompensation sliding-window dataset and datamodule."""

from datetime import datetime, timedelta

import polars as pl
import pytest
import torch
import yaml
from slices.data.decompensation_datamodule import DecompensationDataModule
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


@pytest.fixture
def tmp_ricu_multipatient(tmp_path):
    """Create RICU parquet data with multiple patients (some sharing patient_id)."""
    intime = datetime(2020, 1, 1, 0, 0)
    feature_names = ["hr", "sbp"]

    # 6 stays, 4 patients. Patient 10 has stays 1,2; patient 20 has stays 3,4;
    # patient 30 has stay 5; patient 40 has stay 6.
    stays_data = [
        {
            "stay_id": 1,
            "patient_id": 10,
            "intime": intime,
            "outtime": intime + timedelta(hours=100),
            "los_days": 100 / 24.0,
        },
        {
            "stay_id": 2,
            "patient_id": 10,
            "intime": intime,
            "outtime": intime + timedelta(hours=100),
            "los_days": 100 / 24.0,
        },
        {
            "stay_id": 3,
            "patient_id": 20,
            "intime": intime,
            "outtime": intime + timedelta(hours=100),
            "los_days": 100 / 24.0,
        },
        {
            "stay_id": 4,
            "patient_id": 20,
            "intime": intime,
            "outtime": intime + timedelta(hours=100),
            "los_days": 100 / 24.0,
        },
        {
            "stay_id": 5,
            "patient_id": 30,
            "intime": intime,
            "outtime": intime + timedelta(hours=100),
            "los_days": 100 / 24.0,
        },
        {
            "stay_id": 6,
            "patient_id": 40,
            "intime": intime,
            "outtime": intime + timedelta(hours=100),
            "los_days": 100 / 24.0,
        },
    ]
    mortality_data = [
        {
            "stay_id": sid,
            "date_of_death": None,
            "hospital_expire_flag": 0,
            "dischtime": intime + timedelta(hours=100),
            "discharge_location": "HOME",
        }
        for sid in range(1, 7)
    ]
    timeseries_data = [
        {"stay_id": sid, "hour": h, "hr": 80.0 + h * 0.1 * sid, "sbp": 120.0 + sid}
        for sid in range(1, 7)
        for h in range(100)
    ]

    pl.DataFrame(stays_data).write_parquet(tmp_path / "ricu_stays.parquet")
    pl.DataFrame(mortality_data).write_parquet(tmp_path / "ricu_mortality.parquet")
    pl.DataFrame(timeseries_data).write_parquet(tmp_path / "ricu_timeseries.parquet")

    metadata = {"dataset": "test", "feature_names": feature_names}
    with open(tmp_path / "ricu_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    return tmp_path


class TestDecompensationDataModulePatientSplits:
    """Test patient-level split integrity in DataModule."""

    def test_no_patient_leakage_across_splits(self, tmp_ricu_multipatient):
        """All stays from the same patient must be in the same split."""
        dm = DecompensationDataModule(
            ricu_parquet_root=tmp_ricu_multipatient,
            processed_dir=tmp_ricu_multipatient,
            batch_size=4,
            num_workers=0,
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            eval_stride_hours=6,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
        dm.setup()

        # Collect stay_ids per split
        train_sids = {sid for sid, _, _ in dm.train_dataset.samples}
        val_sids = {sid for sid, _, _ in dm.val_dataset.samples}
        test_sids = {sid for sid, _, _ in dm.test_dataset.samples}

        # No overlap between splits
        assert train_sids.isdisjoint(val_sids), "Train/val stay_id overlap"
        assert train_sids.isdisjoint(test_sids), "Train/test stay_id overlap"
        assert val_sids.isdisjoint(test_sids), "Val/test stay_id overlap"

        # Check patient-level: stays sharing a patient_id must be in same split
        stays_df = pl.read_parquet(tmp_ricu_multipatient / "ricu_stays.parquet")
        sid_to_pid = dict(zip(stays_df["stay_id"].to_list(), stays_df["patient_id"].to_list()))

        all_assigned_sids = train_sids | val_sids | test_sids
        for split_name, split_sids in [
            ("train", train_sids),
            ("val", val_sids),
            ("test", test_sids),
        ]:
            patients_in_split = {sid_to_pid[sid] for sid in split_sids if sid in sid_to_pid}
            for sid in all_assigned_sids:
                if sid in sid_to_pid and sid_to_pid[sid] in patients_in_split:
                    assert sid in split_sids, (
                        f"Stay {sid} (patient {sid_to_pid[sid]}) should be in {split_name} "
                        f"but is missing — patient leakage"
                    )

    def test_all_stays_assigned(self, tmp_ricu_multipatient):
        """Every stay should be assigned to exactly one split."""
        dm = DecompensationDataModule(
            ricu_parquet_root=tmp_ricu_multipatient,
            processed_dir=tmp_ricu_multipatient,
            batch_size=4,
            num_workers=0,
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            eval_stride_hours=6,
            seed=42,
        )
        dm.setup()

        info = dm.get_split_info()
        stays_df = pl.read_parquet(tmp_ricu_multipatient / "ricu_stays.parquet")
        total_stays = len(stays_df)
        assert info["train_stays"] + info["val_stays"] + info["test_stays"] == total_stays


class TestDecompensationDataModuleNormalization:
    """Test normalization stats are computed from training data only."""

    def test_normalization_from_train_only(self, tmp_ricu_multipatient):
        """Val/test datasets should use stats computed from train set."""
        dm = DecompensationDataModule(
            ricu_parquet_root=tmp_ricu_multipatient,
            processed_dir=tmp_ricu_multipatient,
            batch_size=4,
            num_workers=0,
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=6,
            eval_stride_hours=6,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
        dm.setup()

        # Train dataset should have normalization enabled with its own stats
        assert dm.train_dataset.normalize is True
        assert dm.train_dataset.feature_means is not None
        assert dm.train_dataset.feature_stds is not None

        # Val/test should share the exact same stats objects
        assert dm.val_dataset.feature_means is dm._feature_means
        assert dm.val_dataset.feature_stds is dm._feature_stds
        assert dm.test_dataset.feature_means is dm._feature_means
        assert dm.test_dataset.feature_stds is dm._feature_stds

        # Stats should match what train computes independently
        train_means, train_stds = dm.train_dataset.compute_normalization_stats()
        # Note: train_dataset already has normalization set, but compute_normalization_stats
        # operates on raw data, so we compare the stored stats
        assert torch.allclose(dm._feature_means, train_means, atol=1e-5)
        assert torch.allclose(dm._feature_stds, train_stds, atol=1e-5)


class TestDecompensationDataModuleStrides:
    """Test different strides for train vs eval."""

    def test_train_uses_training_stride(self, tmp_ricu_multipatient):
        """Train dataset should use stride_hours, not eval_stride_hours."""
        dm = DecompensationDataModule(
            ricu_parquet_root=tmp_ricu_multipatient,
            processed_dir=tmp_ricu_multipatient,
            batch_size=4,
            num_workers=0,
            obs_window_hours=48,
            pred_window_hours=24,
            stride_hours=12,
            eval_stride_hours=3,
            seed=42,
        )
        dm.setup()

        assert dm.train_dataset.stride_hours == 12
        assert dm.val_dataset.stride_hours == 3
        assert dm.test_dataset.stride_hours == 3

        # Eval sets should have more samples per stay than train
        train_samples = len(dm.train_dataset)
        val_samples = len(dm.val_dataset)
        # val has fewer stays but finer stride, so ratio per stay is higher
        train_stays = len({sid for sid, _, _ in dm.train_dataset.samples})
        val_stays = len({sid for sid, _, _ in dm.val_dataset.samples})
        if train_stays > 0 and val_stays > 0:
            train_per_stay = train_samples / train_stays
            val_per_stay = val_samples / val_stays
            assert val_per_stay > train_per_stay, (
                f"Val should have more samples per stay ({val_per_stay:.1f}) "
                f"than train ({train_per_stay:.1f}) due to finer stride"
            )
