"""Integration tests for extractor label extraction framework."""

from datetime import datetime
from typing import Dict, List

import polars as pl
import pytest
from slices.data.extractors.base import BaseExtractor, ExtractorConfig
from slices.data.labels import LabelConfig


class _MockExtractor(BaseExtractor):
    """Minimal concrete extractor for testing the label extraction framework."""

    def __init__(self, config: ExtractorConfig) -> None:
        super().__init__(config)
        self._stays_df: pl.DataFrame = pl.DataFrame()
        self._data_sources: Dict[str, pl.DataFrame] = {}

    def _get_dataset_name(self) -> str:
        return "mock"

    def extract_stays(self) -> pl.DataFrame:
        return self._stays_df

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        return pl.DataFrame({"stay_id": stay_ids, "hour": [0] * len(stay_ids)})

    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        if source_name not in self._data_sources:
            raise ValueError(f"Unknown data source '{source_name}'")
        return self._data_sources[source_name].filter(pl.col("stay_id").is_in(stay_ids))

    def run(self) -> None:
        pass


@pytest.fixture
def mock_extractor(tmp_path):
    """Create a mock extractor with temporary paths."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    config = ExtractorConfig(
        parquet_root=str(tmp_path / "data"),
        output_dir=str(tmp_path / "processed"),
    )
    return _MockExtractor(config)


class TestExtractorIntegration:
    """Integration tests for label extraction framework."""

    def test_extract_data_source_unknown_source(self, mock_extractor):
        """Test that extract_data_source raises error for unknown source."""
        with pytest.raises(ValueError, match="Unknown data source"):
            mock_extractor.extract_data_source("unknown_source", [1, 2, 3])

    def test_extract_labels_calls_extract_data_source(self, mock_extractor):
        """Test that extract_labels properly calls extract_data_source."""
        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "patient_id": [100, 200],
                "intime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 2, 12, 0)],
                "outtime": [datetime(2020, 1, 3, 10, 0), datetime(2020, 1, 5, 12, 0)],
                "length_of_stay_days": [2.0, 3.0],
            }
        )
        mock_extractor._stays_df = stays_df

        mortality_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "date_of_death": [datetime(2020, 1, 1, 20, 0), None],
                "hospital_expire_flag": [1, 0],
                "dischtime": [datetime(2020, 1, 3, 10, 0), datetime(2020, 1, 5, 12, 0)],
                "discharge_location": ["DIED", "HOME"],
            }
        )
        mock_extractor._data_sources["mortality_info"] = mortality_df

        task_config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        labels = mock_extractor.extract_labels([1, 2], [task_config])

        assert "stay_id" in labels.columns
        assert "mortality_24h" in labels.columns
        assert labels.shape[0] == 2

    def test_extract_multiple_tasks_no_column_conflicts(self, mock_extractor):
        """Test extracting multiple tasks at once - critical for real-world usage."""
        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 200, 300],
                "intime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 2, 12, 0),
                    datetime(2020, 1, 3, 8, 0),
                ],
                "outtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 12, 0),
                    datetime(2020, 1, 6, 8, 0),
                ],
                "length_of_stay_days": [2.0, 3.0, 3.0],
            }
        )
        mock_extractor._stays_df = stays_df

        mortality_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [
                    datetime(2020, 1, 1, 20, 0),  # 10h after admission (within 24h)
                    datetime(2020, 1, 4, 13, 0),  # 49h after admission (outside 24h)
                    None,  # Survived
                ],
                "hospital_expire_flag": [1, 1, 0],
                "dischtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 12, 0),
                    datetime(2020, 1, 6, 8, 0),
                ],
                "discharge_location": ["DIED", "DIED", "HOME"],
            }
        )
        mock_extractor._data_sources["mortality_info"] = mortality_df

        task_configs = [
            LabelConfig(
                task_name="mortality_24h",
                task_type="binary_classification",
                prediction_window_hours=24,
                label_sources=["stays", "mortality_info"],
            ),
            LabelConfig(
                task_name="mortality_hospital",
                task_type="binary_classification",
                prediction_window_hours=None,
                label_sources=["stays", "mortality_info"],
            ),
        ]

        labels = mock_extractor.extract_labels([1, 2, 3], task_configs)

        # Verify all columns exist and are unique
        assert "stay_id" in labels.columns
        assert "mortality_24h" in labels.columns
        assert "mortality_hospital" in labels.columns

        # Verify no duplicate columns
        assert len(labels.columns) == len(
            set(labels.columns)
        ), f"Duplicate columns found: {labels.columns}"

        # Verify shape
        assert labels.shape == (3, 3), f"Expected (3, 3), got {labels.shape}"

        # Verify label values for stay 1 (died at 10h)
        row1 = labels.filter(pl.col("stay_id") == 1)
        assert row1["mortality_24h"][0] == 1  # Within 24h
        assert row1["mortality_hospital"][0] == 1  # Died in hospital

        # Verify label values for stay 2 (died at 49h)
        row2 = labels.filter(pl.col("stay_id") == 2)
        assert row2["mortality_24h"][0] == 0  # Not within 24h (died at 49h)
        assert row2["mortality_hospital"][0] == 1  # Died in hospital

        # Verify label values for stay 3 (survived)
        row3 = labels.filter(pl.col("stay_id") == 3)
        assert row3["mortality_24h"][0] == 0
        assert row3["mortality_hospital"][0] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_extract_labels_with_empty_stay_list(self, mock_extractor):
        """Test extracting labels for empty stay list."""
        empty_stays_df = pl.DataFrame(
            {
                "stay_id": [],
                "patient_id": [],
                "intime": [],
                "outtime": [],
                "length_of_stay_days": [],
            }
        ).cast({"stay_id": pl.Int64, "patient_id": pl.Int64, "length_of_stay_days": pl.Float64})
        mock_extractor._stays_df = empty_stays_df

        empty_mortality_df = pl.DataFrame(
            {
                "stay_id": [],
                "date_of_death": [],
                "hospital_expire_flag": [],
                "dischtime": [],
                "discharge_location": [],
            }
        ).cast({"stay_id": pl.Int64, "hospital_expire_flag": pl.Int32})
        mock_extractor._data_sources["mortality_info"] = empty_mortality_df

        task_config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        labels = mock_extractor.extract_labels([], [task_config])

        assert "stay_id" in labels.columns
        assert "mortality_24h" in labels.columns
        assert labels.shape[0] == 0

    def test_extract_labels_no_tasks(self, mock_extractor):
        """Test extracting labels with no task configs provided."""
        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 200, 300],
                "intime": [datetime(2020, 1, 1)] * 3,
                "outtime": [datetime(2020, 1, 2)] * 3,
                "length_of_stay_days": [1.0] * 3,
            }
        )
        mock_extractor._stays_df = stays_df

        labels = mock_extractor.extract_labels([1, 2, 3], [])

        assert list(labels.columns) == ["stay_id"]
        assert labels.shape == (3, 1)

    def test_extract_labels_with_nulls_in_mortality_data(self, mock_extractor):
        """Test handling of null values in mortality data."""
        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 200, 300],
                "intime": [datetime(2020, 1, 1, 10, 0)] * 3,
                "outtime": [datetime(2020, 1, 3, 10, 0)] * 3,
                "length_of_stay_days": [2.0] * 3,
            }
        )
        mock_extractor._stays_df = stays_df

        mortality_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [None, None, None],
                "hospital_expire_flag": [0, 0, 0],
                "dischtime": [datetime(2020, 1, 3, 10, 0)] * 3,
                "discharge_location": ["HOME"] * 3,
            }
        )
        mock_extractor._data_sources["mortality_info"] = mortality_df

        task_config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            label_sources=["stays", "mortality_info"],
        )

        labels = mock_extractor.extract_labels([1, 2, 3], [task_config])

        assert labels["mortality_hospital"].to_list() == [0, 0, 0]

    def test_many_tasks_extraction(self, mock_extractor):
        """Stress test: extract many tasks at once to verify join performance."""
        num_stays = 100
        stays_df = pl.DataFrame(
            {
                "stay_id": list(range(1, num_stays + 1)),
                "patient_id": list(range(100, 100 + num_stays)),
                "intime": [datetime(2020, 1, 1, 10, 0)] * num_stays,
                "outtime": [datetime(2020, 1, 3, 10, 0)] * num_stays,
                "length_of_stay_days": [2.0] * num_stays,
            }
        )
        mock_extractor._stays_df = stays_df

        mortality_df = pl.DataFrame(
            {
                "stay_id": list(range(1, num_stays + 1)),
                "date_of_death": [None] * num_stays,
                "hospital_expire_flag": [0] * num_stays,
                "dischtime": [datetime(2020, 1, 3, 10, 0)] * num_stays,
                "discharge_location": ["HOME"] * num_stays,
            }
        )
        mock_extractor._data_sources["mortality_info"] = mortality_df

        task_configs = [
            LabelConfig(
                task_name=f"mortality_task_{i}",
                task_type="binary_classification",
                prediction_window_hours=24 * i if i > 0 else None,
                label_sources=["stays", "mortality_info"],
            )
            for i in range(10)
        ]

        labels = mock_extractor.extract_labels(list(range(1, num_stays + 1)), task_configs)

        # Verify shape: 100 stays x (1 stay_id + 10 task columns)
        assert labels.shape == (num_stays, 11)

        # Verify all task columns exist
        for i in range(10):
            assert f"mortality_task_{i}" in labels.columns

        # Verify no duplicate columns
        assert len(labels.columns) == len(set(labels.columns))
