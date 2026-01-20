"""Tests for slices/data/config_loader.py.

Tests YAML configuration loading for datasets and concepts.
"""

import pytest
import yaml
from slices.data.config_loader import (
    TIMESERIES_FILES,
    _parse_extraction_source,
    _parse_static_source,
    _parse_timeseries_concept,
    get_feature_names,
    get_static_feature_names,
    load_dataset_config,
    load_static_concepts,
    load_timeseries_concepts,
)
from slices.data.config_schemas import (
    ColumnSource,
    ItemIDSource,
    RegexMatchSource,
    StringMatchSource,
)


class TestParseExtractionSource:
    """Tests for parsing extraction source dictionaries."""

    def test_parse_itemid_source(self):
        """Parsing itemid source should return ItemIDSource."""
        source_dict = {
            "type": "itemid",
            "table": "icu/chartevents",
            "itemid": [220045],
            "time_col": "charttime",
            "value_col": "valuenum",
        }
        result = _parse_extraction_source(source_dict)

        assert isinstance(result, ItemIDSource)
        assert result.table == "icu/chartevents"
        assert result.itemid == [220045]

    def test_parse_itemid_source_default_type(self):
        """Parsing without type should default to itemid."""
        source_dict = {
            "table": "icu/chartevents",
            "itemid": [220045],
            "time_col": "charttime",
        }
        result = _parse_extraction_source(source_dict)

        assert isinstance(result, ItemIDSource)

    def test_parse_column_source(self):
        """Parsing column source should return ColumnSource."""
        source_dict = {
            "type": "column",
            "table": "vitalperiodic",
            "value_col": "heartrate",
            "time_col": "observationoffset",
        }
        result = _parse_extraction_source(source_dict)

        assert isinstance(result, ColumnSource)
        assert result.table == "vitalperiodic"
        assert result.value_col == "heartrate"

    def test_parse_string_source(self):
        """Parsing string match source should return StringMatchSource."""
        source_dict = {
            "type": "string",
            "table": "lab",
            "match_col": "labname",
            "match_value": "glucose",
            "value_col": "labresult",
            "time_col": "labresultoffset",
        }
        result = _parse_extraction_source(source_dict)

        assert isinstance(result, StringMatchSource)
        assert result.match_value == "glucose"

    def test_parse_regex_source(self):
        """Parsing regex match source should return RegexMatchSource."""
        source_dict = {
            "type": "regex",
            "table": "medication",
            "match_col": "drugname",
            "pattern": "(?i)vancomycin",
            "time_col": "drugstartoffset",
        }
        result = _parse_extraction_source(source_dict)

        assert isinstance(result, RegexMatchSource)
        assert result.pattern == "(?i)vancomycin"

    def test_parse_unknown_source_raises(self):
        """Parsing unknown source type should raise ValueError."""
        source_dict = {
            "type": "unknown_type",
            "table": "some_table",
        }
        with pytest.raises(ValueError, match="Unknown extraction source type"):
            _parse_extraction_source(source_dict)


class TestLoadDatasetConfig:
    """Tests for loading dataset configuration files."""

    def test_load_valid_dataset_config(self, tmp_path):
        """Loading valid dataset config should return DatasetConfig."""
        # Create test config
        config_data = {
            "name": "test_dataset",
            "description": "Test dataset for unit tests",
            "time": {
                "format": "timestamp",
                "reference_table": "icu/icustays",
                "reference_col": "intime",
            },
            "ids": {
                "stay": "stay_id",
                "patient": "subject_id",
                "admission": "hadm_id",
            },
            "tables": {
                "chartevents": "icu/chartevents",
                "labevents": "hosp/labevents",
            },
        }

        config_path = tmp_path / "test_dataset.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = load_dataset_config(tmp_path, "test_dataset")

        assert result.name == "test_dataset"
        assert result.time.format == "timestamp"
        assert result.ids.stay == "stay_id"
        assert "chartevents" in result.tables

    def test_load_missing_config_raises(self, tmp_path):
        """Loading missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset config not found"):
            load_dataset_config(tmp_path, "nonexistent")


class TestParseTimeseriesConcept:
    """Tests for parsing time-series concept configurations."""

    def test_parse_concept_with_dataset(self):
        """Parsing concept with dataset sources should return config."""
        config_dict = {
            "description": "Heart rate",
            "units": "bpm",
            "min": 0,
            "max": 300,
            "feature_set": ["core"],
            "mimic_iv": [
                {
                    "type": "itemid",
                    "table": "icu/chartevents",
                    "itemid": [220045],
                    "time_col": "charttime",
                }
            ],
        }

        result = _parse_timeseries_concept(config_dict, "mimic_iv")

        assert result is not None
        assert result.description == "Heart rate"
        assert result.units == "bpm"
        assert len(result.mimic_iv) == 1
        assert isinstance(result.mimic_iv[0], ItemIDSource)

    def test_parse_concept_missing_dataset_returns_none(self):
        """Parsing concept without requested dataset should return None."""
        config_dict = {
            "description": "Heart rate",
            "eicu": [
                {
                    "type": "column",
                    "table": "vitalperiodic",
                    "value_col": "heartrate",
                    "time_col": "observationoffset",
                }
            ],
        }

        result = _parse_timeseries_concept(config_dict, "mimic_iv")

        assert result is None

    def test_parse_concept_with_multiple_sources(self):
        """Parsing concept with multiple sources should include all."""
        config_dict = {
            "description": "Temperature",
            "mimic_iv": [
                {
                    "type": "itemid",
                    "table": "icu/chartevents",
                    "itemid": [223761],  # Fahrenheit
                    "time_col": "charttime",
                    "transform": "fahrenheit_to_celsius",
                },
                {
                    "type": "itemid",
                    "table": "icu/chartevents",
                    "itemid": [223762],  # Celsius
                    "time_col": "charttime",
                },
            ],
        }

        result = _parse_timeseries_concept(config_dict, "mimic_iv")

        assert result is not None
        assert len(result.mimic_iv) == 2


class TestLoadTimeseriesConcepts:
    """Tests for loading time-series concepts from YAML files."""

    def test_load_concepts_filters_by_feature_set(self, tmp_path):
        """Loading concepts should filter by feature set."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        # Create vitals.yaml with core and extended features
        vitals_data = {
            "heart_rate": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220045],
                        "time_col": "charttime",
                    }
                ],
            },
            "cvp": {
                "feature_set": ["extended"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220074],
                        "time_col": "charttime",
                    }
                ],
            },
        }

        with open(concepts_dir / "vitals.yaml", "w") as f:
            yaml.dump(vitals_data, f)

        # Load core features only
        core_concepts = load_timeseries_concepts(concepts_dir, "mimic_iv", "core")
        assert "heart_rate" in core_concepts
        assert "cvp" not in core_concepts

        # Load extended features
        extended_concepts = load_timeseries_concepts(concepts_dir, "mimic_iv", "extended")
        assert "cvp" in extended_concepts

    def test_load_concepts_filters_by_dataset(self, tmp_path):
        """Loading concepts should filter by dataset availability."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        vitals_data = {
            "mimic_only_feature": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220045],
                        "time_col": "charttime",
                    }
                ],
            },
            "eicu_only_feature": {
                "feature_set": ["core"],
                "eicu": [
                    {
                        "type": "column",
                        "table": "vitalperiodic",
                        "value_col": "heartrate",
                        "time_col": "observationoffset",
                    }
                ],
            },
        }

        with open(concepts_dir / "vitals.yaml", "w") as f:
            yaml.dump(vitals_data, f)

        mimic_concepts = load_timeseries_concepts(concepts_dir, "mimic_iv", "core")
        assert "mimic_only_feature" in mimic_concepts
        assert "eicu_only_feature" not in mimic_concepts

    def test_load_concepts_raises_on_duplicate(self, tmp_path):
        """Loading concepts should raise on duplicate concept names."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        # Create two files with same concept name
        vitals_data = {
            "duplicate_name": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220045],
                        "time_col": "charttime",
                    }
                ],
            },
        }

        labs_data = {
            "duplicate_name": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "hosp/labevents",
                        "itemid": [50912],
                        "time_col": "charttime",
                    }
                ],
            },
        }

        with open(concepts_dir / "vitals.yaml", "w") as f:
            yaml.dump(vitals_data, f)
        with open(concepts_dir / "labs.yaml", "w") as f:
            yaml.dump(labs_data, f)

        with pytest.raises(ValueError, match="Duplicate concept"):
            load_timeseries_concepts(concepts_dir, "mimic_iv", "core")

    def test_load_concepts_filters_by_categories(self, tmp_path):
        """Loading concepts should filter by category if specified."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        vitals_data = {
            "heart_rate": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220045],
                        "time_col": "charttime",
                    }
                ],
            },
        }

        labs_data = {
            "glucose": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "hosp/labevents",
                        "itemid": [50931],
                        "time_col": "charttime",
                    }
                ],
            },
        }

        with open(concepts_dir / "vitals.yaml", "w") as f:
            yaml.dump(vitals_data, f)
        with open(concepts_dir / "labs.yaml", "w") as f:
            yaml.dump(labs_data, f)

        # Load only vitals
        vitals_only = load_timeseries_concepts(
            concepts_dir, "mimic_iv", "core", categories=["vitals"]
        )
        assert "heart_rate" in vitals_only
        assert "glucose" not in vitals_only


class TestLoadStaticConcepts:
    """Tests for loading static/demographic concept configurations."""

    def test_load_static_concepts(self, tmp_path):
        """Loading static concepts should return correct configs."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        static_data = {
            "age": {
                "description": "Patient age at admission",
                "dtype": "numeric",
                "units": "years",
                "min": 0,
                "max": 120,
                "mimic_iv": {
                    "table": "hosp/patients",
                    "column": "anchor_age",
                },
            },
            "gender": {
                "description": "Patient gender",
                "dtype": "categorical",
                "categories": ["M", "F"],
                "mimic_iv": {
                    "table": "hosp/patients",
                    "column": "gender",
                },
            },
        }

        with open(concepts_dir / "static.yaml", "w") as f:
            yaml.dump(static_data, f)

        result = load_static_concepts(concepts_dir, "mimic_iv")

        assert "age" in result
        assert "gender" in result
        assert result["age"].units == "years"
        assert result["gender"].categories == ["M", "F"]

    def test_load_static_concepts_filters_by_dataset(self, tmp_path):
        """Loading static concepts should filter by dataset."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        static_data = {
            "mimic_feature": {
                "mimic_iv": {"table": "hosp/patients", "column": "anchor_age"},
            },
            "eicu_feature": {
                "eicu": {"table": "patient", "column": "age"},
            },
        }

        with open(concepts_dir / "static.yaml", "w") as f:
            yaml.dump(static_data, f)

        mimic_concepts = load_static_concepts(concepts_dir, "mimic_iv")
        assert "mimic_feature" in mimic_concepts
        assert "eicu_feature" not in mimic_concepts

    def test_load_static_concepts_empty_if_no_file(self, tmp_path):
        """Loading static concepts should return empty dict if no file."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        result = load_static_concepts(concepts_dir, "mimic_iv")
        assert result == {}


class TestGetFeatureNames:
    """Tests for feature name retrieval functions."""

    def test_get_feature_names(self, tmp_path):
        """get_feature_names should return ordered list of feature names."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        vitals_data = {
            "heart_rate": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220045],
                        "time_col": "charttime",
                    }
                ],
            },
            "spo2": {
                "feature_set": ["core"],
                "mimic_iv": [
                    {
                        "type": "itemid",
                        "table": "icu/chartevents",
                        "itemid": [220277],
                        "time_col": "charttime",
                    }
                ],
            },
        }

        with open(concepts_dir / "vitals.yaml", "w") as f:
            yaml.dump(vitals_data, f)

        names = get_feature_names(concepts_dir, "mimic_iv", "core")

        assert isinstance(names, list)
        assert "heart_rate" in names
        assert "spo2" in names

    def test_get_static_feature_names(self, tmp_path):
        """get_static_feature_names should return ordered list of static names."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        static_data = {
            "age": {
                "mimic_iv": {"table": "hosp/patients", "column": "anchor_age"},
            },
            "gender": {
                "mimic_iv": {"table": "hosp/patients", "column": "gender"},
            },
        }

        with open(concepts_dir / "static.yaml", "w") as f:
            yaml.dump(static_data, f)

        names = get_static_feature_names(concepts_dir, "mimic_iv")

        assert isinstance(names, list)
        assert "age" in names
        assert "gender" in names


class TestParseStaticSource:
    """Tests for parsing static extraction sources."""

    def test_parse_static_source_basic(self):
        """Parsing basic static source should return StaticExtractionSource."""
        source_dict = {
            "table": "hosp/patients",
            "column": "anchor_age",
        }
        result = _parse_static_source(source_dict)

        assert result.table == "hosp/patients"
        assert result.column == "anchor_age"
        assert result.itemid is None

    def test_parse_static_source_with_itemid(self):
        """Parsing static source with itemid should include it."""
        source_dict = {
            "table": "icu/chartevents",
            "column": "valuenum",
            "itemid": 226730,  # Height
        }
        result = _parse_static_source(source_dict)

        assert result.itemid == 226730

    def test_parse_static_source_with_transform(self):
        """Parsing static source with transform should include it."""
        source_dict = {
            "table": "patient",
            "column": "age",
            "transform": "eicu_age_parse",
        }
        result = _parse_static_source(source_dict)

        assert result.transform == "eicu_age_parse"


class TestTimeseriesFilesConstant:
    """Tests for TIMESERIES_FILES constant."""

    def test_timeseries_files_order(self):
        """TIMESERIES_FILES should have expected categories in order."""
        assert "vitals.yaml" in TIMESERIES_FILES
        assert "labs.yaml" in TIMESERIES_FILES
        assert "medications.yaml" in TIMESERIES_FILES

        # Order matters for feature ordering
        assert TIMESERIES_FILES.index("vitals.yaml") < TIMESERIES_FILES.index("labs.yaml")
