"""Tests for multi-label phenotyping label builder and related infrastructure.

Comprehensive tests for PhenotypingLabelBuilder, helper functions,
factory integration, and multi-label dataset loading.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from slices.data.labels import LabelBuilderFactory, LabelConfig
from slices.data.labels.phenotyping import (
    PhenotypingLabelBuilder,
    _build_icd_to_phenotype_map,
    _find_phenotype_config,
    _load_phenotype_definitions,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINI_PHENOTYPE_DEFINITIONS = {
    "sepsis": {
        "description": "Septicemia (except in labor)",
        "ccs_group": 2,
        "icd9_codes": ["99591", "99592", "78552"],
        "icd10_codes": ["A4181", "A4189", "R6520", "R6521"],
    },
    "respiratory_failure": {
        "description": "Respiratory failure; insufficiency; arrest (adult)",
        "ccs_group": 131,
        "icd9_codes": ["51881", "51882", "51884"],
        "icd10_codes": ["J9600", "J9601", "J9602"],
    },
    "acute_renal_failure": {
        "description": "Acute and unspecified renal failure",
        "ccs_group": 157,
        "icd9_codes": ["5849", "5845", "5846"],
        "icd10_codes": ["N179", "N170", "N171"],
    },
}

MINI_PHENOTYPE_NAMES = list(MINI_PHENOTYPE_DEFINITIONS.keys())


@pytest.fixture
def phenotype_config_path(tmp_path: Path) -> Path:
    """Create a minimal phenotype config YAML in a temporary directory."""
    config_dir = tmp_path / "configs" / "phenotypes"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "ccs_phenotypes.yaml"
    with open(config_file, "w") as f:
        yaml.dump(MINI_PHENOTYPE_DEFINITIONS, f, default_flow_style=False)
    return config_file


@pytest.fixture
def mini_label_config() -> LabelConfig:
    """LabelConfig for phenotyping task using mini definitions."""
    return LabelConfig(
        task_name="phenotyping",
        task_type="multilabel",
        observation_window_hours=48,
        prediction_window_hours=-1,
        gap_hours=0,
        label_sources=["stays", "diagnoses"],
        label_params={
            "phenotype_config": "phenotypes/ccs_phenotypes.yaml",
            "exclude_multi_stay_admissions": True,
        },
        primary_metric="auroc",
        additional_metrics=["auprc"],
        n_classes=3,
        class_names=MINI_PHENOTYPE_NAMES,
    )


@pytest.fixture
def sample_stays() -> pl.DataFrame:
    """Sample ICU stays data with unique hadm_ids."""
    return pl.DataFrame(
        {
            "stay_id": [100, 200, 300, 400],
            "hadm_id": [1001, 1002, 1003, 1004],
            "intime": [
                datetime(2020, 1, 1, 10, 0),
                datetime(2020, 1, 2, 12, 0),
                datetime(2020, 1, 3, 8, 0),
                datetime(2020, 1, 4, 14, 0),
            ],
            "outtime": [
                datetime(2020, 1, 3, 10, 0),
                datetime(2020, 1, 5, 12, 0),
                datetime(2020, 1, 6, 8, 0),
                datetime(2020, 1, 10, 14, 0),
            ],
        }
    )


@pytest.fixture
def sample_diagnoses() -> pl.DataFrame:
    """Sample diagnoses that map to known phenotypes.

    stay_id=100: sepsis via ICD-9 99591
    stay_id=200: respiratory_failure via ICD-10 J9600
    stay_id=300: acute_renal_failure via ICD-9 5849 AND sepsis via ICD-9 78552
    stay_id=400: no matching ICD codes (unknown code)
    """
    return pl.DataFrame(
        {
            "stay_id": [100, 200, 300, 300, 400],
            "icd_code": ["99591", "J9600", "5849", "78552", "Z9981"],
            "icd_version": [9, 10, 9, 9, 10],
        }
    )


def _make_builder(
    label_config: LabelConfig,
    phenotype_config_path: Path,
) -> PhenotypingLabelBuilder:
    """Create a PhenotypingLabelBuilder with the config path monkey-patched."""
    builder = PhenotypingLabelBuilder(label_config)
    # Override the config path resolution to use our test config
    builder.config.label_params["_resolved_config_path"] = str(phenotype_config_path)
    return builder


def _build_labels_with_test_config(
    builder: PhenotypingLabelBuilder,
    raw_data: Dict[str, pl.DataFrame],
    phenotype_config_path: Path,
) -> pl.DataFrame:
    """Build labels with _find_phenotype_config patched to return test config."""
    with patch(
        "slices.data.labels.phenotyping._find_phenotype_config",
        return_value=phenotype_config_path,
    ):
        return builder.build_labels(raw_data)


# ===========================================================================
# TestPhenotypingConfig
# ===========================================================================


class TestPhenotypingConfig:
    """Tests for LabelConfig creation for phenotyping tasks."""

    def test_phenotyping_config_creation(self):
        """Test basic LabelConfig for phenotyping task."""
        config = LabelConfig(
            task_name="phenotyping",
            task_type="multilabel",
            observation_window_hours=48,
            prediction_window_hours=-1,
            label_sources=["stays", "diagnoses"],
            label_params={
                "phenotype_config": "phenotypes/ccs_phenotypes.yaml",
                "exclude_multi_stay_admissions": True,
            },
            n_classes=10,
            class_names=[
                "sepsis",
                "respiratory_failure",
                "acute_renal_failure",
                "chf",
                "shock",
                "chronic_kidney_disease",
                "diabetes",
                "copd",
                "pneumonia",
                "coronary_atherosclerosis",
            ],
        )

        assert config.task_name == "phenotyping"
        assert config.task_type == "multilabel"
        assert config.n_classes == 10
        assert len(config.class_names) == 10
        assert "stays" in config.label_sources
        assert "diagnoses" in config.label_sources
        assert config.label_params["exclude_multi_stay_admissions"] is True

    def test_phenotyping_config_defaults(self):
        """Test phenotyping LabelConfig with minimal required fields."""
        config = LabelConfig(
            task_name="phenotyping",
            task_type="multilabel",
            label_sources=["stays", "diagnoses"],
        )

        assert config.gap_hours == 0
        assert config.primary_metric == "auroc"
        assert config.label_params == {}
        assert config.n_classes is None

    def test_phenotyping_config_with_all_fields(self):
        """Test phenotyping LabelConfig with all fields populated."""
        config = LabelConfig(
            task_name="phenotyping",
            task_type="multilabel",
            observation_window_hours=48,
            prediction_window_hours=-1,
            gap_hours=0,
            label_sources=["stays", "diagnoses"],
            label_params={
                "phenotype_config": "phenotypes/ccs_phenotypes.yaml",
                "exclude_multi_stay_admissions": True,
            },
            primary_metric="auroc",
            additional_metrics=["auprc"],
            n_classes=3,
            class_names=MINI_PHENOTYPE_NAMES,
        )

        assert config.observation_window_hours == 48
        assert config.prediction_window_hours == -1
        assert config.additional_metrics == ["auprc"]
        assert config.n_classes == 3
        assert config.class_names == MINI_PHENOTYPE_NAMES


# ===========================================================================
# TestPhenotypingLabelBuilder
# ===========================================================================


class TestPhenotypingLabelBuilder:
    """Tests for PhenotypingLabelBuilder.build_labels()."""

    def test_basic_phenotyping(
        self,
        mini_label_config,
        phenotype_config_path,
        sample_stays,
        sample_diagnoses,
    ):
        """Create synthetic stays and diagnoses, verify correct multi-label output."""
        builder = PhenotypingLabelBuilder(mini_label_config)
        raw_data = {"stays": sample_stays, "diagnoses": sample_diagnoses}

        labels = _build_labels_with_test_config(builder, raw_data, phenotype_config_path)

        # Should have one row per stay
        assert len(labels) == 4
        assert "stay_id" in labels.columns

        # Should have one column per phenotype, prefixed with 'phenotyping_'
        for name in MINI_PHENOTYPE_NAMES:
            assert f"phenotyping_{name}" in labels.columns

        # stay_id=100 has sepsis only (ICD-9 99591)
        row_100 = labels.filter(pl.col("stay_id") == 100)
        assert row_100["phenotyping_sepsis"][0] == 1
        assert row_100["phenotyping_respiratory_failure"][0] == 0
        assert row_100["phenotyping_acute_renal_failure"][0] == 0

        # stay_id=200 has respiratory_failure only (ICD-10 J9600)
        row_200 = labels.filter(pl.col("stay_id") == 200)
        assert row_200["phenotyping_sepsis"][0] == 0
        assert row_200["phenotyping_respiratory_failure"][0] == 1
        assert row_200["phenotyping_acute_renal_failure"][0] == 0

        # stay_id=300 has both sepsis (ICD-9 78552) and acute_renal_failure (ICD-9 5849)
        row_300 = labels.filter(pl.col("stay_id") == 300)
        assert row_300["phenotyping_sepsis"][0] == 1
        assert row_300["phenotyping_respiratory_failure"][0] == 0
        assert row_300["phenotyping_acute_renal_failure"][0] == 1

        # stay_id=400 has no matching codes
        row_400 = labels.filter(pl.col("stay_id") == 400)
        assert row_400["phenotyping_sepsis"][0] == 0
        assert row_400["phenotyping_respiratory_failure"][0] == 0
        assert row_400["phenotyping_acute_renal_failure"][0] == 0

    def test_no_diagnoses_returns_all_zeros(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Stay with no diagnoses at all gets all-zero vector."""
        stays = pl.DataFrame(
            {
                "stay_id": [100],
                "hadm_id": [1001],
                "intime": [datetime(2020, 1, 1, 10, 0)],
                "outtime": [datetime(2020, 1, 3, 10, 0)],
            }
        )
        diagnoses = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "icd_code": pl.Series([], dtype=pl.Utf8),
                "icd_version": pl.Series([], dtype=pl.Int64),
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        assert len(labels) == 1
        for name in MINI_PHENOTYPE_NAMES:
            assert labels[f"phenotyping_{name}"][0] == 0

    def test_all_phenotypes_present(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Stay with ICD codes matching all phenotypes gets all-ones vector."""
        stays = pl.DataFrame(
            {
                "stay_id": [100],
                "hadm_id": [1001],
                "intime": [datetime(2020, 1, 1, 10, 0)],
                "outtime": [datetime(2020, 1, 3, 10, 0)],
            }
        )
        # One ICD code per phenotype
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100, 100, 100],
                "icd_code": ["99591", "51881", "5849"],  # sepsis, resp_fail, renal_fail
                "icd_version": [9, 9, 9],
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        assert len(labels) == 1
        for name in MINI_PHENOTYPE_NAMES:
            assert (
                labels[f"phenotyping_{name}"][0] == 1
            ), f"Expected phenotyping_{name} == 1 but got {labels[f'phenotyping_{name}'][0]}"

    def test_multi_stay_exclusion(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """hadm_id with 2 stays gets null labels for both stays."""
        # Two stays sharing the same hadm_id
        stays = pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "hadm_id": [1001, 1001, 1002],  # 100 and 200 share hadm_id
                "intime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 10, 0),
                ],
                "outtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 10, 0),
                    datetime(2020, 1, 7, 10, 0),
                ],
            }
        )
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "icd_code": ["99591", "99591", "99591"],
                "icd_version": [9, 9, 9],
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        assert len(labels) == 3

        # Multi-stay stays (100, 200) should have null labels
        row_100 = labels.filter(pl.col("stay_id") == 100)
        row_200 = labels.filter(pl.col("stay_id") == 200)
        for name in MINI_PHENOTYPE_NAMES:
            assert row_100[f"phenotyping_{name}"][0] is None
            assert row_200[f"phenotyping_{name}"][0] is None

        # Single-stay (300) should have valid labels
        row_300 = labels.filter(pl.col("stay_id") == 300)
        assert row_300["phenotyping_sepsis"][0] == 1

    def test_multi_stay_exclusion_disabled(
        self,
        phenotype_config_path,
    ):
        """When exclude_multi_stay_admissions=false, no exclusion happens."""
        config = LabelConfig(
            task_name="phenotyping",
            task_type="multilabel",
            label_sources=["stays", "diagnoses"],
            label_params={
                "phenotype_config": "phenotypes/ccs_phenotypes.yaml",
                "exclude_multi_stay_admissions": False,
            },
            n_classes=3,
            class_names=MINI_PHENOTYPE_NAMES,
        )

        # Two stays sharing the same hadm_id
        stays = pl.DataFrame(
            {
                "stay_id": [100, 200],
                "hadm_id": [1001, 1001],
                "intime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 3, 10, 0),
                ],
                "outtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 10, 0),
                ],
            }
        )
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100, 200],
                "icd_code": ["99591", "J9600"],
                "icd_version": [9, 10],
            }
        )

        builder = PhenotypingLabelBuilder(config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        assert len(labels) == 2

        # Neither should be null -- exclusion is disabled
        row_100 = labels.filter(pl.col("stay_id") == 100)
        row_200 = labels.filter(pl.col("stay_id") == 200)
        assert row_100["phenotyping_sepsis"][0] == 1
        assert row_200["phenotyping_respiratory_failure"][0] == 1

    def test_icd9_codes_mapping(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Verify multiple ICD-9 codes correctly map to phenotypes."""
        stays = pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "hadm_id": [1001, 1002, 1003],
                "intime": [datetime(2020, 1, 1)] * 3,
                "outtime": [datetime(2020, 1, 3)] * 3,
            }
        )
        # Use different ICD-9 codes for each phenotype
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "icd_code": [
                    "99592",  # sepsis (alternate ICD-9)
                    "51882",  # respiratory_failure (alternate ICD-9)
                    "5845",  # acute_renal_failure (alternate ICD-9)
                ],
                "icd_version": [9, 9, 9],
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        row_100 = labels.filter(pl.col("stay_id") == 100)
        assert row_100["phenotyping_sepsis"][0] == 1

        row_200 = labels.filter(pl.col("stay_id") == 200)
        assert row_200["phenotyping_respiratory_failure"][0] == 1

        row_300 = labels.filter(pl.col("stay_id") == 300)
        assert row_300["phenotyping_acute_renal_failure"][0] == 1

    def test_icd10_codes_mapping(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Verify multiple ICD-10 codes correctly map to phenotypes."""
        stays = pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "hadm_id": [1001, 1002, 1003],
                "intime": [datetime(2020, 1, 1)] * 3,
                "outtime": [datetime(2020, 1, 3)] * 3,
            }
        )
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "icd_code": [
                    "R6521",  # sepsis ICD-10
                    "J9602",  # respiratory_failure ICD-10
                    "N171",  # acute_renal_failure ICD-10
                ],
                "icd_version": [10, 10, 10],
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        row_100 = labels.filter(pl.col("stay_id") == 100)
        assert row_100["phenotyping_sepsis"][0] == 1

        row_200 = labels.filter(pl.col("stay_id") == 200)
        assert row_200["phenotyping_respiratory_failure"][0] == 1

        row_300 = labels.filter(pl.col("stay_id") == 300)
        assert row_300["phenotyping_acute_renal_failure"][0] == 1

    def test_empty_stays(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Empty stays DataFrame returns empty DataFrame with correct schema."""
        stays = pl.DataFrame(
            schema={
                "stay_id": pl.Int64,
                "hadm_id": pl.Int64,
                "intime": pl.Datetime,
                "outtime": pl.Datetime,
            }
        )
        diagnoses = pl.DataFrame(
            schema={
                "stay_id": pl.Int64,
                "icd_code": pl.Utf8,
                "icd_version": pl.Int64,
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        assert len(labels) == 0
        assert "stay_id" in labels.columns
        # The _empty_result method uses class_names from config
        for name in MINI_PHENOTYPE_NAMES:
            assert f"phenotyping_{name}" in labels.columns

    def test_mixed_icd_versions(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Mix of ICD-9 and ICD-10 codes for the same stay."""
        stays = pl.DataFrame(
            {
                "stay_id": [100],
                "hadm_id": [1001],
                "intime": [datetime(2020, 1, 1, 10, 0)],
                "outtime": [datetime(2020, 1, 3, 10, 0)],
            }
        )
        # ICD-9 sepsis code + ICD-10 respiratory_failure code
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100, 100],
                "icd_code": ["99591", "J9601"],
                "icd_version": [9, 10],
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        row = labels.filter(pl.col("stay_id") == 100)
        assert row["phenotyping_sepsis"][0] == 1
        assert row["phenotyping_respiratory_failure"][0] == 1
        assert row["phenotyping_acute_renal_failure"][0] == 0

    def test_unknown_icd_version_ignored(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """ICD codes with unexpected version numbers are silently ignored."""
        stays = pl.DataFrame(
            {
                "stay_id": [100],
                "hadm_id": [1001],
                "intime": [datetime(2020, 1, 1)],
                "outtime": [datetime(2020, 1, 3)],
            }
        )
        diagnoses = pl.DataFrame(
            {
                "stay_id": [100],
                "icd_code": ["99591"],
                "icd_version": [11],  # Unknown ICD version
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        # Should be all zeros since version 11 is ignored
        for name in MINI_PHENOTYPE_NAMES:
            assert labels[f"phenotyping_{name}"][0] == 0

    def test_column_types_are_int32(
        self,
        mini_label_config,
        phenotype_config_path,
        sample_stays,
        sample_diagnoses,
    ):
        """Phenotype columns should be cast to Int32."""
        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder,
            {"stays": sample_stays, "diagnoses": sample_diagnoses},
            phenotype_config_path,
        )

        for name in MINI_PHENOTYPE_NAMES:
            col_name = f"phenotyping_{name}"
            assert (
                labels[col_name].dtype == pl.Int32
            ), f"Column {col_name} has dtype {labels[col_name].dtype}, expected Int32"

    def test_stay_id_column_is_int64(
        self,
        mini_label_config,
        phenotype_config_path,
        sample_stays,
        sample_diagnoses,
    ):
        """stay_id column should be cast to Int64."""
        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder,
            {"stays": sample_stays, "diagnoses": sample_diagnoses},
            phenotype_config_path,
        )

        assert labels["stay_id"].dtype == pl.Int64

    def test_validate_inputs_missing_source(self, mini_label_config):
        """Builder should raise ValueError if required data sources are missing."""
        builder = PhenotypingLabelBuilder(mini_label_config)

        with pytest.raises(ValueError, match="missing"):
            builder.build_labels({"stays": pl.DataFrame({"stay_id": [1]})})

    def test_diagnoses_for_unknown_stay_ignored(
        self,
        mini_label_config,
        phenotype_config_path,
    ):
        """Diagnoses referencing stay_ids not in stays are silently ignored."""
        stays = pl.DataFrame(
            {
                "stay_id": [100],
                "hadm_id": [1001],
                "intime": [datetime(2020, 1, 1)],
                "outtime": [datetime(2020, 1, 3)],
            }
        )
        # Diagnosis for stay_id=999 which is not in stays
        diagnoses = pl.DataFrame(
            {
                "stay_id": [999],
                "icd_code": ["99591"],
                "icd_version": [9],
            }
        )

        builder = PhenotypingLabelBuilder(mini_label_config)
        labels = _build_labels_with_test_config(
            builder, {"stays": stays, "diagnoses": diagnoses}, phenotype_config_path
        )

        assert len(labels) == 1
        # stay_id=100 should have all zeros since its diagnoses don't exist
        for name in MINI_PHENOTYPE_NAMES:
            assert labels[f"phenotyping_{name}"][0] == 0


# ===========================================================================
# TestPhenotypingHelpers
# ===========================================================================


class TestPhenotypingHelpers:
    """Tests for phenotyping helper functions."""

    def test_load_phenotype_definitions(self, phenotype_config_path):
        """Load YAML and verify structure."""
        definitions = _load_phenotype_definitions(phenotype_config_path)

        assert isinstance(definitions, dict)
        assert "sepsis" in definitions
        assert "respiratory_failure" in definitions
        assert "acute_renal_failure" in definitions

        # Each phenotype should have icd9_codes and icd10_codes
        for name, defn in definitions.items():
            assert "icd9_codes" in defn, f"Missing icd9_codes for {name}"
            assert "icd10_codes" in defn, f"Missing icd10_codes for {name}"
            assert isinstance(defn["icd9_codes"], list)
            assert isinstance(defn["icd10_codes"], list)

    def test_load_phenotype_definitions_file_not_found(self, tmp_path):
        """Raise FileNotFoundError when config file doesn't exist."""
        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Phenotype config not found"):
            _load_phenotype_definitions(missing_path)

    def test_build_icd_to_phenotype_map(self):
        """Verify mapping correctness for both ICD-9 and ICD-10."""
        icd9_map, icd10_map = _build_icd_to_phenotype_map(MINI_PHENOTYPE_DEFINITIONS)

        # ICD-9 checks
        assert "99591" in icd9_map
        assert "sepsis" in icd9_map["99591"]

        assert "51881" in icd9_map
        assert "respiratory_failure" in icd9_map["51881"]

        assert "5849" in icd9_map
        assert "acute_renal_failure" in icd9_map["5849"]

        # ICD-10 checks
        assert "A4181" in icd10_map
        assert "sepsis" in icd10_map["A4181"]

        assert "J9600" in icd10_map
        assert "respiratory_failure" in icd10_map["J9600"]

        assert "N179" in icd10_map
        assert "acute_renal_failure" in icd10_map["N179"]

    def test_build_icd_to_phenotype_map_code_shared_across_phenotypes(self):
        """Verify that an ICD code can map to multiple phenotypes."""
        definitions = {
            "condition_a": {"icd9_codes": ["12345"], "icd10_codes": []},
            "condition_b": {"icd9_codes": ["12345"], "icd10_codes": []},
        }

        icd9_map, _ = _build_icd_to_phenotype_map(definitions)

        assert "12345" in icd9_map
        assert len(icd9_map["12345"]) == 2
        assert "condition_a" in icd9_map["12345"]
        assert "condition_b" in icd9_map["12345"]

    def test_build_icd_to_phenotype_map_empty_definitions(self):
        """Empty definitions produce empty maps."""
        icd9_map, icd10_map = _build_icd_to_phenotype_map({})
        assert icd9_map == {}
        assert icd10_map == {}

    def test_build_icd_to_phenotype_map_numeric_codes_converted_to_string(self):
        """ICD codes that are numeric in YAML are converted to strings."""
        definitions = {
            "test_pheno": {
                "icd9_codes": [1234, 5678],  # numeric
                "icd10_codes": ["A001"],
            }
        }

        icd9_map, icd10_map = _build_icd_to_phenotype_map(definitions)

        # Codes should be stored as strings
        assert "1234" in icd9_map
        assert "5678" in icd9_map
        assert "A001" in icd10_map

    def test_find_phenotype_config_not_found(self, tmp_path, monkeypatch):
        """Verify FileNotFoundError when config cannot be found anywhere."""
        # Change cwd to a directory without pyproject.toml so Strategy 1 fails.
        monkeypatch.chdir(tmp_path)

        # Strategy 2 uses Path(__file__).parents[4] which resolves to
        # somewhere in the installed package tree. We need that path to also
        # not contain the config. Patching __file__ in the module ensures
        # the fallback path points to our empty tmp_path.
        fake_file = tmp_path / "a" / "b" / "c" / "d" / "fake.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()

        with patch(
            "slices.data.labels.phenotyping.__file__",
            str(fake_file),
        ):
            with pytest.raises(FileNotFoundError, match="Could not locate phenotype config"):
                _find_phenotype_config({"phenotype_config": "nonexistent/path.yaml"})


# ===========================================================================
# TestPhenotypingFactory
# ===========================================================================


class TestPhenotypingFactory:
    """Tests for factory integration with PhenotypingLabelBuilder."""

    def test_factory_creates_phenotyping_builder(self):
        """Verify factory creates PhenotypingLabelBuilder for phenotyping tasks."""
        config = LabelConfig(
            task_name="phenotyping",
            task_type="multilabel",
            label_sources=["stays", "diagnoses"],
        )

        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, PhenotypingLabelBuilder)

    def test_factory_creates_phenotyping_builder_with_suffix(self):
        """Factory extracts 'phenotyping' from task names with underscores."""
        config = LabelConfig(
            task_name="phenotyping_ccs10",
            task_type="multilabel",
            label_sources=["stays", "diagnoses"],
        )

        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, PhenotypingLabelBuilder)

    def test_factory_registry(self):
        """Verify 'phenotyping' is registered in the factory."""
        available = LabelBuilderFactory.list_available()

        assert "phenotyping" in available
        assert available["phenotyping"] is PhenotypingLabelBuilder

    def test_factory_phenotyping_and_mortality_coexist(self):
        """Both phenotyping and mortality builders are registered."""
        available = LabelBuilderFactory.list_available()

        assert "phenotyping" in available
        assert "mortality" in available


# ===========================================================================
# TestMultilabelDatasetLoading
# ===========================================================================


class TestMultilabelDatasetLoading:
    """Tests for multi-label detection and tensor loading in ICUDataset.

    These tests verify the multi-label detection logic in
    _precompute_labels_and_static() without constructing a full ICUDataset
    (which requires heavy I/O and preprocessing). Instead, we test the
    detection logic and tensor shapes directly.
    """

    def _make_labels_df(self) -> pl.DataFrame:
        """Create a labels DataFrame with multi-label phenotyping columns."""
        return pl.DataFrame(
            {
                "stay_id": [100, 200, 300],
                "phenotyping_sepsis": [1, 0, 1],
                "phenotyping_respiratory_failure": [0, 1, 0],
                "phenotyping_acute_renal_failure": [0, 0, 1],
            }
        )

    def test_multilabel_detection(self):
        """Verify multi-label columns are detected when task_name column is absent."""
        labels_df = self._make_labels_df()
        task_name = "phenotyping"

        # The detection logic from ICUDataset._precompute_labels_and_static
        multilabel_cols = [col for col in labels_df.columns if col.startswith(f"{task_name}_")]
        is_multilabel = len(multilabel_cols) > 0 and task_name not in labels_df.columns

        assert is_multilabel is True
        assert len(multilabel_cols) == 3
        assert "phenotyping_sepsis" in multilabel_cols
        assert "phenotyping_respiratory_failure" in multilabel_cols
        assert "phenotyping_acute_renal_failure" in multilabel_cols

    def test_multilabel_detection_false_for_scalar_task(self):
        """Scalar task (e.g. mortality) should not be detected as multi-label."""
        labels_df = pl.DataFrame(
            {
                "stay_id": [100, 200],
                "mortality_24h": [1, 0],
            }
        )
        task_name = "mortality_24h"

        multilabel_cols = [col for col in labels_df.columns if col.startswith(f"{task_name}_")]
        is_multilabel = len(multilabel_cols) > 0 and task_name not in labels_df.columns

        assert is_multilabel is False

    def test_multilabel_tensor_shape(self):
        """Verify (N, n_classes) tensor shape for phenotyping."""
        labels_df = self._make_labels_df()
        task_name = "phenotyping"

        # Simulate the label extraction logic from ICUDataset
        multilabel_cols = [col for col in labels_df.columns if col.startswith(f"{task_name}_")]

        labels_by_stay = {row["stay_id"]: row for row in labels_df.to_dicts()}
        stay_ids = [100, 200, 300]

        label_values = []
        for stay_id in stay_ids:
            label_row = labels_by_stay[stay_id]
            vals = [label_row[col] for col in multilabel_cols]
            label_values.append(vals)

        labels_tensor = torch.tensor(label_values, dtype=torch.float32)

        # Should be (N=3, n_classes=3)
        assert labels_tensor.shape == (3, 3)

        # Verify specific values
        # stay_id=100: sepsis=1, resp_fail=0, renal_fail=0
        assert labels_tensor[0].tolist() == [1.0, 0.0, 0.0]
        # stay_id=200: sepsis=0, resp_fail=1, renal_fail=0
        assert labels_tensor[1].tolist() == [0.0, 1.0, 0.0]
        # stay_id=300: sepsis=1, resp_fail=0, renal_fail=1
        assert labels_tensor[2].tolist() == [1.0, 0.0, 1.0]

    def test_multilabel_tensor_with_10_phenotypes(self):
        """Verify (N, 10) tensor shape for full phenotyping config."""
        phenotype_names = [
            "sepsis",
            "respiratory_failure",
            "acute_renal_failure",
            "chf",
            "shock",
            "chronic_kidney_disease",
            "diabetes",
            "copd",
            "pneumonia",
            "coronary_atherosclerosis",
        ]

        # Create labels DataFrame with 10 phenotype columns
        data = {"stay_id": [100, 200]}
        for name in phenotype_names:
            data[f"phenotyping_{name}"] = [0, 0]
        # Set a few positives
        data["phenotyping_sepsis"] = [1, 0]
        data["phenotyping_diabetes"] = [0, 1]

        labels_df = pl.DataFrame(data)

        task_name = "phenotyping"
        multilabel_cols = [col for col in labels_df.columns if col.startswith(f"{task_name}_")]

        labels_by_stay = {row["stay_id"]: row for row in labels_df.to_dicts()}
        label_values = []
        for stay_id in [100, 200]:
            label_row = labels_by_stay[stay_id]
            vals = [label_row[col] for col in multilabel_cols]
            label_values.append(vals)

        labels_tensor = torch.tensor(label_values, dtype=torch.float32)

        # Should be (N=2, n_classes=10)
        assert labels_tensor.shape == (2, 10)

    def test_multilabel_null_labels_detected_as_missing(self):
        """Null values in multi-label columns are detected as missing."""
        labels_df = pl.DataFrame(
            {
                "stay_id": [100, 200],
                "phenotyping_sepsis": [1, None],
                "phenotyping_respiratory_failure": [0, None],
                "phenotyping_acute_renal_failure": [0, None],
            }
        )

        task_name = "phenotyping"
        multilabel_cols = [col for col in labels_df.columns if col.startswith(f"{task_name}_")]

        labels_by_stay = {row["stay_id"]: row for row in labels_df.to_dicts()}

        # Simulate missing label detection logic from ICUDataset
        missing_label_stays = []
        valid_label_values = []

        for stay_id in [100, 200]:
            label_row = labels_by_stay[stay_id]
            vals = [label_row.get(col) for col in multilabel_cols]
            if any(v is None for v in vals):
                missing_label_stays.append(stay_id)
            else:
                valid_label_values.append(vals)

        assert 200 in missing_label_stays
        assert 100 not in missing_label_stays
        assert len(valid_label_values) == 1


# ===========================================================================
# TestPhenotypingIntegration
# ===========================================================================


class TestPhenotypingIntegration:
    """Integration tests verifying ICUDataset loads multi-label phenotyping labels."""

    def test_dataset_loads_multilabel_phenotyping(self, tmp_path: Path):
        """Create mock parquet files and load via ICUDataset with phenotyping task."""
        from slices.data.dataset import ICUDataset

        data_dir = tmp_path / "integration_data"
        data_dir.mkdir()

        n_stays = 5
        seq_length = 48
        n_features = 3
        n_phenotypes = 10
        stay_ids = list(range(1, n_stays + 1))

        phenotype_names = [
            "sepsis",
            "respiratory_failure",
            "acute_renal_failure",
            "chf",
            "shock",
            "chronic_kidney_disease",
            "diabetes",
            "copd",
            "pneumonia",
            "coronary_atherosclerosis",
        ]

        # -- timeseries.parquet: nested list format --
        np.random.seed(0)
        timeseries_data = []
        mask_data = []
        for _ in stay_ids:
            ts = np.random.randn(seq_length, n_features).tolist()
            mask = np.ones((seq_length, n_features), dtype=bool).tolist()
            timeseries_data.append(ts)
            mask_data.append(mask)

        pl.DataFrame(
            {"stay_id": stay_ids, "timeseries": timeseries_data, "mask": mask_data}
        ).write_parquet(data_dir / "timeseries.parquet")

        # -- labels.parquet: 10 phenotyping_* columns --
        labels_dict: Dict = {"stay_id": stay_ids}
        np.random.seed(1)
        for name in phenotype_names:
            labels_dict[f"phenotyping_{name}"] = np.random.randint(0, 2, size=n_stays).tolist()
        pl.DataFrame(labels_dict).write_parquet(data_dir / "labels.parquet")

        # -- static.parquet --
        pl.DataFrame(
            {
                "stay_id": stay_ids,
                "age": [65.0, 70.0, 55.0, 80.0, 45.0],
                "gender": ["M", "F", "M", "F", "M"],
                "los_days": [3.0, 4.0, 2.0, 5.0, 3.5],
            }
        ).write_parquet(data_dir / "static.parquet")

        # -- metadata.yaml --
        metadata = {
            "feature_names": [f"feat_{i}" for i in range(n_features)],
            "seq_length_hours": seq_length,
            "task_names": ["phenotyping"],
        }
        with open(data_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Load dataset
        dataset = ICUDataset(data_dir, task_name="phenotyping", normalize=False)

        assert len(dataset) == n_stays

        sample = dataset[0]
        assert "label" in sample
        assert sample["label"].shape == (n_phenotypes,)
        assert sample["label"].dtype == torch.float32
