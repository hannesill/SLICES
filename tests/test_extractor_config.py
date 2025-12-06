"""Tests for ExtractorConfig dataclass and validation."""

import pytest

from slices.data.extractors.base import ExtractorConfig


class TestExtractorConfigBasic:
    """Test basic ExtractorConfig instantiation."""

    def test_valid_minimal_config(self):
        """Test creation with only required parquet_root."""
        config = ExtractorConfig(parquet_root="/path/to/parquet")

        assert config.parquet_root == "/path/to/parquet"

    def test_missing_parquet_root_raises_error(self):
        """Test that missing parquet_root raises TypeError."""
        with pytest.raises(TypeError):
            ExtractorConfig()

    def test_default_values(self):
        """Test default configuration values."""
        config = ExtractorConfig(parquet_root="/path/to/parquet")

        assert config.output_dir == "data/processed"
        assert config.seq_length_hours == 48
        assert config.feature_set == "core"
        assert config.concepts_dir is None
        assert config.tasks_dir is None
        assert config.min_stay_hours == 6
        assert "mortality_24h" in config.tasks

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExtractorConfig(
            parquet_root="/custom/path",
            output_dir="/custom/output",
            seq_length_hours=72,
            feature_set="extended",
            concepts_dir="/custom/concepts",
            tasks_dir="/custom/tasks",
            tasks=["mortality_24h"],
            min_stay_hours=12,
        )

        assert config.parquet_root == "/custom/path"
        assert config.output_dir == "/custom/output"
        assert config.seq_length_hours == 72
        assert config.feature_set == "extended"
        assert config.concepts_dir == "/custom/concepts"
        assert config.tasks_dir == "/custom/tasks"
        assert config.tasks == ["mortality_24h"]
        assert config.min_stay_hours == 12


class TestExtractorConfigParquetRootValidation:
    """Test parquet_root validation."""

    def test_empty_string_raises_error(self):
        """Test that empty string parquet_root raises ValueError."""
        with pytest.raises(ValueError, match="parquet_root cannot be empty"):
            ExtractorConfig(parquet_root="")

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only parquet_root raises ValueError."""
        with pytest.raises(ValueError, match="parquet_root cannot be empty"):
            ExtractorConfig(parquet_root="   ")

    def test_whitespace_only_tabs_raises_error(self):
        """Test that tabs-only parquet_root raises ValueError."""
        with pytest.raises(ValueError, match="parquet_root cannot be empty"):
            ExtractorConfig(parquet_root="\t\t")

    def test_valid_path_with_spaces_accepted(self):
        """Test that valid paths containing spaces are accepted."""
        config = ExtractorConfig(parquet_root="/path/with spaces/data")
        assert config.parquet_root == "/path/with spaces/data"

    def test_relative_path_accepted(self):
        """Test that relative paths are accepted."""
        config = ExtractorConfig(parquet_root="data/mimic-iv")
        assert config.parquet_root == "data/mimic-iv"


class TestExtractorConfigSeqLengthValidation:
    """Test seq_length_hours validation."""

    def test_zero_raises_error(self):
        """Test that zero seq_length_hours raises ValueError."""
        with pytest.raises(ValueError, match="seq_length_hours must be positive"):
            ExtractorConfig(parquet_root="/path", seq_length_hours=0)

    def test_negative_raises_error(self):
        """Test that negative seq_length_hours raises ValueError."""
        with pytest.raises(ValueError, match="seq_length_hours must be positive"):
            ExtractorConfig(parquet_root="/path", seq_length_hours=-24)

    def test_positive_value_accepted(self):
        """Test that positive seq_length_hours is accepted."""
        config = ExtractorConfig(parquet_root="/path", seq_length_hours=24)
        assert config.seq_length_hours == 24

    def test_large_value_accepted(self):
        """Test that large seq_length_hours values are accepted."""
        config = ExtractorConfig(parquet_root="/path", seq_length_hours=168)  # 1 week
        assert config.seq_length_hours == 168


class TestExtractorConfigMinStayValidation:
    """Test min_stay_hours validation."""

    def test_negative_raises_error(self):
        """Test that negative min_stay_hours raises ValueError."""
        with pytest.raises(ValueError, match="min_stay_hours cannot be negative"):
            ExtractorConfig(parquet_root="/path", min_stay_hours=-1)

    def test_zero_accepted(self):
        """Test that zero min_stay_hours is accepted (include all stays)."""
        config = ExtractorConfig(parquet_root="/path", min_stay_hours=0)
        assert config.min_stay_hours == 0

    def test_positive_value_accepted(self):
        """Test that positive min_stay_hours is accepted."""
        config = ExtractorConfig(parquet_root="/path", min_stay_hours=12)
        assert config.min_stay_hours == 12


class TestExtractorConfigFeatureSetValidation:
    """Test feature_set validation."""

    def test_core_accepted(self):
        """Test that 'core' feature_set is accepted."""
        config = ExtractorConfig(parquet_root="/path", feature_set="core")
        assert config.feature_set == "core"

    def test_extended_accepted(self):
        """Test that 'extended' feature_set is accepted."""
        config = ExtractorConfig(parquet_root="/path", feature_set="extended")
        assert config.feature_set == "extended"

    def test_invalid_feature_set_raises_error(self):
        """Test that invalid feature_set raises ValueError."""
        with pytest.raises(ValueError, match="feature_set must be one of"):
            ExtractorConfig(parquet_root="/path", feature_set="invalid")

    def test_empty_feature_set_raises_error(self):
        """Test that empty feature_set raises ValueError."""
        with pytest.raises(ValueError, match="feature_set must be one of"):
            ExtractorConfig(parquet_root="/path", feature_set="")

    def test_case_sensitive(self):
        """Test that feature_set validation is case-sensitive."""
        with pytest.raises(ValueError, match="feature_set must be one of"):
            ExtractorConfig(parquet_root="/path", feature_set="CORE")


class TestExtractorConfigOutputDirValidation:
    """Test output_dir validation."""

    def test_empty_output_dir_raises_error(self):
        """Test that empty output_dir raises ValueError."""
        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            ExtractorConfig(parquet_root="/path", output_dir="")

    def test_whitespace_only_output_dir_raises_error(self):
        """Test that whitespace-only output_dir raises ValueError."""
        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            ExtractorConfig(parquet_root="/path", output_dir="   ")

    def test_valid_output_dir_accepted(self):
        """Test that valid output_dir paths are accepted."""
        config = ExtractorConfig(parquet_root="/path", output_dir="/output/data")
        assert config.output_dir == "/output/data"


class TestExtractorConfigTasksParameter:
    """Test tasks parameter handling."""

    def test_default_tasks_list(self):
        """Test that default tasks list contains expected mortality tasks."""
        config = ExtractorConfig(parquet_root="/path")
        
        assert "mortality_24h" in config.tasks
        assert "mortality_48h" in config.tasks
        assert "mortality_hospital" in config.tasks

    def test_custom_tasks_list(self):
        """Test custom tasks list."""
        config = ExtractorConfig(parquet_root="/path", tasks=["custom_task"])
        assert config.tasks == ["custom_task"]

    def test_empty_tasks_list_accepted(self):
        """Test that empty tasks list is accepted (extract without labels)."""
        config = ExtractorConfig(parquet_root="/path", tasks=[])
        assert config.tasks == []

    def test_tasks_list_not_shared_between_instances(self):
        """Test that default tasks list is not shared between instances."""
        config1 = ExtractorConfig(parquet_root="/path1")
        config2 = ExtractorConfig(parquet_root="/path2")
        
        config1.tasks.append("new_task")
        
        # config2's tasks should not be affected
        assert "new_task" not in config2.tasks


class TestExtractorConfigOptionalPaths:
    """Test optional path parameters."""

    def test_concepts_dir_none_by_default(self):
        """Test concepts_dir is None by default (auto-detected)."""
        config = ExtractorConfig(parquet_root="/path")
        assert config.concepts_dir is None

    def test_tasks_dir_none_by_default(self):
        """Test tasks_dir is None by default (auto-detected)."""
        config = ExtractorConfig(parquet_root="/path")
        assert config.tasks_dir is None

    def test_custom_concepts_dir(self):
        """Test custom concepts_dir path."""
        config = ExtractorConfig(parquet_root="/path", concepts_dir="/custom/concepts")
        assert config.concepts_dir == "/custom/concepts"

    def test_custom_tasks_dir(self):
        """Test custom tasks_dir path."""
        config = ExtractorConfig(parquet_root="/path", tasks_dir="/custom/tasks")
        assert config.tasks_dir == "/custom/tasks"


class TestExtractorConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_seq_length_equals_one_hour(self):
        """Test minimum practical seq_length_hours of 1."""
        config = ExtractorConfig(parquet_root="/path", seq_length_hours=1)
        assert config.seq_length_hours == 1

    def test_min_stay_hours_equals_seq_length(self):
        """Test min_stay_hours equal to seq_length_hours."""
        config = ExtractorConfig(
            parquet_root="/path",
            seq_length_hours=24,
            min_stay_hours=24,
        )
        assert config.min_stay_hours == config.seq_length_hours

    def test_min_stay_hours_greater_than_seq_length(self):
        """Test min_stay_hours greater than seq_length_hours (valid use case)."""
        config = ExtractorConfig(
            parquet_root="/path",
            seq_length_hours=24,
            min_stay_hours=48,  # Only include longer stays
        )
        assert config.min_stay_hours > config.seq_length_hours

    def test_unicode_paths_accepted(self):
        """Test that unicode characters in paths are accepted."""
        config = ExtractorConfig(parquet_root="/données/患者")
        assert config.parquet_root == "/données/患者"

    def test_windows_style_path_accepted(self):
        """Test that Windows-style paths are accepted."""
        config = ExtractorConfig(parquet_root="C:\\Users\\data\\mimic")
        assert config.parquet_root == "C:\\Users\\data\\mimic"
