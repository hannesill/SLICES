"""Package-level tests for SLICES.

Tests that the package is properly structured, importable, and has correct version.
"""


class TestPackageStructure:
    """Tests for package structure and imports."""

    def test_main_package_importable(self) -> None:
        """Test that main package can be imported."""
        import slices

        assert slices is not None

    def test_version_exists_and_valid(self) -> None:
        """Test that package version is accessible and follows semver."""
        import slices

        assert hasattr(slices, "__version__")
        version = slices.__version__

        # Version should be a string
        assert isinstance(version, str)

        # Should follow semver (x.y.z)
        parts = version.split(".")
        assert len(parts) >= 2, f"Version {version} should have at least major.minor"

        # Major and minor should be numeric
        assert parts[0].isdigit(), f"Major version should be numeric: {parts[0]}"
        assert parts[1].isdigit(), f"Minor version should be numeric: {parts[1]}"

    def test_data_subpackage_importable(self) -> None:
        """Test that data subpackage can be imported."""
        from slices.data import extractors, labels

        assert extractors is not None
        assert labels is not None

    def test_core_classes_importable(self) -> None:
        """Test that core classes can be imported from expected locations."""
        from slices.data.datamodule import ICUDataModule, icu_collate_fn
        from slices.data.dataset import ICUDataset
        from slices.data.extractors.base import BaseExtractor, ExtractorConfig
        from slices.data.extractors.mimic_iv import MIMICIVExtractor
        from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig
        from slices.data.labels.mortality import MortalityLabelBuilder

        # Verify classes exist and are the right type
        assert isinstance(BaseExtractor, type)
        assert isinstance(ExtractorConfig, type)
        assert isinstance(MIMICIVExtractor, type)
        assert isinstance(ICUDataset, type)
        assert isinstance(ICUDataModule, type)
        assert callable(icu_collate_fn)
        assert isinstance(LabelConfig, type)
        assert isinstance(LabelBuilder, type)
        assert isinstance(LabelBuilderFactory, type)
        assert isinstance(MortalityLabelBuilder, type)

    def test_data_io_importable(self) -> None:
        """Test that data_io module imports successfully."""
        from slices.data.data_io import convert_csv_to_parquet

        assert callable(convert_csv_to_parquet)

    def test_mimic_extractor_inherits_base(self) -> None:
        """Test that MIMICIVExtractor properly inherits from BaseExtractor."""
        from slices.data.extractors.base import BaseExtractor
        from slices.data.extractors.mimic_iv import MIMICIVExtractor

        assert issubclass(MIMICIVExtractor, BaseExtractor)

    def test_mortality_builder_inherits_base(self) -> None:
        """Test that MortalityLabelBuilder properly inherits from LabelBuilder."""
        from slices.data.labels.base import LabelBuilder
        from slices.data.labels.mortality import MortalityLabelBuilder

        assert issubclass(MortalityLabelBuilder, LabelBuilder)

    def test_datamodule_inherits_lightning(self) -> None:
        """Test that ICUDataModule inherits from LightningDataModule."""
        import lightning.pytorch as L
        from slices.data.datamodule import ICUDataModule

        assert issubclass(ICUDataModule, L.LightningDataModule)

    def test_dataset_inherits_torch_dataset(self) -> None:
        """Test that ICUDataset inherits from torch Dataset."""
        from slices.data.dataset import ICUDataset
        from torch.utils.data import Dataset

        assert issubclass(ICUDataset, Dataset)


class TestPackageDependencies:
    """Tests for package dependencies availability."""

    def test_core_dependencies_available(self) -> None:
        """Test that core dependencies can be imported."""
        import duckdb
        import lightning.pytorch
        import polars
        import torch
        import yaml

        assert torch is not None
        assert polars is not None
        assert duckdb is not None
        assert yaml is not None
        assert lightning.pytorch is not None

    def test_torch_cuda_status(self) -> None:
        """Test that we can check CUDA availability (doesn't require CUDA)."""
        import torch

        # Should be able to check without error
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)
