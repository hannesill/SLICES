"""Tests for slices/debug/embeddings.py.

Tests embedding quality analysis, collapse detection, and dimensionality metrics.
"""

import numpy as np
import pytest
import torch
from slices.debug.embeddings import (
    CollapseMetrics,
    DimensionalityMetrics,
    EmbeddingQualityReport,
    EmbeddingStats,
    _sample_cosine_similarities,
    _to_numpy,
    analyze_embeddings,
    compare_embeddings,
    compute_effective_rank,
    compute_embedding_stats,
    compute_participation_ratio,
    compute_pca_metrics,
    compute_uniformity_loss,
    detect_collapse,
    load_embeddings_from_file,
    save_embeddings_to_file,
)


class TestToNumpy:
    """Tests for _to_numpy helper function."""

    def test_to_numpy_from_numpy(self):
        """_to_numpy should pass through numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_from_torch(self):
        """_to_numpy should convert torch tensors."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_numpy_from_list(self):
        """_to_numpy should convert lists to numpy arrays."""
        data = [1.0, 2.0, 3.0]
        result = _to_numpy(data)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)


class TestComputeEmbeddingStats:
    """Tests for compute_embedding_stats function."""

    def test_stats_shape_info(self):
        """Stats should include correct shape information."""
        embeddings = np.random.randn(100, 64)
        stats = compute_embedding_stats(embeddings)

        assert stats.n_samples == 100
        assert stats.embedding_dim == 64

    def test_stats_mean_and_std(self):
        """Stats should compute correct mean and std."""
        # Create embeddings with known properties
        embeddings = np.ones((50, 10)) * 5.0
        embeddings[:, 0] = np.arange(50)  # First dim varies from 0 to 49

        stats = compute_embedding_stats(embeddings)

        # Mean of first dim should be ~24.5
        assert stats.mean[0] == pytest.approx(24.5, abs=0.1)
        # Other dims have mean 5
        assert stats.mean[1] == pytest.approx(5.0)

    def test_stats_min_max(self):
        """Stats should compute correct min and max per dimension."""
        embeddings = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]])
        stats = compute_embedding_stats(embeddings)

        np.testing.assert_array_equal(stats.min_val, [0.0, 0.0])
        np.testing.assert_array_equal(stats.max_val, [10.0, 10.0])

    def test_stats_global_mean_std(self):
        """Stats should compute global mean and std across all values."""
        embeddings = np.zeros((10, 10))
        stats = compute_embedding_stats(embeddings)

        assert stats.global_mean == 0.0
        assert stats.global_std == 0.0

    def test_stats_with_torch_input(self):
        """Stats should work with torch tensor input."""
        embeddings = torch.randn(50, 32)
        stats = compute_embedding_stats(embeddings)

        assert stats.n_samples == 50
        assert stats.embedding_dim == 32


class TestDimensionalityMetrics:
    """Tests for PCA-based dimensionality metrics."""

    def test_pca_metrics_shape(self):
        """PCA metrics should have correct array shapes."""
        embeddings = np.random.randn(100, 32)
        metrics = compute_pca_metrics(embeddings)

        # n_components defaults to min(n_samples, dim) = 32
        assert len(metrics.pca_explained_variance) == 32
        assert len(metrics.pca_cumulative_variance) == 32

    def test_pca_cumulative_variance_monotonic(self):
        """Cumulative variance should be monotonically increasing."""
        embeddings = np.random.randn(100, 32)
        metrics = compute_pca_metrics(embeddings)

        diffs = np.diff(metrics.pca_cumulative_variance)
        assert np.all(diffs >= 0)

    def test_pca_cumulative_variance_ends_at_one(self):
        """Cumulative variance should end at approximately 1.0."""
        embeddings = np.random.randn(100, 32)
        metrics = compute_pca_metrics(embeddings)

        assert metrics.pca_cumulative_variance[-1] == pytest.approx(1.0, abs=0.01)

    def test_variance_95_dims(self):
        """Variance 95% dimensions should be reasonable."""
        embeddings = np.random.randn(100, 32)
        metrics = compute_pca_metrics(embeddings)

        assert 1 <= metrics.variance_95_dims <= 32
        assert 1 <= metrics.variance_99_dims <= 32
        assert metrics.variance_95_dims <= metrics.variance_99_dims


class TestEffectiveRankAndParticipationRatio:
    """Tests for effective rank and participation ratio."""

    def test_effective_rank_uniform_eigenvalues(self):
        """Effective rank of uniform eigenvalues should equal num eigenvalues."""
        # Uniform eigenvalues
        eigenvalues = np.ones(10)
        effective_rank = compute_effective_rank(eigenvalues)

        # exp(entropy) of uniform distribution over 10 items = 10
        assert effective_rank == pytest.approx(10.0, abs=0.1)

    def test_effective_rank_single_dominant(self):
        """Effective rank with single dominant eigenvalue should be ~1."""
        eigenvalues = np.array([1.0, 0.0, 0.0, 0.0])
        effective_rank = compute_effective_rank(eigenvalues)

        assert effective_rank == pytest.approx(1.0, abs=0.1)

    def test_effective_rank_zero_eigenvalues(self):
        """Effective rank of all zeros should be 0."""
        eigenvalues = np.zeros(10)
        effective_rank = compute_effective_rank(eigenvalues)

        assert effective_rank == 0.0

    def test_participation_ratio_uniform(self):
        """Participation ratio of uniform eigenvalues should equal D."""
        eigenvalues = np.ones(8)
        pr = compute_participation_ratio(eigenvalues)

        # (8*1)^2 / (8*1^2) = 64/8 = 8
        assert pr == pytest.approx(8.0)

    def test_participation_ratio_single_dominant(self):
        """Participation ratio with one dominant should be ~1."""
        eigenvalues = np.array([1.0, 0.0, 0.0, 0.0])
        pr = compute_participation_ratio(eigenvalues)

        assert pr == pytest.approx(1.0)

    def test_participation_ratio_zeros(self):
        """Participation ratio of zeros should be 0."""
        eigenvalues = np.zeros(10)
        pr = compute_participation_ratio(eigenvalues)

        assert pr == 0.0


class TestCollapseDetection:
    """Tests for embedding collapse detection."""

    def test_detect_collapse_healthy_embeddings(self):
        """Healthy random embeddings should not be detected as collapsed."""
        # Well-spread embeddings
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 32))

        metrics = detect_collapse(embeddings)

        assert isinstance(metrics, CollapseMetrics)
        assert metrics.is_collapsed is False
        assert metrics.collapse_score < 0.5

    def test_detect_collapse_collapsed_embeddings(self):
        """Identical embeddings should be detected as collapsed."""
        # All embeddings are identical (complete collapse)
        embeddings = np.ones((100, 32))

        metrics = detect_collapse(embeddings)

        assert metrics.is_collapsed is True
        assert metrics.collapse_score > 0.5
        assert metrics.dead_dimensions == 32  # All dims have zero variance

    def test_detect_collapse_dead_dimensions(self):
        """Should detect dimensions with near-zero variance."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 32))
        # Make first 5 dims constant (dead)
        embeddings[:, :5] = 0.0

        metrics = detect_collapse(embeddings)

        assert metrics.dead_dimensions == 5
        assert all(i in metrics.dead_dimension_indices for i in range(5))

    def test_detect_collapse_cosine_similarity(self):
        """Should compute mean cosine similarity correctly."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 16))

        metrics = detect_collapse(embeddings)

        # For random embeddings, mean cosine should be near 0
        assert -0.5 < metrics.mean_cosine_similarity < 0.5

    def test_detect_collapse_high_similarity(self):
        """High cosine similarity should indicate collapse."""
        # Create nearly identical embeddings
        base = np.ones((1, 32))
        noise = np.random.randn(100, 32) * 0.01
        embeddings = base + noise

        metrics = detect_collapse(embeddings, cosine_threshold=0.9)

        assert metrics.mean_cosine_similarity > 0.9
        assert metrics.is_collapsed is True


class TestUniformityLoss:
    """Tests for uniformity loss computation."""

    def test_uniformity_loss_uniform_distribution(self):
        """Well-spread embeddings should have low uniformity loss."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 32))

        uniformity = compute_uniformity_loss(embeddings)

        # Uniformity should be negative for spread embeddings
        assert uniformity < 0

    def test_uniformity_loss_collapsed(self):
        """Collapsed embeddings should have high uniformity loss."""
        # All points at same location
        embeddings = np.ones((100, 32))

        uniformity = compute_uniformity_loss(embeddings)

        # For identical points, sq_dist=0, exp(0)=1, log(1)=0
        # But with identical embeddings, all are at same point
        assert uniformity == pytest.approx(0.0, abs=0.1)

    def test_uniformity_loss_small_sample(self):
        """Uniformity loss should handle small samples."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        uniformity = compute_uniformity_loss(embeddings, n_pairs=100)

        assert isinstance(uniformity, float)


class TestSampleCosineSimilarities:
    """Tests for cosine similarity sampling."""

    def test_sample_cosine_single_sample(self):
        """Single sample should return zero similarity."""
        embeddings = np.array([[1.0, 0.0, 0.0]])
        mean, std = _sample_cosine_similarities(embeddings, n_pairs=10)

        assert mean == 0.0
        assert std == 0.0

    def test_sample_cosine_orthogonal(self):
        """Orthogonal vectors should have zero similarity."""
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        mean, std = _sample_cosine_similarities(embeddings, n_pairs=100)

        assert mean == pytest.approx(0.0, abs=0.01)

    def test_sample_cosine_parallel(self):
        """Parallel vectors should have similarity of 1."""
        embeddings = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        mean, std = _sample_cosine_similarities(embeddings, n_pairs=100)

        assert mean == pytest.approx(1.0, abs=0.01)


class TestAnalyzeEmbeddings:
    """Tests for the high-level analyze_embeddings function."""

    def test_analyze_returns_report(self):
        """analyze_embeddings should return complete report."""
        embeddings = np.random.randn(100, 32)
        report = analyze_embeddings(embeddings)

        assert isinstance(report, EmbeddingQualityReport)
        assert isinstance(report.stats, EmbeddingStats)
        assert isinstance(report.dimensionality, DimensionalityMetrics)
        assert isinstance(report.collapse, CollapseMetrics)

    def test_analyze_generates_warnings_for_collapse(self):
        """analyze_embeddings should warn about collapse."""
        # Collapsed embeddings
        embeddings = np.ones((100, 32))
        report = analyze_embeddings(embeddings)

        assert len(report.warnings) > 0
        assert any("collapsed" in w.lower() for w in report.warnings)

    def test_analyze_generates_warnings_for_dead_dims(self):
        """analyze_embeddings should warn about dead dimensions."""
        embeddings = np.random.randn(100, 32)
        embeddings[:, :10] = 0.0  # 10 dead dims out of 32

        report = analyze_embeddings(embeddings)

        assert any("variance" in w.lower() or "dimension" in w.lower() for w in report.warnings)


class TestCompareEmbeddings:
    """Tests for comparing two embedding sets."""

    def test_compare_returns_dict(self):
        """compare_embeddings should return dict with both sets."""
        emb_a = np.random.randn(50, 16)
        emb_b = np.random.randn(100, 16)

        result = compare_embeddings(emb_a, emb_b, label_a="train", label_b="test")

        assert "train" in result
        assert "test" in result
        assert result["train"]["n_samples"] == 50
        assert result["test"]["n_samples"] == 100

    def test_compare_includes_key_metrics(self):
        """compare_embeddings should include key comparison metrics."""
        emb_a = np.random.randn(50, 16)
        emb_b = np.random.randn(50, 16)

        result = compare_embeddings(emb_a, emb_b)

        for label in ["A", "B"]:
            assert "global_mean" in result[label]
            assert "global_std" in result[label]
            assert "effective_rank" in result[label]
            assert "is_collapsed" in result[label]


class TestEmbeddingIO:
    """Tests for saving and loading embeddings."""

    def test_save_load_npz(self, tmp_path):
        """Should save and load embeddings in NPZ format."""
        embeddings = np.random.randn(50, 32)
        stay_ids = np.arange(50)

        path = tmp_path / "embeddings.npz"
        save_embeddings_to_file(embeddings, path, stay_ids)

        loaded_emb, loaded_ids = load_embeddings_from_file(path)

        np.testing.assert_array_almost_equal(loaded_emb, embeddings)
        np.testing.assert_array_equal(loaded_ids, stay_ids)

    def test_save_load_npz_without_ids(self, tmp_path):
        """Should save and load embeddings without stay_ids."""
        embeddings = np.random.randn(50, 32)

        path = tmp_path / "embeddings.npz"
        save_embeddings_to_file(embeddings, path)

        loaded_emb, loaded_ids = load_embeddings_from_file(path)

        np.testing.assert_array_almost_equal(loaded_emb, embeddings)
        assert loaded_ids is None

    def test_save_load_pt(self, tmp_path):
        """Should save and load embeddings in PT format."""
        embeddings = np.random.randn(50, 32)
        stay_ids = list(range(50))

        path = tmp_path / "embeddings.pt"
        save_embeddings_to_file(embeddings, path, stay_ids)

        loaded_emb, loaded_ids = load_embeddings_from_file(path)

        np.testing.assert_array_almost_equal(loaded_emb, embeddings)

    def test_save_load_npy(self, tmp_path):
        """Should load embeddings from NPY format."""
        embeddings = np.random.randn(50, 32)

        path = tmp_path / "embeddings.npy"
        np.save(path, embeddings)

        loaded_emb, loaded_ids = load_embeddings_from_file(path)

        np.testing.assert_array_almost_equal(loaded_emb, embeddings)
        assert loaded_ids is None

    def test_load_unsupported_format_raises(self, tmp_path):
        """Loading unsupported format should raise ValueError."""
        path = tmp_path / "embeddings.xyz"
        path.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_embeddings_from_file(path)

    def test_save_unsupported_format_raises(self, tmp_path):
        """Saving to unsupported format should raise ValueError."""
        embeddings = np.random.randn(10, 10)
        path = tmp_path / "embeddings.xyz"

        with pytest.raises(ValueError, match="Unsupported file format"):
            save_embeddings_to_file(embeddings, path)

    def test_save_from_torch_tensor(self, tmp_path):
        """Should save embeddings from torch tensor."""
        embeddings = torch.randn(50, 32)

        path = tmp_path / "embeddings.npz"
        save_embeddings_to_file(embeddings, path)

        loaded_emb, _ = load_embeddings_from_file(path)

        np.testing.assert_array_almost_equal(loaded_emb, embeddings.numpy())
