"""Embedding quality analysis and diagnostics.

Provides metrics to detect embedding collapse, assess effective dimensionality,
and analyze embedding space structure.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from sklearn.decomposition import PCA

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _to_numpy(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert input to numpy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingStats:
    """Statistics for a set of embeddings.

    Attributes:
        n_samples: Number of embedding vectors.
        embedding_dim: Dimensionality of embeddings.
        mean: Per-dimension mean (shape: [embedding_dim]).
        std: Per-dimension standard deviation.
        min_val: Per-dimension minimum.
        max_val: Per-dimension maximum.
        global_mean: Mean across all values.
        global_std: Std across all values.
    """

    n_samples: int
    embedding_dim: int
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    global_mean: float
    global_std: float


@dataclass
class DimensionalityMetrics:
    """Effective dimensionality metrics.

    Attributes:
        pca_explained_variance: Variance explained by each PC.
        pca_cumulative_variance: Cumulative variance by component.
        effective_rank: Effective rank (entropy of normalized eigenvalues).
        participation_ratio: Participation ratio.
        variance_95_dims: Dimensions needed for 95% variance.
        variance_99_dims: Dimensions needed for 99% variance.
    """

    pca_explained_variance: np.ndarray
    pca_cumulative_variance: np.ndarray
    effective_rank: float
    participation_ratio: float
    variance_95_dims: int
    variance_99_dims: int


@dataclass
class CollapseMetrics:
    """Metrics for detecting embedding collapse.

    Attributes:
        is_collapsed: Whether embeddings appear collapsed.
        collapse_score: 0-1 score (1 = fully collapsed).
        dead_dimensions: Number of dimensions with near-zero variance.
        dead_dimension_indices: Indices of dead dimensions.
        mean_cosine_similarity: Average pairwise cosine similarity.
        std_cosine_similarity: Std of pairwise cosine similarities.
        uniformity_loss: Uniformity loss (higher = more collapsed).
    """

    is_collapsed: bool
    collapse_score: float
    dead_dimensions: int
    dead_dimension_indices: List[int]
    mean_cosine_similarity: float
    std_cosine_similarity: float
    uniformity_loss: float


@dataclass
class EmbeddingQualityReport:
    """Complete embedding quality report.

    Attributes:
        stats: Basic statistics.
        dimensionality: Effective dimensionality metrics.
        collapse: Collapse detection metrics.
        warnings: List of warning messages.
        recommendations: List of recommendations.
    """

    stats: EmbeddingStats
    dimensionality: DimensionalityMetrics
    collapse: CollapseMetrics
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------


def compute_embedding_stats(
    embeddings: Union[np.ndarray, "torch.Tensor"],
) -> EmbeddingStats:
    """Compute basic statistics for embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).

    Returns:
        EmbeddingStats with per-dimension and global statistics.
    """
    emb = _to_numpy(embeddings)
    n_samples, embedding_dim = emb.shape

    return EmbeddingStats(
        n_samples=n_samples,
        embedding_dim=embedding_dim,
        mean=np.mean(emb, axis=0),
        std=np.std(emb, axis=0),
        min_val=np.min(emb, axis=0),
        max_val=np.max(emb, axis=0),
        global_mean=float(np.mean(emb)),
        global_std=float(np.std(emb)),
    )


def compute_pca_metrics(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    n_components: Optional[int] = None,
) -> DimensionalityMetrics:
    """Compute PCA-based dimensionality metrics.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        n_components: Number of PCA components (default: min(n_samples, dim)).

    Returns:
        DimensionalityMetrics with variance analysis.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for PCA metrics. " "Install with: pip install scikit-learn"
        )

    emb = _to_numpy(embeddings)
    n_samples, embedding_dim = emb.shape

    if n_components is None:
        n_components = min(n_samples, embedding_dim)

    pca = PCA(n_components=n_components)
    pca.fit(emb)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Compute effective rank and participation ratio
    effective_rank = compute_effective_rank(pca.explained_variance_)
    participation_ratio = compute_participation_ratio(pca.explained_variance_)

    # Find dimensions for 95% and 99% variance
    variance_95_dims = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    variance_99_dims = int(np.searchsorted(cumulative_variance, 0.99) + 1)

    return DimensionalityMetrics(
        pca_explained_variance=explained_variance,
        pca_cumulative_variance=cumulative_variance,
        effective_rank=effective_rank,
        participation_ratio=participation_ratio,
        variance_95_dims=min(variance_95_dims, n_components),
        variance_99_dims=min(variance_99_dims, n_components),
    )


def compute_effective_rank(eigenvalues: np.ndarray) -> float:
    """Compute effective rank from eigenvalues.

    Effective rank = exp(entropy of normalized eigenvalues)
    This gives a continuous measure of dimensionality.

    Args:
        eigenvalues: PCA eigenvalues (explained variances).

    Returns:
        Effective rank (1 to D, where D is number of eigenvalues).
    """
    # Normalize to probabilities
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    if total == 0:
        return 0.0

    p = eigenvalues / total
    # Filter zeros to avoid log(0)
    p = p[p > 0]

    # Compute entropy
    entropy = -np.sum(p * np.log(p))

    return float(np.exp(entropy))


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute participation ratio.

    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
    This measures how many dimensions participate in the variance.

    Args:
        eigenvalues: PCA eigenvalues.

    Returns:
        Participation ratio (1 = all variance in one dim, D = uniform).
    """
    eigenvalues = np.abs(eigenvalues)
    sum_sq = np.sum(eigenvalues) ** 2
    sq_sum = np.sum(eigenvalues**2)

    if sq_sum == 0:
        return 0.0

    return float(sum_sq / sq_sum)


def detect_collapse(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    dead_dim_threshold: float = 1e-6,
    cosine_threshold: float = 0.95,
    n_pairs: int = 10000,
) -> CollapseMetrics:
    """Detect embedding collapse and dead dimensions.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        dead_dim_threshold: Variance threshold for dead dimensions.
        cosine_threshold: Cosine similarity threshold for collapse detection.
        n_pairs: Number of random pairs for cosine similarity.

    Returns:
        CollapseMetrics with collapse detection results.
    """
    emb = _to_numpy(embeddings)
    n_samples, embedding_dim = emb.shape

    # Find dead dimensions (near-zero variance)
    dim_std = np.std(emb, axis=0)
    dead_dims = np.where(dim_std < dead_dim_threshold)[0]

    # Sample pairwise cosine similarities
    mean_cosine, std_cosine = _sample_cosine_similarities(emb, n_pairs)

    # Compute uniformity loss
    uniformity = compute_uniformity_loss(emb, n_pairs=n_pairs)

    # Determine collapse
    # Collapsed if: high cosine similarity OR many dead dimensions OR low uniformity
    collapse_score = 0.0
    if mean_cosine > cosine_threshold:
        collapse_score += 0.4
    collapse_score += 0.3 * (len(dead_dims) / embedding_dim)
    collapse_score += 0.3 * min(1.0, uniformity / 5.0)  # uniformity > 5 is very bad

    is_collapsed = collapse_score > 0.5 or mean_cosine > cosine_threshold

    return CollapseMetrics(
        is_collapsed=is_collapsed,
        collapse_score=min(1.0, collapse_score),
        dead_dimensions=len(dead_dims),
        dead_dimension_indices=dead_dims.tolist(),
        mean_cosine_similarity=mean_cosine,
        std_cosine_similarity=std_cosine,
        uniformity_loss=uniformity,
    )


def _sample_cosine_similarities(
    embeddings: np.ndarray,
    n_pairs: int,
) -> Tuple[float, float]:
    """Sample pairwise cosine similarities.

    Args:
        embeddings: Embedding array.
        n_pairs: Number of pairs to sample.

    Returns:
        Tuple of (mean, std) cosine similarity.
    """
    n_samples = len(embeddings)
    if n_samples < 2:
        return 0.0, 0.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    normalized = embeddings / norms

    # Sample random pairs
    n_pairs = min(n_pairs, n_samples * (n_samples - 1) // 2)
    rng = np.random.default_rng(42)

    cosines = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_samples, size=2, replace=False)
        cosine = np.dot(normalized[i], normalized[j])
        cosines.append(cosine)

    return float(np.mean(cosines)), float(np.std(cosines))


def compute_uniformity_loss(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    t: float = 2.0,
    n_pairs: int = 10000,
) -> float:
    """Compute uniformity loss (Wang & Isola, 2020).

    Uniformity measures how well embeddings are spread on the hypersphere.
    Lower is better (more uniform distribution).

    L_uniform = log E[exp(-t * ||z_i - z_j||^2)]

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        t: Temperature parameter.
        n_pairs: Number of random pairs to sample.

    Returns:
        Uniformity loss value (lower is better).
    """
    emb = _to_numpy(embeddings)
    n_samples = len(emb)

    if n_samples < 2:
        return 0.0

    # Normalize to unit sphere
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = emb / norms

    # Sample random pairs
    n_pairs = min(n_pairs, n_samples * (n_samples - 1) // 2)
    rng = np.random.default_rng(42)

    sq_dists = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_samples, size=2, replace=False)
        sq_dist = np.sum((normalized[i] - normalized[j]) ** 2)
        sq_dists.append(sq_dist)

    sq_dists = np.array(sq_dists)

    # Uniformity loss
    uniformity = np.log(np.mean(np.exp(-t * sq_dists)))

    return float(uniformity)


# ---------------------------------------------------------------------------
# High-Level Analysis
# ---------------------------------------------------------------------------


def analyze_embeddings(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    dead_dim_threshold: float = 1e-6,
    cosine_threshold: float = 0.95,
) -> EmbeddingQualityReport:
    """Run complete embedding quality analysis.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        dead_dim_threshold: Threshold for dead dimension detection.
        cosine_threshold: Threshold for collapse detection.

    Returns:
        Complete EmbeddingQualityReport with warnings and recommendations.
    """
    emb = _to_numpy(embeddings)

    # Compute all metrics
    stats = compute_embedding_stats(emb)
    dimensionality = compute_pca_metrics(emb)
    collapse = detect_collapse(emb, dead_dim_threshold, cosine_threshold)

    # Generate warnings and recommendations
    warnings = []
    recommendations = []

    # Check for collapse
    if collapse.is_collapsed:
        warnings.append("Embeddings appear to be collapsed!")
        recommendations.append("Consider using a contrastive loss or increasing model capacity.")

    # Check for dead dimensions
    dead_ratio = collapse.dead_dimensions / stats.embedding_dim
    if dead_ratio > 0.1:
        warnings.append(
            f"{collapse.dead_dimensions} dimensions ({dead_ratio:.1%}) have near-zero variance."
        )
        recommendations.append("Consider reducing embedding dimension or using regularization.")

    # Check effective dimensionality
    if dimensionality.effective_rank < stats.embedding_dim * 0.1:
        warnings.append(
            f"Low effective rank ({dimensionality.effective_rank:.1f}) "
            f"compared to embedding dim ({stats.embedding_dim})."
        )
        recommendations.append(
            "Model may be underfitting. Try increasing model capacity or training longer."
        )

    # Check for high cosine similarity
    if collapse.mean_cosine_similarity > 0.8:
        warnings.append(f"High mean cosine similarity ({collapse.mean_cosine_similarity:.3f}).")
        if collapse.mean_cosine_similarity < cosine_threshold:
            recommendations.append(
                "Embeddings are becoming similar but not fully collapsed. Monitor training."
            )

    # Check variance concentration
    if dimensionality.variance_95_dims < stats.embedding_dim * 0.1:
        warnings.append(
            f"95% variance explained by only {dimensionality.variance_95_dims} dimensions."
        )

    return EmbeddingQualityReport(
        stats=stats,
        dimensionality=dimensionality,
        collapse=collapse,
        warnings=warnings,
        recommendations=recommendations,
    )


def compare_embeddings(
    embeddings_a: Union[np.ndarray, "torch.Tensor"],
    embeddings_b: Union[np.ndarray, "torch.Tensor"],
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, Dict]:
    """Compare two sets of embeddings.

    Useful for comparing train vs test embeddings, or embeddings from
    different checkpoints.

    Args:
        embeddings_a: First embedding set.
        embeddings_b: Second embedding set.
        label_a: Label for first set.
        label_b: Label for second set.

    Returns:
        Dict with comparative metrics for each set.
    """
    report_a = analyze_embeddings(embeddings_a)
    report_b = analyze_embeddings(embeddings_b)

    return {
        label_a: {
            "n_samples": report_a.stats.n_samples,
            "global_mean": report_a.stats.global_mean,
            "global_std": report_a.stats.global_std,
            "effective_rank": report_a.dimensionality.effective_rank,
            "variance_95_dims": report_a.dimensionality.variance_95_dims,
            "is_collapsed": report_a.collapse.is_collapsed,
            "mean_cosine_similarity": report_a.collapse.mean_cosine_similarity,
            "dead_dimensions": report_a.collapse.dead_dimensions,
        },
        label_b: {
            "n_samples": report_b.stats.n_samples,
            "global_mean": report_b.stats.global_mean,
            "global_std": report_b.stats.global_std,
            "effective_rank": report_b.dimensionality.effective_rank,
            "variance_95_dims": report_b.dimensionality.variance_95_dims,
            "is_collapsed": report_b.collapse.is_collapsed,
            "mean_cosine_similarity": report_b.collapse.mean_cosine_similarity,
            "dead_dimensions": report_b.collapse.dead_dimensions,
        },
    }


# ---------------------------------------------------------------------------
# Model Integration
# ---------------------------------------------------------------------------


def extract_embeddings_from_model(
    model: "torch.nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    device: Optional["torch.device"] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from a trained encoder.

    Args:
        model: Encoder model. Should accept (timeseries, mask) and return embeddings.
        dataloader: DataLoader yielding batches with 'timeseries' and 'mask' keys.
        device: Device to run inference on (default: CPU).
        max_samples: Maximum samples to extract (None = all).

    Returns:
        Tuple of (embeddings array, stay_ids array).

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for model inference.")

    import torch

    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_stay_ids = []
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            timeseries = batch["timeseries"].to(device)
            mask = batch["mask"].to(device)
            stay_ids = batch.get("stay_id", None)

            # Get embeddings from model
            embeddings = model(timeseries, mask)

            all_embeddings.append(embeddings.cpu().numpy())
            if stay_ids is not None:
                all_stay_ids.extend(stay_ids.cpu().numpy().tolist())

            n_samples += len(embeddings)
            if max_samples is not None and n_samples >= max_samples:
                break

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    if max_samples is not None:
        embeddings_array = embeddings_array[:max_samples]

    stay_ids_array = np.array(all_stay_ids[:max_samples] if all_stay_ids else [])

    return embeddings_array, stay_ids_array


def load_embeddings_from_file(
    path: Union[str, Path],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load embeddings from file (NPZ or PT format).

    Args:
        path: Path to embedding file.

    Returns:
        Tuple of (embeddings, stay_ids or None).
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path)
        embeddings = data["embeddings"]
        stay_ids = data.get("stay_ids", None)
    elif path.suffix == ".pt":
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to load .pt files.")
        import torch

        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            embeddings = _to_numpy(data["embeddings"])
            stay_ids = _to_numpy(data.get("stay_ids")) if "stay_ids" in data else None
        else:
            embeddings = _to_numpy(data)
            stay_ids = None
    elif path.suffix == ".npy":
        embeddings = np.load(path)
        stay_ids = None
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return embeddings, stay_ids


def save_embeddings_to_file(
    embeddings: Union[np.ndarray, "torch.Tensor"],
    path: Union[str, Path],
    stay_ids: Optional[Union[np.ndarray, List[int]]] = None,
) -> Path:
    """Save embeddings to file.

    Args:
        embeddings: Embedding array.
        path: Output path (.npz or .pt).
        stay_ids: Optional stay IDs to save alongside embeddings.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    emb = _to_numpy(embeddings)

    if path.suffix == ".npz":
        if stay_ids is not None:
            np.savez(path, embeddings=emb, stay_ids=np.asarray(stay_ids))
        else:
            np.savez(path, embeddings=emb)
    elif path.suffix == ".pt":
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to save .pt files.")
        import torch

        data = {"embeddings": torch.from_numpy(emb)}
        if stay_ids is not None:
            data["stay_ids"] = torch.tensor(stay_ids)
        torch.save(data, path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return path
