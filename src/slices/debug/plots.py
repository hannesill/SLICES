"""Visualization functions for debugging.

Optional matplotlib dependency with graceful fallback.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

# Optional imports with graceful fallback
try:
    import matplotlib.figure
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import umap

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

if TYPE_CHECKING:
    pass


def _require_matplotlib() -> None:
    """Raise helpful error if matplotlib not installed."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. " "Install with: pip install matplotlib"
        )


def _require_sklearn() -> None:
    """Raise helpful error if sklearn not installed."""
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for dimensionality reduction. "
            "Install with: pip install scikit-learn"
        )


def _save_or_show(fig: "matplotlib.figure.Figure", save_path: Optional[Path]) -> None:
    """Save figure to file or display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Embedding Visualizations
# ---------------------------------------------------------------------------


def plot_pca_variance(
    explained_variance: np.ndarray,
    n_components: int = 50,
    title: str = "PCA Explained Variance",
    save_path: Optional[Path] = None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot PCA explained variance (scree plot).

    Args:
        explained_variance: Array of explained variance ratios from PCA.
        n_components: Number of components to show.
        title: Plot title.
        save_path: Optional path to save figure.

    Returns:
        Figure object if matplotlib available.
    """
    _require_matplotlib()

    n_show = min(n_components, len(explained_variance))
    cumulative = np.cumsum(explained_variance)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Individual variance
    ax1.bar(range(n_show), explained_variance[:n_show], alpha=0.7)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Individual Explained Variance")

    # Cumulative variance
    ax2.plot(range(n_show), cumulative[:n_show], "b-", linewidth=2)
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
    ax2.axhline(y=0.99, color="orange", linestyle="--", label="99% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Explained Variance")
    ax2.legend()
    ax2.set_ylim([0, 1.05])

    fig.suptitle(title)
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_embedding_2d(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "pca",
    title: str = "Embedding Space",
    colorbar_label: str = "Label",
    save_path: Optional[Path] = None,
    **kwargs: Any,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot 2D projection of embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        labels: Optional labels for coloring points (numeric or categorical).
        method: Dimensionality reduction method ('pca', 'tsne', 'umap').
        title: Plot title.
        colorbar_label: Label for colorbar (if labels are numeric).
        save_path: Optional path to save figure.
        **kwargs: Additional kwargs for reduction method.

    Returns:
        Figure object if available.
    """
    _require_matplotlib()

    # Perform dimensionality reduction
    if method == "pca":
        _require_sklearn()
        reducer = PCA(n_components=2, **kwargs)
        coords = reducer.fit_transform(embeddings)
        subtitle = f"PCA (var explained: {reducer.explained_variance_ratio_.sum():.1%})"
    elif method == "tsne":
        _require_sklearn()
        perplexity = kwargs.pop("perplexity", min(30, len(embeddings) - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, **kwargs)
        coords = reducer.fit_transform(embeddings)
        subtitle = f"t-SNE (perplexity={perplexity})"
    elif method == "umap":
        if not _HAS_UMAP:
            raise ImportError(
                "umap-learn is required for UMAP. Install with: pip install umap-learn"
            )
        n_neighbors = kwargs.pop("n_neighbors", min(15, len(embeddings) - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, **kwargs)
        coords = reducer.fit_transform(embeddings)
        subtitle = f"UMAP (n_neighbors={n_neighbors})"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        # Check if labels are categorical or numeric
        unique_labels = np.unique(
            labels[~np.isnan(labels)] if np.issubdtype(labels.dtype, np.floating) else labels
        )

        if len(unique_labels) <= 10:
            # Categorical coloring
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    label=str(label),
                    alpha=0.6,
                    s=20,
                )
            ax.legend(title=colorbar_label)
        else:
            # Continuous coloring
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.6,
                s=20,
            )
            plt.colorbar(scatter, ax=ax, label=colorbar_label)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{title}\n{subtitle}")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_embedding_distribution(
    embeddings: np.ndarray,
    dimension_indices: Optional[List[int]] = None,
    n_dims: int = 9,
    title: str = "Embedding Dimension Distributions",
    save_path: Optional[Path] = None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot histograms of embedding dimensions.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        dimension_indices: Specific dimensions to plot (None = first n_dims).
        n_dims: Number of dimensions if indices not specified.
        title: Plot title.
        save_path: Optional path to save.

    Returns:
        Figure object.
    """
    _require_matplotlib()

    if dimension_indices is None:
        dimension_indices = list(range(min(n_dims, embeddings.shape[1])))

    n_plots = len(dimension_indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (ax, dim) in enumerate(zip(axes, dimension_indices)):
        values = embeddings[:, dim]
        ax.hist(values, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(f"Dimension {dim}")
        ax.set_ylabel("Count")
        ax.set_title(f"Dim {dim}: mean={values.mean():.2f}, std={values.std():.2f}")

    # Hide unused subplots
    for ax in axes[len(dimension_indices) :]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_cosine_similarity_heatmap(
    embeddings: np.ndarray,
    n_samples: int = 100,
    title: str = "Pairwise Cosine Similarity",
    save_path: Optional[Path] = None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot heatmap of pairwise cosine similarities.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        n_samples: Number of samples for heatmap.
        title: Plot title.
        save_path: Optional path to save.

    Returns:
        Figure object.
    """
    _require_matplotlib()

    # Subsample if needed
    if len(embeddings) > n_samples:
        indices = np.random.default_rng(42).choice(len(embeddings), n_samples, replace=False)
        emb = embeddings[indices]
    else:
        emb = embeddings

    # Normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = emb / norms

    # Compute cosine similarity matrix
    cos_sim = normalized @ normalized.T

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(ax.images[0], ax=ax, label="Cosine Similarity")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    ax.set_title(f"{title}\n(mean={cos_sim.mean():.3f}, diag excluded)")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Pipeline Visualizations
# ---------------------------------------------------------------------------


def plot_missingness_heatmap(
    mask: np.ndarray,
    feature_names: List[str],
    stay_ids: Optional[List[int]] = None,
    n_stays: int = 20,
    title: str = "Observation Mask",
    save_path: Optional[Path] = None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot heatmap of observation mask for selected stays.

    Args:
        mask: Array of shape (n_stays, seq_len, n_features) or (seq_len, n_features).
        feature_names: Feature names for y-axis.
        stay_ids: Stay IDs for labeling rows.
        n_stays: Number of stays to show.
        title: Plot title.
        save_path: Optional path to save.

    Returns:
        Figure object.
    """
    _require_matplotlib()

    # Handle single patient case
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]

    # Subsample stays
    if len(mask) > n_stays:
        mask = mask[:n_stays]
        if stay_ids is not None:
            stay_ids = stay_ids[:n_stays]

    n_stays_actual, seq_len, n_features = mask.shape

    # Create figure with subplots for each patient
    fig, axes = plt.subplots(n_stays_actual, 1, figsize=(12, 2 * n_stays_actual), sharex=True)
    if n_stays_actual == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.imshow(
            mask[idx].T, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_ylabel("Feature")
        if stay_ids is not None:
            ax.set_title(f"Stay {stay_ids[idx]}")
        else:
            ax.set_title(f"Patient {idx}")

        # Show feature names on y-axis for first plot
        if idx == 0 and len(feature_names) <= 20:
            ax.set_yticks(range(n_features))
            ax.set_yticklabels(feature_names, fontsize=6)

    axes[-1].set_xlabel("Hour")

    fig.suptitle(f"{title}\n(Green = Observed)")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_patient_timeseries(
    stay_id: int,
    timeseries: np.ndarray,
    mask: np.ndarray,
    feature_names: List[str],
    features_to_plot: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot timeseries for a single patient.

    Args:
        stay_id: Patient stay ID.
        timeseries: Array of shape (seq_len, n_features).
        mask: Observation mask of same shape.
        feature_names: Feature names.
        features_to_plot: Specific features to plot (None = first 6).
        title: Plot title (default: f"Stay {stay_id}").
        save_path: Optional path to save.

    Returns:
        Figure object.
    """
    _require_matplotlib()

    if features_to_plot is None:
        features_to_plot = feature_names[:6]

    n_features = len(features_to_plot)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    hours = np.arange(timeseries.shape[0])

    for ax, feat_name in zip(axes, features_to_plot):
        if feat_name not in feature_to_idx:
            ax.set_visible(False)
            continue

        feat_idx = feature_to_idx[feat_name]
        values = timeseries[:, feat_idx]
        observed = mask[:, feat_idx].astype(bool)

        # Plot all values as line
        ax.plot(hours, values, "b-", alpha=0.3, linewidth=1)

        # Mark observed values
        ax.scatter(
            hours[observed],
            values[observed],
            c="blue",
            s=20,
            label="Observed",
            zorder=5,
        )

        # Mark imputed/missing values
        ax.scatter(
            hours[~observed],
            values[~observed],
            c="red",
            s=20,
            marker="x",
            label="Missing",
            zorder=5,
        )

        ax.set_xlabel("Hour")
        ax.set_ylabel(feat_name)
        ax.set_title(feat_name)
        ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for ax in axes[len(features_to_plot) :]:
        ax.set_visible(False)

    fig.suptitle(title or f"Stay {stay_id}")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_feature_distributions(
    values_dict: dict,
    title: str = "Feature Value Distributions",
    n_features: int = 9,
    save_path: Optional[Path] = None,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot histograms of feature values across all patients.

    Args:
        values_dict: Dict mapping feature names to value arrays.
        title: Plot title.
        n_features: Max number of features to plot.
        save_path: Optional path to save.

    Returns:
        Figure object.
    """
    _require_matplotlib()

    feature_names = list(values_dict.keys())[:n_features]
    n_plots = len(feature_names)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for ax, feat_name in zip(axes, feature_names):
        values = values_dict[feat_name]
        values = values[~np.isnan(values)]  # Remove NaN

        if len(values) > 0:
            ax.hist(values, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{feat_name}\n(n={len(values)}, mean={values.mean():.2f})")
        else:
            ax.set_title(f"{feat_name}\n(no data)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for ax in axes[len(feature_names) :]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_debug_report(
    processed_dir: Path,
    embeddings: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Generate comprehensive debug report with all visualizations.

    Args:
        processed_dir: Directory with processed parquet files.
        embeddings: Optional embeddings array.
        labels: Optional labels for embedding coloring.
        output_dir: Where to save plots (default: processed_dir/debug_plots).

    Returns:
        Dict mapping plot names to file paths.
    """
    _require_matplotlib()

    import polars as pl
    import yaml

    processed_dir = Path(processed_dir)
    if output_dir is None:
        output_dir = processed_dir / "debug_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # Load metadata
    metadata_path = processed_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        feature_names = metadata.get("feature_names", [])
    else:
        feature_names = []

    # Load timeseries for missingness visualization
    timeseries_path = processed_dir / "timeseries.parquet"
    if timeseries_path.exists():
        timeseries_df = pl.read_parquet(timeseries_path)

        # Get first few patients for missingness heatmap
        sample_df = timeseries_df.head(10)
        masks = np.array([row["mask"] for row in sample_df.iter_rows(named=True)])
        stay_ids = sample_df["stay_id"].to_list()

        path = output_dir / "missingness_heatmap.png"
        plot_missingness_heatmap(
            masks,
            feature_names,
            stay_ids=stay_ids,
            save_path=path,
        )
        exported["missingness_heatmap"] = path

    # Embedding plots if provided
    if embeddings is not None:
        from .embeddings import compute_pca_metrics

        # PCA variance
        pca_metrics = compute_pca_metrics(embeddings)
        path = output_dir / "pca_variance.png"
        plot_pca_variance(
            pca_metrics.pca_explained_variance,
            save_path=path,
        )
        exported["pca_variance"] = path

        # 2D projections
        for method in ["pca", "tsne"]:
            path = output_dir / f"embedding_{method}.png"
            plot_embedding_2d(
                embeddings,
                labels=labels,
                method=method,
                save_path=path,
            )
            exported[f"embedding_{method}"] = path

        # Distribution
        path = output_dir / "embedding_distributions.png"
        plot_embedding_distribution(embeddings, save_path=path)
        exported["embedding_distributions"] = path

        # Cosine similarity
        path = output_dir / "cosine_similarity.png"
        plot_cosine_similarity_heatmap(embeddings, save_path=path)
        exported["cosine_similarity"] = path

    return exported
