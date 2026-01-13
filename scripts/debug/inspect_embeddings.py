"""CLI for embedding quality analysis and visualization.

Analyzes embedding quality to detect common issues like collapse,
dead dimensions, and poor variance distribution.

Example usage:
    # Analyze embeddings from a checkpoint
    uv run python scripts/debug/inspect_embeddings.py \
        checkpoint=outputs/encoder.pt \
        processed_dir=data/processed/mimic-iv-demo

    # Load pre-computed embeddings
    uv run python scripts/debug/inspect_embeddings.py \
        embeddings_file=outputs/embeddings.npz

    # Generate visualizations
    uv run python scripts/debug/inspect_embeddings.py \
        checkpoint=outputs/encoder.pt \
        processed_dir=data/processed/mimic-iv-demo \
        plots=true
"""

from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from slices.debug import (
    EmbeddingQualityReport,
    analyze_embeddings,
    load_embeddings_from_file,
    save_embeddings_to_file,
)


def print_report(report: EmbeddingQualityReport) -> None:
    """Print embedding quality report to console."""
    print("\n" + "=" * 60)
    print("EMBEDDING QUALITY REPORT")
    print("=" * 60)

    print("\n[Statistics]")
    print(f"  Samples: {report.stats.n_samples:,}")
    print(f"  Dimensions: {report.stats.embedding_dim}")
    print(f"  Global mean: {report.stats.global_mean:.4f}")
    print(f"  Global std: {report.stats.global_std:.4f}")

    print("\n[Dimensionality]")
    print(f"  Effective rank: {report.dimensionality.effective_rank:.1f}")
    print(f"  Participation ratio: {report.dimensionality.participation_ratio:.1f}")
    print(f"  Dims for 95% variance: {report.dimensionality.variance_95_dims}")
    print(f"  Dims for 99% variance: {report.dimensionality.variance_99_dims}")

    print("\n[Collapse Detection]")
    collapse_status = "YES - Embeddings appear collapsed!" if report.collapse.is_collapsed else "No"
    print(f"  Collapsed: {collapse_status}")
    print(f"  Collapse score: {report.collapse.collapse_score:.3f}")
    print(f"  Dead dimensions: {report.collapse.dead_dimensions}")
    print(f"  Mean cosine sim: {report.collapse.mean_cosine_similarity:.3f}")
    print(f"  Uniformity loss: {report.collapse.uniformity_loss:.3f}")

    if report.warnings:
        print("\n[Warnings]")
        for w in report.warnings:
            print(f"  ! {w}")

    if report.recommendations:
        print("\n[Recommendations]")
        for r in report.recommendations:
            print(f"  -> {r}")

    print("\n" + "=" * 60)


def extract_embeddings_from_checkpoint(
    checkpoint_path: str,
    processed_dir: str,
    max_samples: int = 5000,
) -> tuple:
    """Extract embeddings from a trained encoder checkpoint.

    Args:
        checkpoint_path: Path to encoder checkpoint.
        processed_dir: Path to processed data directory.
        max_samples: Maximum samples to extract.

    Returns:
        Tuple of (embeddings, stay_ids, labels).
    """
    import torch
    from slices.data.datamodule import ICUDataModule
    from slices.models.encoders import build_encoder

    # Setup data
    dm = ICUDataModule(processed_dir=processed_dir)
    dm.setup()

    # Load encoder
    print(f"Loading encoder from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "encoder_name" in checkpoint:
        encoder = build_encoder(
            checkpoint["encoder_name"],
            checkpoint["encoder_config"],
        )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "state_dict" in checkpoint:
        # Lightning checkpoint format
        from slices.training import SSLPretrainModule

        model = SSLPretrainModule.load_from_checkpoint(checkpoint_path)
        encoder = model.encoder
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {checkpoint.keys()}")

    encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    # Extract embeddings
    print(f"Extracting embeddings (max {max_samples} samples)...")
    all_embeddings = []
    all_stay_ids = []
    n_samples = 0

    with torch.no_grad():
        for batch in dm.val_dataloader():
            timeseries = batch["timeseries"].to(device)
            mask = batch["mask"].to(device)

            embeddings = encoder(timeseries, mask)
            all_embeddings.append(embeddings.cpu().numpy())

            if "stay_id" in batch:
                all_stay_ids.extend(batch["stay_id"].tolist())

            n_samples += len(embeddings)
            if n_samples >= max_samples:
                break

    embeddings_array = np.concatenate(all_embeddings, axis=0)[:max_samples]
    stay_ids_array = np.array(all_stay_ids[:max_samples]) if all_stay_ids else None

    # Load labels for coloring
    labels_path = Path(processed_dir) / "labels.parquet"
    labels = None
    if labels_path.exists() and stay_ids_array is not None:
        labels_df = pl.read_parquet(labels_path)
        # Get first label column that's not stay_id
        label_cols = [c for c in labels_df.columns if c != "stay_id"]
        if label_cols:
            label_col = label_cols[0]
            stay_to_label = dict(
                zip(
                    labels_df["stay_id"].to_list(),
                    labels_df[label_col].to_list(),
                )
            )
            labels = np.array([stay_to_label.get(sid, np.nan) for sid in stay_ids_array])

    return embeddings_array, stay_ids_array, labels


@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(cfg: DictConfig) -> None:
    """Analyze embedding quality."""
    print("=" * 70)
    print("Embedding Quality Analysis")
    print("=" * 70)

    labels = None
    stay_ids = None

    # Load embeddings
    if cfg.get("embeddings_file"):
        embeddings, stay_ids = load_embeddings_from_file(cfg.embeddings_file)
        print(f"\nLoaded embeddings from {cfg.embeddings_file}")
        print(f"  Shape: {embeddings.shape}")
    elif cfg.get("checkpoint"):
        if not cfg.get("processed_dir"):
            raise ValueError("processed_dir is required when using checkpoint")

        embeddings, stay_ids, labels = extract_embeddings_from_checkpoint(
            cfg.checkpoint,
            cfg.processed_dir,
            max_samples=cfg.get("max_samples", 5000),
        )
        print(f"  Shape: {embeddings.shape}")
    else:
        raise ValueError("Must provide embeddings_file or checkpoint")

    # Analyze embeddings
    print("\nAnalyzing embeddings...")
    report = analyze_embeddings(embeddings)
    print_report(report)

    # Save embeddings if extracted from checkpoint
    if cfg.get("checkpoint") and not cfg.get("embeddings_file"):
        output_dir = Path(cfg.get("output_dir") or Path(cfg.processed_dir) / "debug_embeddings")
        output_dir.mkdir(parents=True, exist_ok=True)

        emb_path = output_dir / "embeddings.npz"
        save_embeddings_to_file(embeddings, emb_path, stay_ids=stay_ids)
        print(f"\nSaved embeddings to {emb_path}")

    # Generate plots if requested
    if cfg.get("plots", False):
        try:
            from slices.debug import (
                plot_cosine_similarity_heatmap,
                plot_embedding_2d,
                plot_embedding_distribution,
                plot_pca_variance,
            )

            output_dir = Path(cfg.get("output_dir") or "debug_plots")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nGenerating plots in {output_dir}...")

            # PCA variance
            plot_pca_variance(
                report.dimensionality.pca_explained_variance,
                save_path=output_dir / "pca_variance.png",
            )
            print("  - pca_variance.png")

            # 2D projections
            for method in ["pca", "tsne"]:
                plot_embedding_2d(
                    embeddings,
                    labels=labels,
                    method=method,
                    colorbar_label="Mortality" if labels is not None else "Sample",
                    save_path=output_dir / f"embedding_{method}.png",
                )
                print(f"  - embedding_{method}.png")

            # Distributions
            plot_embedding_distribution(
                embeddings,
                save_path=output_dir / "embedding_distributions.png",
            )
            print("  - embedding_distributions.png")

            # Cosine similarity
            plot_cosine_similarity_heatmap(
                embeddings,
                save_path=output_dir / "cosine_similarity.png",
            )
            print("  - cosine_similarity.png")

            print(f"\nPlots saved to {output_dir}")

        except ImportError as e:
            print(f"\nSkipping plots (missing dependency): {e}")
            print("Install with: pip install matplotlib")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
