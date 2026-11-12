"""Debug utilities for SLICES.

Provides tools for:
- Sentinel patient selection for targeted debugging
- Pipeline snapshot export for data inspection
- Embedding quality analysis and collapse detection
- Visualization functions (requires matplotlib)

Example:
    >>> from slices.debug import select_sentinel_patients, SentinelConfig
    >>> from slices.debug import analyze_embeddings, EmbeddingQualityReport

    # Select sentinel patients
    >>> config = SentinelConfig(n_per_stratum=5)
    >>> sentinels = select_sentinel_patients(static_df, config)

    # Analyze embeddings
    >>> report = analyze_embeddings(embeddings)
    >>> print(report.warnings)
"""

from .embeddings import (
    CollapseMetrics,
    DimensionalityMetrics,
    EmbeddingQualityReport,
    EmbeddingStats,
    analyze_embeddings,
    compare_embeddings,
    compute_effective_rank,
    compute_embedding_stats,
    compute_participation_ratio,
    compute_pca_metrics,
    compute_uniformity_loss,
    detect_collapse,
    extract_embeddings_from_model,
    load_embeddings_from_file,
    save_embeddings_to_file,
)
from .sampling import (
    SelectionStrategy,
    SentinelConfig,
    SentinelSlot,
    StratificationCriterion,
    compute_missingness,
    default_age_criterion,
    default_los_criterion,
    default_mortality_criterion,
    default_unit_criterion,
    get_default_criteria,
    get_default_sentinel_slots,
    select_by_missingness,
    select_extreme_stays,
    select_sentinel_patients,
)
from .snapshots import (
    LegacyPipelineStage,
    PipelineSnapshot,
    SnapshotConfig,
    SnapshotMixin,
    capture_dense_snapshot,
    capture_labels_snapshot,
    capture_stays_snapshot,
    create_snapshots_from_processed,
    export_all_snapshots,
    export_snapshot,
    flatten_dense_timeseries,
    unflatten_timeseries,
)
from .staged_snapshots import (
    ExtractionStage,  # Backwards compatibility alias
    MultiStageCapture,
    PatientStageCapture,
    PipelineStage,  # Canonical stage enum
    StageData,
    compute_stage_diff,
    filter_to_stay_ids,
    flatten_binned_to_long,
    generate_html_report,
)

# Plots are optional - import only what's available
try:
    from .plots import (  # noqa: F401
        generate_debug_report,
        plot_cosine_similarity_heatmap,
        plot_embedding_2d,
        plot_embedding_distribution,
        plot_feature_distributions,
        plot_missingness_heatmap,
        plot_patient_timeseries,
        plot_pca_variance,
    )

    _HAS_PLOTS = True
except ImportError:
    _HAS_PLOTS = False


__all__ = [
    # Sampling
    "SelectionStrategy",
    "SentinelConfig",
    "SentinelSlot",
    "StratificationCriterion",
    "compute_missingness",
    "default_age_criterion",
    "default_los_criterion",
    "default_mortality_criterion",
    "default_unit_criterion",
    "get_default_criteria",
    "get_default_sentinel_slots",
    "select_by_missingness",
    "select_extreme_stays",
    "select_sentinel_patients",
    # Snapshots
    "LegacyPipelineStage",
    "PipelineSnapshot",
    "SnapshotConfig",
    "SnapshotMixin",
    "capture_dense_snapshot",
    "capture_labels_snapshot",
    "capture_stays_snapshot",
    "create_snapshots_from_processed",
    "export_all_snapshots",
    "export_snapshot",
    "flatten_dense_timeseries",
    "unflatten_timeseries",
    # Staged Snapshots (multi-stage pipeline inspection)
    "ExtractionStage",  # Backwards compatibility alias for PipelineStage
    "MultiStageCapture",
    "PatientStageCapture",
    "PipelineStage",  # Canonical stage enum - use this one
    "StageData",
    "compute_stage_diff",
    "filter_to_stay_ids",
    "flatten_binned_to_long",
    "generate_html_report",
    # Embeddings
    "CollapseMetrics",
    "DimensionalityMetrics",
    "EmbeddingQualityReport",
    "EmbeddingStats",
    "analyze_embeddings",
    "compare_embeddings",
    "compute_effective_rank",
    "compute_embedding_stats",
    "compute_participation_ratio",
    "compute_pca_metrics",
    "compute_uniformity_loss",
    "detect_collapse",
    "extract_embeddings_from_model",
    "load_embeddings_from_file",
    "save_embeddings_to_file",
]

# Conditionally add plot functions to __all__
if _HAS_PLOTS:
    __all__.extend(
        [
            "generate_debug_report",
            "plot_cosine_similarity_heatmap",
            "plot_embedding_2d",
            "plot_embedding_distribution",
            "plot_feature_distributions",
            "plot_missingness_heatmap",
            "plot_patient_timeseries",
            "plot_pca_variance",
        ]
    )
