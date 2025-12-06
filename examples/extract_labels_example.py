"""Example script demonstrating task label extraction.

This shows how to use the task configuration system to extract
mortality prediction labels from MIMIC-IV.

Minimal version for end-to-end testing with mortality tasks only.
"""

import yaml
from pathlib import Path

from slices.data.extractors.mimic_iv import MIMICIVExtractor, ExtractorConfig
from slices.data.tasks import TaskConfig


def load_task_config(task_name: str) -> TaskConfig:
    """Load a task configuration from YAML file.
    
    Args:
        task_name: Name of the task (e.g., 'mortality_24h').
        
    Returns:
        TaskConfig instance.
    """
    config_path = Path(__file__).parent.parent / "configs" / "tasks" / f"{task_name}.yaml"
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return TaskConfig(**config_dict)


def main():
    """Extract mortality labels from MIMIC-IV (minimal end-to-end test)."""
    
    # -------------------------------------------------------------------------
    # 1. Configure extractor (point to your local MIMIC-IV Parquet files)
    # -------------------------------------------------------------------------
    extractor_config = ExtractorConfig(
        parquet_root="data/parquet/mimic-iv-demo",
        output_dir="data/processed/mimic-iv-demo",
    )
    
    extractor = MIMICIVExtractor(extractor_config)
    
    # -------------------------------------------------------------------------
    # 2. Get ICU stays to extract
    # -------------------------------------------------------------------------
    print("Extracting ICU stays...")
    stays = extractor.extract_stays()
    print(f"Found {len(stays)} ICU stays")
    
    # For this example, use first 1000 stays
    stay_ids = stays["stay_id"].to_list()[:1000]
    print(f"Extracting labels for {len(stay_ids)} stays")
    
    # -------------------------------------------------------------------------
    # 3. Load task configurations (mortality tasks only for minimal testing)
    # -------------------------------------------------------------------------
    task_names = [
        "mortality_24h",
        "mortality_48h",
        "mortality_hospital",
    ]
    
    print(f"\nLoading {len(task_names)} task configurations:")
    task_configs = []
    for task_name in task_names:
        print(f"  - {task_name}")
        task_configs.append(load_task_config(task_name))
    
    # -------------------------------------------------------------------------
    # 4. Extract labels for all tasks
    # -------------------------------------------------------------------------
    print("\nExtracting labels...")
    labels = extractor.extract_labels(stay_ids, task_configs)
    
    print(f"\nExtracted labels shape: {labels.shape}")
    print(f"Columns: {labels.columns}")
    
    # -------------------------------------------------------------------------
    # 5. Display summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LABEL STATISTICS")
    print("=" * 60)
    
    for task_name in task_names:
        if task_name in labels.columns:
            pos_count = labels[task_name].sum()
            total = len(labels)
            prevalence = (pos_count / total) * 100
            print(f"{task_name:25s}: {pos_count:5d}/{total:5d} ({prevalence:5.1f}%)")
    
    # -------------------------------------------------------------------------
    # 6. Save labels to Parquet
    # -------------------------------------------------------------------------
    output_dir = Path(extractor_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "labels.parquet"
    labels.write_parquet(output_path)
    print(f"\nSaved labels to: {output_path}")
    
    # -------------------------------------------------------------------------
    # 7. Display sample
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAMPLE LABELS (first 10 stays)")
    print("=" * 60)
    print(labels.head(10))


if __name__ == "__main__":
    main()
