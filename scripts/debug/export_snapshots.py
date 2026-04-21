#!/usr/bin/env python
"""Export CSV snapshots from a processed SLICES dataset.

Example usage:
    uv run python scripts/debug/export_snapshots.py
    uv run python scripts/debug/export_snapshots.py data.processed_dir=data/processed/eicu
    uv run python scripts/debug/export_snapshots.py stay_ids='[123,456]'
"""

from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig
from slices.debug.snapshots import create_snapshots_from_processed


def resolve_stay_ids(processed_dir: Path, cfg: DictConfig) -> list[int]:
    """Return explicit stay IDs or a small deterministic sample from static.parquet."""
    if cfg.get("stay_ids"):
        return [int(stay_id) for stay_id in cfg.stay_ids]

    static_path = processed_dir / "static.parquet"
    if not static_path.exists():
        raise FileNotFoundError(
            f"Cannot infer stay_ids because {static_path} does not exist. "
            "Pass stay_ids='[id1,id2]' or generate the processed dataset first."
        )

    n_stays = int(cfg.get("sentinel", {}).get("n_per_stratum", 3))
    static_df = pl.read_parquet(static_path, columns=["stay_id"]).head(max(n_stays, 1))
    return [int(stay_id) for stay_id in static_df["stay_id"].to_list()]


@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.data.processed_dir)
    output_dir = (
        Path(cfg.debug_output_dir)
        if cfg.get("debug_output_dir")
        else processed_dir / "debug_snapshots"
    )

    stay_ids = resolve_stay_ids(processed_dir, cfg)
    exported = create_snapshots_from_processed(
        processed_dir=processed_dir,
        stay_ids=stay_ids,
        output_dir=output_dir,
        include_labels=True,
        flatten_timeseries=bool(cfg.get("flatten_timeseries", True)),
    )

    print(f"Exported {len(exported)} snapshot artifacts to {output_dir}")
    for name, path in exported.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
