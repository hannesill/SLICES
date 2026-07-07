#!/usr/bin/env python3
"""Audit best-validation checkpoint and final encoder exports.

The canonical source for the best-validation checkpoint is Lightning's
ModelCheckpoint callback state inside each exported ``checkpoints/last.ckpt``.
The final downstream encoder is ``encoder.pt``. It is saved after
``trainer.fit`` and does not currently store epoch metadata, so this script
does not infer final training duration from ``checkpoints/last.ckpt``.

Usage:
    uv run python scripts/diagnostics/analyze_encoder_epochs.py
    uv run python scripts/diagnostics/analyze_encoder_epochs.py --strict
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import torch

DEFAULT_ENCODER_ROOT = Path("outputs/archives/slices-thesis-encoders-thesis-v1")
DEFAULT_OUTPUT = Path("results/slices-thesis-encoder-audit-v1/encoder_epoch_audit.csv")
CSV_COLUMNS = [
    "run",
    "dataset",
    "paradigm",
    "seed",
    "retained_last_ckpt_epoch",
    "retained_last_ckpt_global_step",
    "best_epoch",
    "best_global_step",
    "retained_ckpt_epoch_gap",
    "monitor",
    "best_model_score",
    "current_score",
    "best_model_path",
    "retained_last_model_path",
    "best_ckpt_exists_local",
    "best_is_retained_last_ckpt",
    "final_encoder_equals_best_encoder",
    "final_encoder_equals_retained_last_ckpt",
    "retained_last_ckpt_equals_highest_saved_ssl_ckpt",
    "final_encoder_sha256",
    "best_encoder_sha256",
    "retained_last_ckpt_sha256",
    "highest_saved_ssl_ckpt_sha256",
    "n_best_k_models",
    "saved_ssl_ckpt_epochs",
    "status",
    "notes",
]
STATUS_OK = "ok"
STATUS_MISSING_LAST_CKPT = "missing_last_ckpt"
STATUS_MISSING_CALLBACK_METADATA = "missing_callback_metadata"
STATUS_MISSING_BEST_CKPT = "missing_best_ckpt"
STATUS_EPOCH_PARSE_FAILED = "epoch_parse_failed"
STATUS_ENCODER_COMPARE_FAILED = "encoder_compare_failed"
EPOCH_RE = re.compile(r"(?:^|[/\\])ssl-(\d+)\.ckpt$")


@dataclass
class CallbackMetadata:
    monitor: str
    best_model_path: str
    best_model_score: Any
    current_score: Any
    best_k_models: dict[Any, Any]
    last_model_path: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit best-validation versus last-epoch encoder exports.",
    )
    parser.add_argument(
        "--encoder-root",
        type=Path,
        default=DEFAULT_ENCODER_ROOT,
        help=f"Export root containing encoder_manifest.csv. Defaults to {DEFAULT_ENCODER_ROOT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any run cannot be audited cleanly.",
    )
    return parser.parse_args(argv)


def load_torch(path: Path) -> Any:
    """Load a Torch artifact on CPU across PyTorch default changes."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scalar_to_csv(value: Any) -> str:
    if value is None:
        return ""
    if torch.is_tensor(value):
        if value.numel() == 1:
            return f"{value.detach().cpu().item():.8g}"
        return str(value.detach().cpu().tolist())
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_to_csv(value: bool | None) -> str:
    if value is None:
        return ""
    return "True" if value else "False"


def parse_epoch_from_ckpt_path(path: str) -> int | None:
    match = EPOCH_RE.search(path)
    if not match:
        return None
    return int(match.group(1))


def resolve_local_ckpt_path(run_dir: Path, checkpoint_path: str) -> Path:
    return run_dir / "checkpoints" / Path(checkpoint_path).name


def extract_callback_metadata(checkpoint: dict[str, Any]) -> CallbackMetadata | None:
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return None

    candidates: list[dict[str, Any]] = []
    for callback_state in callbacks.values():
        if not isinstance(callback_state, dict):
            continue
        if "best_model_path" in callback_state or "monitor" in callback_state:
            candidates.append(callback_state)

    if not candidates:
        return None

    candidate = next(
        (state for state in candidates if state.get("monitor") in {"val/loss", "best_val_loss"}),
        candidates[0],
    )
    best_model_path = candidate.get("best_model_path")
    if not best_model_path:
        return None

    return CallbackMetadata(
        monitor=str(candidate.get("monitor", "")),
        best_model_path=str(best_model_path),
        best_model_score=candidate.get("best_model_score"),
        current_score=candidate.get("current_score"),
        best_k_models=candidate.get("best_k_models") or {},
        last_model_path=str(candidate.get("last_model_path", "")),
    )


def sorted_saved_epochs(row: dict[str, str], run_dir: Path) -> str:
    saved = row.get("saved_ssl_ckpt_epochs", "").strip()
    if saved:
        return saved

    epochs: list[int] = []
    for path in (run_dir / "checkpoints").glob("ssl-*.ckpt"):
        epoch = parse_epoch_from_ckpt_path(str(path))
        if epoch is not None:
            epochs.append(epoch)
    return ";".join(str(epoch) for epoch in sorted(epochs))


def extract_encoder_state(path: Path) -> dict[str, Any]:
    artifact = load_torch(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path.name} is not a dict")
    state = artifact.get("encoder_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path.name} has no encoder_state_dict")
    return state


def extract_full_ckpt_encoder_state(path: Path) -> dict[str, Any]:
    artifact = load_torch(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path.name} is not a dict")
    state = artifact.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path.name} has no state_dict")

    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"{path.name} has no encoder.* state_dict entries")
    return encoder_state


def state_dicts_equal(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str]:
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        missing_left = sorted(right_keys - left_keys)
        missing_right = sorted(left_keys - right_keys)
        details = []
        if missing_left:
            details.append(f"missing_in_encoder={missing_left[:3]}")
        if missing_right:
            details.append(f"missing_in_best={missing_right[:3]}")
        return False, ";".join(details)

    for key in sorted(left_keys):
        left_value = left[key]
        right_value = right[key]
        if torch.is_tensor(left_value) and torch.is_tensor(right_value):
            if left_value.shape != right_value.shape:
                return False, f"shape_mismatch={key}"
            if left_value.dtype != right_value.dtype:
                return False, f"dtype_mismatch={key}"
            if not torch.equal(left_value.cpu(), right_value.cpu()):
                return False, f"tensor_diff={key}"
        elif left_value != right_value:
            return False, f"value_diff={key}"

    return True, ""


def compare_final_and_best_encoder_exports(run_dir: Path) -> tuple[bool | None, str]:
    encoder_path = run_dir / "encoder.pt"
    best_encoder_path = run_dir / "encoder_best_val.pt"
    missing = [path.name for path in [encoder_path, best_encoder_path] if not path.exists()]
    if missing:
        return None, f"missing_encoder_exports={','.join(missing)}"

    try:
        encoder_state = extract_encoder_state(encoder_path)
        best_encoder_state = extract_encoder_state(best_encoder_path)
        return state_dicts_equal(encoder_state, best_encoder_state)
    except Exception as exc:  # noqa: BLE001 - diagnostic script records the failure.
        return None, f"encoder_compare_error={type(exc).__name__}:{exc}"


def compare_final_encoder_to_full_ckpt(run_dir: Path, ckpt_path: Path) -> tuple[bool | None, str]:
    encoder_path = run_dir / "encoder.pt"
    if not encoder_path.exists():
        return None, f"missing_encoder_export={encoder_path.name}"
    if not ckpt_path.exists():
        return None, f"missing_full_ckpt={ckpt_path.name}"

    try:
        encoder_state = extract_encoder_state(encoder_path)
        ckpt_encoder_state = extract_full_ckpt_encoder_state(ckpt_path)
        return state_dicts_equal(encoder_state, ckpt_encoder_state)
    except Exception as exc:  # noqa: BLE001 - diagnostic script records the failure.
        return None, f"encoder_ckpt_compare_error={type(exc).__name__}:{exc}"


def highest_saved_ssl_ckpt(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("ssl-*.ckpt"):
        epoch = parse_epoch_from_ckpt_path(str(path))
        if epoch is not None:
            candidates.append((epoch, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def base_row(manifest_row: dict[str, str]) -> dict[str, str]:
    return {column: "" for column in CSV_COLUMNS} | {
        "run": manifest_row.get("run", ""),
        "dataset": manifest_row.get("dataset", ""),
        "paradigm": manifest_row.get("paradigm", ""),
        "seed": manifest_row.get("seed", ""),
    }


def set_status(row: dict[str, str], status: str, note: str) -> None:
    if row["status"] in {"", STATUS_OK}:
        row["status"] = status
    if note:
        row["notes"] = ";".join(filter(None, [row["notes"], note]))


def audit_run(manifest_row: dict[str, str], encoder_root: Path) -> dict[str, str]:
    row = base_row(manifest_row)
    row["status"] = STATUS_OK

    run = row["run"]
    run_dir = encoder_root / run
    row["saved_ssl_ckpt_epochs"] = sorted_saved_epochs(manifest_row, run_dir)

    encoder_path = run_dir / "encoder.pt"
    best_encoder_path = run_dir / "encoder_best_val.pt"
    if encoder_path.exists():
        row["final_encoder_sha256"] = sha256_file(encoder_path)
    if best_encoder_path.exists():
        row["best_encoder_sha256"] = sha256_file(best_encoder_path)

    last_ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    if not last_ckpt_path.exists():
        set_status(row, STATUS_MISSING_LAST_CKPT, f"missing={last_ckpt_path}")
        return row

    try:
        last_ckpt = load_torch(last_ckpt_path)
    except Exception as exc:  # noqa: BLE001 - diagnostic script records the failure.
        set_status(row, STATUS_MISSING_LAST_CKPT, f"load_error={type(exc).__name__}:{exc}")
        return row

    if not isinstance(last_ckpt, dict):
        set_status(row, STATUS_MISSING_LAST_CKPT, "last_ckpt_not_dict")
        return row

    last_epoch = last_ckpt.get("epoch")
    last_global_step = last_ckpt.get("global_step")
    row["retained_last_ckpt_epoch"] = scalar_to_csv(last_epoch)
    row["retained_last_ckpt_global_step"] = scalar_to_csv(last_global_step)
    row["retained_last_ckpt_sha256"] = sha256_file(last_ckpt_path)

    highest_ckpt = highest_saved_ssl_ckpt(run_dir)
    if highest_ckpt is not None:
        row["highest_saved_ssl_ckpt_sha256"] = sha256_file(highest_ckpt)
        row["retained_last_ckpt_equals_highest_saved_ssl_ckpt"] = bool_to_csv(
            row["retained_last_ckpt_sha256"] == row["highest_saved_ssl_ckpt_sha256"]
        )

    metadata = extract_callback_metadata(last_ckpt)
    if metadata is None:
        set_status(row, STATUS_MISSING_CALLBACK_METADATA, "no_model_checkpoint_state")
    else:
        row["monitor"] = metadata.monitor
        row["best_model_path"] = metadata.best_model_path
        row["retained_last_model_path"] = metadata.last_model_path
        row["best_model_score"] = scalar_to_csv(metadata.best_model_score)
        row["current_score"] = scalar_to_csv(metadata.current_score)
        row["n_best_k_models"] = str(len(metadata.best_k_models))

        best_epoch = parse_epoch_from_ckpt_path(metadata.best_model_path)
        if best_epoch is None:
            set_status(
                row,
                STATUS_EPOCH_PARSE_FAILED,
                f"best_model_path={metadata.best_model_path}",
            )
        else:
            row["best_epoch"] = str(best_epoch)
            if isinstance(last_epoch, int):
                row["retained_ckpt_epoch_gap"] = str(last_epoch - best_epoch)
                row["best_is_retained_last_ckpt"] = bool_to_csv(best_epoch == last_epoch)

        best_ckpt_path = resolve_local_ckpt_path(run_dir, metadata.best_model_path)
        best_ckpt_exists = best_ckpt_path.exists()
        row["best_ckpt_exists_local"] = bool_to_csv(best_ckpt_exists)
        if not best_ckpt_exists:
            set_status(row, STATUS_MISSING_BEST_CKPT, f"missing={best_ckpt_path}")
        else:
            try:
                best_ckpt = load_torch(best_ckpt_path)
                if isinstance(best_ckpt, dict):
                    row["best_global_step"] = scalar_to_csv(best_ckpt.get("global_step"))
                else:
                    set_status(row, STATUS_MISSING_BEST_CKPT, "best_ckpt_not_dict")
            except Exception as exc:  # noqa: BLE001 - diagnostic script records the failure.
                set_status(
                    row,
                    STATUS_MISSING_BEST_CKPT,
                    f"best_load_error={type(exc).__name__}:{exc}",
                )

    equals, compare_note = compare_final_and_best_encoder_exports(run_dir)
    row["final_encoder_equals_best_encoder"] = bool_to_csv(equals)
    if equals is None:
        set_status(row, STATUS_ENCODER_COMPARE_FAILED, compare_note)
    elif compare_note:
        row["notes"] = ";".join(filter(None, [row["notes"], compare_note]))

    equals_retained, retained_note = compare_final_encoder_to_full_ckpt(run_dir, last_ckpt_path)
    row["final_encoder_equals_retained_last_ckpt"] = bool_to_csv(equals_retained)
    if equals_retained is None:
        set_status(row, STATUS_ENCODER_COMPARE_FAILED, retained_note)
    elif retained_note:
        row["notes"] = ";".join(filter(None, [row["notes"], retained_note]))

    return row


def read_manifest(encoder_root: Path) -> list[dict[str, str]]:
    manifest_path = encoder_root / "encoder_manifest.csv"
    with manifest_path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def percent(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def row_bool(row: dict[str, str], column: str) -> bool:
    return row.get(column) == "True"


def print_group_summary(rows: Sequence[dict[str, str]], keys: Sequence[str], title: str) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)

    print(f"\n{title}")
    header = " | ".join(
        [
            *keys,
            "n",
            "best_is_retained_last",
            "final_eq_best_encoder",
            "final_eq_retained_ckpt",
            "retained_gap_median",
        ]
    )
    print(header)
    print("-" * len(header))
    for group_key in sorted(groups):
        group_rows = groups[group_key]
        n = len(group_rows)
        best_last = sum(row_bool(row, "best_is_retained_last_ckpt") for row in group_rows)
        best_encoder_equal = sum(
            row_bool(row, "final_encoder_equals_best_encoder") for row in group_rows
        )
        retained_equal = sum(
            row_bool(row, "final_encoder_equals_retained_last_ckpt") for row in group_rows
        )
        gaps = [
            int(row["retained_ckpt_epoch_gap"])
            for row in group_rows
            if row.get("retained_ckpt_epoch_gap")
        ]
        gap_text = f"{median(gaps):.1f}" if gaps else ""
        print(
            " | ".join(
                [
                    *group_key,
                    str(n),
                    f"{best_last} ({percent(best_last, n)})",
                    f"{best_encoder_equal} ({percent(best_encoder_equal, n)})",
                    f"{retained_equal} ({percent(retained_equal, n)})",
                    gap_text,
                ]
            )
        )


def print_summary(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    total = len(rows)
    ok_rows = [row for row in rows if row["status"] == STATUS_OK]
    best_last = sum(row_bool(row, "best_is_retained_last_ckpt") for row in rows)
    best_encoder_equal = sum(row_bool(row, "final_encoder_equals_best_encoder") for row in rows)
    retained_equal = sum(row_bool(row, "final_encoder_equals_retained_last_ckpt") for row in rows)
    gaps = [
        int(row["retained_ckpt_epoch_gap"]) for row in rows if row.get("retained_ckpt_epoch_gap")
    ]
    statuses = Counter(row["status"] for row in rows)

    print("\nEncoder epoch audit")
    print(f"Output: {output_path}")
    print(f"Total runs: {total}")
    print(f"OK rows: {len(ok_rows)} ({percent(len(ok_rows), total)})")
    print(
        "Best epoch equals retained last checkpoint: " f"{best_last} ({percent(best_last, total)})"
    )
    print(
        "Final encoder equals best-val encoder: "
        f"{best_encoder_equal} ({percent(best_encoder_equal, total)})"
    )
    print(
        "Final encoder equals retained last.ckpt encoder: "
        f"{retained_equal} ({percent(retained_equal, total)})"
    )
    if gaps:
        print(
            "Retained-checkpoint epoch gap: "
            f"median={median(gaps):.1f}, min={min(gaps)}, max={max(gaps)}"
        )
    else:
        print("Retained-checkpoint epoch gap: no parsed values")

    if len(statuses) > 1 or STATUS_OK not in statuses:
        print("\nStatuses")
        for status, count in sorted(statuses.items()):
            print(f"{status}: {count}")

    print_group_summary(rows, ["paradigm"], "By paradigm")
    print_group_summary(rows, ["dataset"], "By dataset")
    print_group_summary(rows, ["paradigm", "dataset"], "By paradigm x dataset")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    encoder_root = args.encoder_root
    output_path = args.output or DEFAULT_OUTPUT

    manifest_rows = read_manifest(encoder_root)
    rows = [audit_run(row, encoder_root) for row in manifest_rows]
    write_csv(rows, output_path)
    print_summary(rows, output_path)

    bad_rows = [row for row in rows if row["status"] != STATUS_OK]
    if args.strict and bad_rows:
        examples = ", ".join(f"{row['run']}={row['status']}" for row in bad_rows[:5])
        raise RuntimeError(f"{len(bad_rows)} rows failed audit: {examples}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
