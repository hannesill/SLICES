"""Standard-library verification/rendering of anonymous checkpoint aggregates."""

import csv
import itertools
import json
import math
import statistics as st
from pathlib import Path

SEEDS = {42, 123, 456, 789, 1011}
METHODS = ("mae", "jepa", "contrastive")
TASKS = ("mortality_hospital", "los_remaining")


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Checkpoint statistic mismatch: {a} != {b}")


def check_checkpoint_results(root):
    directory = Path(root) / "results/checkpoint-sensitivity"
    jobs = [json.loads(p.read_text()) for p in sorted((directory / "jobs").glob("*.json"))]
    if len(jobs) != 15 or any("data_receipt" in j for j in jobs):
        raise ValueError("Invalid public checkpoint job projection")
    probes = [p for j in jobs for p in j["probes"]]
    keys = [(p["paradigm"], p["task"], p["seed"], p["variant"]) for p in probes]
    expected = set(itertools.product(METHODS, TASKS, SEEDS, ("final", "best_val")))
    if len(keys) != 60 or set(keys) != expected:
        raise ValueError("Checkpoint grid must contain exactly 60 distinct probes")
    keyed = dict(zip(keys, probes))
    exported = csv_rows(directory / "per_probe.csv")
    export_keys = [(r["paradigm"], r["task"], int(r["seed"]), r["variant"]) for r in exported]
    if len(export_keys) != 60 or set(export_keys) != expected:
        raise ValueError("Exported checkpoint probe grid mismatch")
    for key, row in zip(export_keys, exported):
        for field in (
            "test_primary",
            "validation_primary",
            "selected_epoch_one_based",
            "n_train",
            "n_val",
            "n_test",
        ):
            close(row[field], keyed[key][field])
        for field in ("checkpoint_sha256", "head_initialization_sha256"):
            if row[field] != keyed[key][field]:
                raise ValueError(f"Exported checkpoint identity mismatch: {field}")
    pairs = csv_rows(directory / "paired_differences.csv")
    pairkeys = [(p["paradigm"], p["task"], int(p["seed"])) for p in pairs]
    if len(pairkeys) != 30 or set(pairkeys) != set(itertools.product(METHODS, TASKS, SEEDS)):
        raise ValueError("Checkpoint paired grid incomplete or duplicated")
    for key, row in zip(pairkeys, pairs):
        a, b = (keyed[(*key, variant)] for variant in ("final", "best_val"))
        for field in (
            "n_train",
            "n_val",
            "n_test",
            "head_initialization_sha256",
            "batch_generator_seed",
        ):
            if a[field] != b[field]:
                raise ValueError(f"Unmatched checkpoint pair: {field}")
        for p in (a, b):
            if p["status"] != "complete" or not math.isfinite(p["test_primary"]):
                raise ValueError("Incomplete or nonfinite checkpoint probe")
            metric = "test/mae" if p["task"] == "los_remaining" else "test/auprc"
            if p["dataset"] != "miiv" or p["metric"] != metric or p["test_primary"] < 0:
                raise ValueError("Checkpoint task, dataset or metric mismatch")
            if metric == "test/auprc" and p["test_primary"] > 1:
                raise ValueError("Checkpoint AUPRC outside [0, 1]")
            values = [h["val_primary"] for h in p["history"]]
            if not values or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid validation history")
            best = min(values) if p["task"] == "los_remaining" else max(values)
            close(p["validation_primary"], best)
            if p["selected_epoch_one_based"] != values.index(best) + 1:
                raise ValueError("Validation checkpoint selection mismatch")
        close(row["final"], a["test_primary"])
        close(row["best_val"], b["test_primary"])
        close(row["best_minus_final"], b["test_primary"] - a["test_primary"])
    summaries = csv_rows(directory / "aggregate.csv")
    if len(summaries) != 6 or {(r["paradigm"], r["task"]) for r in summaries} != set(
        itertools.product(METHODS, TASKS)
    ):
        raise ValueError("Checkpoint aggregate grid incomplete")
    for summary in summaries:
        group = [
            r for r in pairs if (r["paradigm"], r["task"]) == (summary["paradigm"], summary["task"])
        ]
        if int(summary["n_pairs"]) != 5:
            raise ValueError("Checkpoint aggregate does not have five pairs")
        for prefix in ("final", "best_val", "best_minus_final"):
            values = [float(r[prefix]) for r in group]
            for suffix, computed in [
                ("mean", st.mean(values)),
                ("sd", st.stdev(values)),
                ("min", min(values)),
                ("max", max(values)),
            ]:
                close(summary[f"{prefix}_{suffix}"], computed)
    historical = csv_rows(directory / "frozen_final_comparison.csv")
    if len(historical) != 30 or {
        (r["paradigm"], r["task"], int(r["seed"])) for r in historical
    } != set(itertools.product(METHODS, TASKS, SEEDS)):
        raise ValueError("Missing historical comparison")
    for row in historical:
        key = (row["paradigm"], row["task"], int(row["seed"]), "final")
        close(row["test_primary"], keyed[key]["test_primary"])
        close(
            row["diagnostic_minus_frozen_final"],
            float(row["test_primary"]) - float(row["frozen_final_metric"]),
        )
    support = csv_rows(directory / "label_support.csv")
    if len(support) != 30 or {(r["task"], int(r["seed"]), r["split"]) for r in support} != set(
        itertools.product(TASKS, SEEDS, ("train", "val", "test"))
    ):
        raise ValueError("Missing labelled support rows")
    for row in support:
        if int(row["full_cohort_split_stays"]) != int(row["labelled_stays"]) + int(
            row["missing_label_stays"]
        ):
            raise ValueError("Label support arithmetic mismatch")
        field = {"train": "n_train", "val": "n_val", "test": "n_test"}[row["split"]]
        for method, variant in itertools.product(METHODS, ("final", "best_val")):
            close(
                row["labelled_stays"],
                keyed[(method, row["task"], int(row["seed"]), variant)][field],
            )
    ledger = json.loads((directory / "runtime_ledger.json").read_text())
    close(ledger["charged_seconds"], sum(j["seconds"] for j in ledger["jobs"]))
    return {
        "status": "pass",
        "probes": 60,
        "pairs": 30,
        "validation_selections": 60,
        "support_rows": 30,
        "worker_seconds": ledger["charged_seconds"],
        "private_data_and_weight_identity": "producer receipts; not reverified from public ZIP",
        "cells": summaries,
    }


def render_checkpoint(root, output):
    report = check_checkpoint_results(root)
    names = {
        "mae": "MAE",
        "jepa": "JEPA",
        "contrastive": "Contrastive",
        "mortality_hospital": "Hospital mortality",
        "los_remaining": "Remaining LOS",
    }
    lines = [
        r"\begin{tabular}{llrlll}",
        r"\toprule",
        r"Task & Encoder & $n$ & Rerun final & Best SSL val. & $\Delta$ mean [range] \\",
        r"\midrule",
    ]
    for task, method in itertools.product(TASKS, METHODS):
        row = next(r for r in report["cells"] if (r["task"], r["paradigm"]) == (task, method))
        values = [names[task], names[method], "5"]
        for prefix in ("final", "best_val"):
            values.append(f"{float(row[prefix+'_mean']):.4f} $\\pm$ {float(row[prefix+'_sd']):.4f}")
        values.append(
            f"{float(row['best_minus_final_mean']):+.4f} "
            f"[{float(row['best_minus_final_min']):+.4f}, "
            f"{float(row['best_minus_final_max']):+.4f}]"
        )
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (Path(output) / "checkpoint.tex").write_text("\n".join(lines) + "\n")
    return report
