#!/usr/bin/env python3
# ruff: noqa: E402 -- establish the hard timeout before importing numerical libraries
"""Post-hoc CPU raw/mask linear control, MIMIC-IV hospital mortality only.

No patient-level outputs; no writes to input data/cache; hard 15 minute timeout.
Fixed SGD logistic loss, alpha=1e-4, <=20 epochs, validation AP selection.
This is not a trainer-matched substitute for the frozen SSL probes.
"""

import signal

signal.alarm(900)

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import polars as pl
import sklearn
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

from slices.data.splits import _split_patient_sets
from slices.data.tensor_preprocessing import (
    apply_normalization_and_imputation,
    compute_normalization_stats,
    extract_tensors_from_dataframe,
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.out.exists():
        raise ValueError("New analysis needs a fresh output directory")
    a.out.mkdir(parents=True)
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    frame = pl.read_parquet(a.data / "timeseries.parquet").sort("stay_id")
    stay_ids = frame["stay_id"].to_list()
    static = pl.read_parquet(a.data / "static.parquet")
    patients = dict(zip(static["stay_id"].to_list(), static["patient_id"].to_list()))
    labels = pl.read_parquet(a.data / "labels.parquet")
    # Processed labels are long form in the benchmark export.
    if "task_name" in labels.columns:
        labels = labels.filter(pl.col("task_name") == "mortality_hospital")
        labelmap = dict(zip(labels["stay_id"].to_list(), labels["label"].to_list()))
    else:
        labelmap = dict(zip(labels["stay_id"].to_list(), labels["mortality_hospital"].to_list()))
    y = np.array([np.nan if labelmap.get(s) is None else labelmap[s] for s in stay_ids])
    valid = np.isfinite(y)
    x, mask = extract_tensors_from_dataframe(frame, 24, 84)
    del frame
    mask_features = mask.numpy().reshape(len(stay_ids), -1).astype(np.float32)
    provenance = {
        "scope": "post-hoc MIMIC-IV hospital mortality; full-label raw/mask SGD logistic controls",
        "source_files": {
            n: sha(a.data / n)
            for n in ("timeseries.parquet", "static.parquet", "labels.parquet", "metadata.yaml")
        },
        "script_sha256": sha(__file__),
        "code_base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
        ).stdout.strip()
        or None,
        "versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "cpu_threads": 2,
        "timeout_seconds": 900,
        "max_epochs": 20,
        "alpha": 1e-4,
        "checkpoint": "maximum validation average precision; test once after selection",
        "patient_outputs_saved": False,
        "results": [],
    }
    for seed in (42, 123, 456, 789, 1011):
        patient_sets = _split_patient_sets({patients[s] for s in stay_ids}, seed, 0.7, 0.15)
        sets = [
            np.array([i for i, s in enumerate(stay_ids) if patients[s] in group])
            for group in patient_sets
        ]
        splits = [idx[valid[idx]] for idx in sets]
        means, stds = compute_normalization_stats(x, mask, sets[0].tolist(), 84, True)
        norm = (
            apply_normalization_and_imputation(x, means, stds, True, 84).numpy().reshape(len(x), -1)
        )
        for name, features in [
            ("mask_only", mask_features),
            ("values_and_mask", np.concatenate((norm, mask_features), axis=1)),
        ]:
            fit_start = time.monotonic()
            train, val, test = splits
            xt = np.ascontiguousarray(features[train])
            xv = np.ascontiguousarray(features[val])
            yt = y[train].astype(int)
            counts = np.bincount(yt, minlength=2)
            weights = np.sqrt(len(yt) / (2 * counts))[yt].astype(np.float32)
            model = SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                random_state=seed,
                learning_rate="optimal",
                shuffle=True,
                average=False,
            )
            best_score = -1
            for epoch in range(20):
                model.partial_fit(xt, yt, classes=np.array([0, 1]), sample_weight=weights)
                score = average_precision_score(y[val], model.decision_function(xv))
                if score > best_score:
                    best_score = score
                    best = copy.deepcopy(model)
                    best_epoch = epoch + 1
            test_score = average_precision_score(y[test], best.decision_function(features[test]))
            row = dict(
                dataset="miiv",
                task="mortality_hospital",
                seed=seed,
                representation=name,
                n_train=len(train),
                n_val=len(val),
                n_test=len(test),
                n_test_positive=int(y[test].sum()),
                test_auprc=float(test_score),
                val_auprc=float(best_score),
                best_epoch=best_epoch,
                seconds=time.monotonic() - fit_start,
            )
            provenance["results"].append(row)
            provenance["elapsed_seconds"] = time.monotonic() - start
            (a.out / "results.json").write_text(json.dumps(provenance, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        del norm
    provenance["complete"] = True
    provenance["elapsed_seconds"] = time.monotonic() - start
    (a.out / "results.json").write_text(json.dumps(provenance, indent=2) + "\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
