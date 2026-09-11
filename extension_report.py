"""Report the frozen TS2Vec extension against matched core-contrastive runs.

Standard library only. No new experiments; no pooling across tasks or protocols.
"""

import argparse
import csv
import itertools
import json
import statistics as st
from pathlib import Path

from audit import (
    DATASETS,
    SEEDS,
    TASKS,
    check_rows,
    digest,
    metric,
    read_csv,
    summary,
    verify_manifest,
)


def extension_report(rows, configs, output):
    config_map = {c["scientific_id"]: c for c in configs}
    if len(config_map) != len(configs):
        raise ValueError("Duplicate scientific configuration identity")
    check_rows(rows)
    cells = []
    pairs = []
    for dataset, task, protocol in itertools.product(
        DATASETS, TASKS, ("linear_probe", "full_finetune")
    ):
        groups = {}
        for method, experiment in [
            ("ts2vec", "ts2vec_extension"),
            ("contrastive", "core_ssl_benchmark"),
        ]:
            group = [
                r
                for r in rows
                if (
                    r["dataset"],
                    r["task"],
                    r["protocol"],
                    r["paradigm"],
                    r["experiment_class"],
                    r["label_fraction"],
                    r["model_size"],
                )
                == (dataset, task, protocol, method, experiment, 1.0, "default")
            ]
            if len(group) != 5 or {r["seed"] for r in group} != SEEDS:
                raise ValueError("Extension comparison requires exactly five matched seeds")
            groups[method] = {r["seed"]: r for r in group}
        deltas = []
        verified_support = 0
        for seed in sorted(SEEDS):
            a, b = (groups[m][seed] for m in ("ts2vec", "contrastive"))
            for row in (a, b):
                cfg = config_map.get(row["scientific_id"])
                if cfg is None:
                    raise ValueError("Missing scientific configuration")
                for key in ("dataset", "task", "protocol", "paradigm", "seed"):
                    if cfg["scientific_cell"][key] != row[key]:
                        raise ValueError(f"Scientific configuration mismatch: {key}")
                if (
                    row["_eval_checkpoint_source"] != "best"
                    or str(row["_best_ckpt_load_ok"]).lower() != "true"
                ):
                    raise ValueError("Validation-selected test checkpoint required")
            support_fields = ["test/n_samples"]
            if task != "los_remaining":
                support_fields.append("test/n_positive")
            for key in support_fields:
                if a[key] is not None and b[key] is not None and a[key] != b[key]:
                    raise ValueError("Unmatched test support")
            support_known = all(a[key] is not None and b[key] is not None for key in support_fields)
            verified_support += int(support_known)
            delta = a[metric(a)] - b[metric(b)]
            deltas.append(delta)
            pairs.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "protocol": protocol,
                    "seed": seed,
                    "metric": metric(a),
                    "ts2vec_id": a["scientific_id"],
                    "contrastive_id": b["scientific_id"],
                    "ts2vec_minus_contrastive": delta,
                    "test_support": "matched" if support_known else "unknown",
                }
            )
        means = {m: summary([r[metric(r)] for r in g.values()]) for m, g in groups.items()}
        delta = st.mean(deltas)
        cells.append(
            {
                "dataset": dataset,
                "task": task,
                "protocol": protocol,
                "metric": metric(a),
                "n_pairs": 5,
                "test_support_pairs_verified": verified_support,
                "test_support_pairs_unknown": 5 - verified_support,
                **means,
                "paired_difference": summary(deltas),
                "winner": (
                    "tie"
                    if delta == 0
                    else (
                        "ts2vec"
                        if (delta < 0 if task == "los_remaining" else delta > 0)
                        else "contrastive"
                    )
                ),
            }
        )
    report = {
        "source": "frozen slices-thesis-v1; existing scientific results only",
        "evidence_policy": "descriptive; sample SD across five seeds; no significance claims",
        "checks": [
            "exact grid",
            "matched seeds",
            "reject differing test support when both counts exist; missing stays unknown",
            "configuration join",
            "successful best downstream checkpoint loading",
        ],
        "not_verified": [
            "private split identity",
            "test support for rows without exported counts",
            "normalization identity",
            "retraining equivalence",
        ],
        "cells": cells,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen-extension.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "frozen-pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return {
        "cells": len(cells),
        "pairs": len(pairs),
        "report_sha256": digest(output / "frozen-extension.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("extension-report"))
    args = parser.parse_args()
    verify_manifest(args.root)
    if args.out.exists():
        raise ValueError("Output exists; choose a fresh directory")
    print(
        json.dumps(
            extension_report(
                read_csv(args.root / "results/per_seed.csv"),
                json.loads((args.root / "results/scientific_configs.json").read_text()),
                args.out,
            )
        )
    )
