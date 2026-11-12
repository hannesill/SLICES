"""Maintained-framework reproduction table."""

import json
import math
import statistics as st
from collections import defaultdict

from audit import SSL

from rebuttal_tables.common import (
    LABEL,
    LOS,
    f,
    header,
    load,
    readme,
    same,
    say,
    sd,
    table,
)


def reproduction(out, pkg, ref, docs):
    sets = {
        "P4b": load(pkg / "p4b_head_downstream_from_frozen_encoders.csv"),
        "P4a": load(pkg / "p4_head_reproduction_downstream.csv"),
    }
    summaries = {
        "P4b": json.loads(
            (pkg / "p4b_head_downstream_from_frozen_encoders_summary.json").read_text()
        ),
        "P4a": json.loads((pkg / "p4_head_reproduction_summary.json").read_text()),
    }
    doc = readme(docs, "Protocol", "Runs", "Within 1 seed sd")
    name = "reproduction"
    lines, primary = [], {}
    for key, start in (("P4b", "Archived encoders"), ("P4a", "Full pipeline")):
        prim = [r for r in sets[key] if r["is_primary_metric"] == "True"]
        for r in prim:
            if r["dataset"] != "miiv" or r["metric"] != ("mae" if r["task"] == LOS else "auprc"):
                raise ValueError("Unexpected reproduction row")
            frozen = ref["miiv", r["task"], r["protocol"], r["paradigm"]]
            r["delta"] = float(r["rerun"]) - frozen[int(r["seed"])]
            r["sd"] = sd(frozen)
            same(frozen[int(r["seed"])], float(r["frozen_same_seed"]), "P4 frozen value", 1e-6)
            same(r["sd"], float(r["frozen_seed_sd"]), "P4 frozen seed SD", 1e-6)
            same(r["delta"], float(r["diff"]), "P4 difference", 1e-6)
            for k in (1, 2):
                if (abs(r["delta"]) <= k * r["sd"]) != (r[f"within_{k}_sd"] == "True"):
                    raise ValueError("P4 within-SD flag mismatch")
        primary[key] = prim
        s = summaries[key]
        if (
            len(prim),
            sum(abs(r["delta"]) <= r["sd"] for r in prim),
            sum(abs(r["delta"]) <= 2 * r["sd"] for r in prim),
        ) != (s["n_downstream_runs"], s["n_primary_within_1_sd"], s["n_primary_within_2_sd"]):
            raise ValueError(f"{key} summary counts mismatch")
        same(
            max(abs(r["delta"]) for r in prim if r["task"] != LOS),
            s["primary_abs_diff_max_auprc"],
            "P4 max AUPRC",
            1e-6,
        )
        same(
            max(abs(r["delta"]) for r in prim if r["task"] == LOS),
            s["primary_abs_diff_max_los_mae"],
            "P4 max LOS",
            1e-6,
        )
        for p in ("linear_probe", "full_finetune"):
            rs = [r for r in prim if r["protocol"] == p]
            cells = []
            for group in ([r for r in rs if r["task"] != LOS], [r for r in rs if r["task"] == LOS]):
                diffs = [abs(r["delta"]) for r in group]
                if max(diffs) == 0:
                    cells.append("0 (0)")
                elif max(diffs) < 5e-4:
                    cells.append(f"$<10^{{{math.floor(math.log10(max(diffs))) + 1}}}$")
                else:
                    cells.append(f"{f(st.mean(diffs))} ({f(max(diffs))})")
            within = [sum(abs(r["delta"]) <= k * r["sd"] for r in rs) for k in (1, 2)]
            if key == "P4a":
                label = {"linear_probe": "Linear probe", "full_finetune": "Full fine-tune"}[p]
                row = next(x for x in doc if x[0] == label)
                if [int(row[1]), int(row[2]), int(row[3])] != [len(rs), *within]:
                    raise ValueError("README P4a counts mismatch")
            lines.append([start, LABEL[p], str(len(rs)), *cells, *map(str, within)])
            auprc = [r["delta"] for r in rs if r["task"] != LOS]
            los = [r["delta"] for r in rs if r["task"] == LOS]
            say(
                name,
                f"{key} {LABEL[p]}: {len(rs)} runs, within 1 SD {within[0]}, within 2 SD "
                f"{within[1]}; max |diff| AUPRC {max(map(abs, auprc)):.2g}, LOS "
                f"{max(map(abs, los)):.2g} days; mean signed diff AUPRC {st.mean(auprc):+.4f}, "
                f"LOS {st.mean(los):+.3f} days",
            )
    table(
        out / "reproduction.tex",
        header(
            "Starting point",
            "Protocol",
            "Runs",
            r"Mean $|\Delta|$ AUPRC (max)",
            r"Mean $|\Delta|$ LOS days (max)",
            r"$\leq$1 SD",
            r"$\leq$2 SD",
        ),
        lines,
        "llrrrrr",
    )
    for key, prim in primary.items():
        cells = defaultdict(list)
        for r in prim:
            cells[int(r["seed"]), r["task"], r["protocol"]].append(r)
        agree = defaultdict(int)
        total = defaultdict(int)
        for (seed, task, p), rs in cells.items():
            if sorted(r["paradigm"] for r in rs) != sorted(SSL):
                raise ValueError("Reproduction cell lacks an objective")
            higher = task != LOS
            rerun = sorted(rs, key=lambda r: float(r["rerun"]), reverse=higher)
            frozen = sorted(
                rs, key=lambda r: ref["miiv", task, p, r["paradigm"]][seed], reverse=higher
            )
            agree[p] += [r["paradigm"] for r in rerun] == [r["paradigm"] for r in frozen]
            total[p] += 1
        probe, finetune = (f"{agree[p]}/{total[p]}" for p in ("linear_probe", "full_finetune"))
        say(
            name,
            f"{key} objective order equals frozen same-seed order in {sum(agree.values())}/"
            f"{sum(total.values())} seed x task x protocol cells ({probe} probe, "
            f"{finetune} fine-tune)",
        )
        los_ft = [r["delta"] for r in prim if r["task"] == LOS and r["protocol"] == "full_finetune"]
        n_higher = sum(x > 0 for x in los_ft)
        say(name, f"{key} LOS fine-tune rerun higher than frozen in {n_higher}/{len(los_ft)}")
