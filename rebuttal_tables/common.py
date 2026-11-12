"""Shared constants and helpers for the rebuttal table renderer."""

import csv
import statistics as st
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from audit import SEEDS, TASKS

PKG = Path("results/ml4h-rebuttal-v1")
POSITIVES = Path("results/camera-ready-v1/low_label_train_positives.csv")
CLS = TASKS[:3]
LOS = "los_remaining"
ALTS = ("jepa", "contrastive")
PORTIONS = ("eicu", "miiv")
FRACTIONS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
# Display names as in render.py (not imported: render.py needs matplotlib).
LABEL = {
    "combined": "Combined",
    "miiv": "MIMIC-IV",
    "eicu": "eICU-CRD",
    "mae": "MAE",
    "jepa": "JEPA",
    "contrastive": "Contrastive",
    "supervised": "Supervised",
    "gru_d": "GRU-D",
    "xgboost": "XGBoost",
    "linear_probe": "Probe",
    "full_finetune": "Fine-tune",
    "mortality_24h": "24h mortality",
    "mortality_hospital": "Hospital mortality",
    "aki_kdigo": "AKI",
    "los_remaining": "Remaining LOS",
}
SHORT = {"mae": "MAE", "jepa": "JEPA", "contrastive": "Con.", "supervised": "Sup."}
SHORT.update(gru_d="GRU-D", xgboost="XGBoost")
# Full-label supervised baselines shown beside fine-tuned SSL in Tables 1 and 2.
BASELINES = ("supervised", "gru_d", "xgboost")
# Names used in the package README tables.
README_NAME = {"mae": "MAE", "jepa": "JEPA", "contrastive": "Contrastive", "miiv": "MIMIC-IV"}
README_NAME.update(eicu="eICU")


# ---------------------------------------------------------------- helpers


def load(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows, *keys):
    out = {}
    for r in rows:
        key = tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]
        if key in out:
            raise ValueError(f"Duplicate aggregate row {key}")
        out[key] = r
    return out


def seeds(rows, value, optional=False, **match):
    """{seed: value} over rows matching every key; requires exactly the five seeds."""
    out = {}
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            seed = int(r["seed"])
            if seed in out:
                raise ValueError(f"Duplicate seed {seed} for {match}")
            out[seed] = float(r[value])
    if optional and not out:
        return None
    if set(out) != SEEDS:
        raise ValueError(f"Seed set {sorted(out)} for {match}")
    return out


def count(text):
    value = float(text)
    if value != int(value) or value < 0:
        raise ValueError(f"Not a count: {text}")
    return int(value)


def mean(v):
    return st.mean(v.values())


def sd(v):
    return st.stdev(v.values())


def paired(a, b):
    return [a[s] - b[s] for s in sorted(SEEDS)]


def wins(a, b, higher=True):
    return sum((x > 0) if higher else (x < 0) for x in paired(a, b))


def f(value, digits=3):
    return f"{value:.{digits}f}"


def msd(v, digits=3):
    return f"{f(mean(v), digits)} ({f(sd(v), digits)})"


def bold(text, flag):
    return r"\textbf{" + text + "}" if flag else text


def same(a, b, what, tol=1e-9):
    if not abs(a - b) <= tol:
        raise ValueError(f"{what}: recomputed {a!r}, packaged {b!r}")


def table(path, head, rows, align, pre=()):
    """Bare booktabs tabular; `head` holds raw header lines, a row is cells or a raw line.

    `pre` holds lines placed before the tabular (a local \\tabcolsep for the widest tables).
    """
    lines = [*pre, r"\begin{tabular}{" + align + "}", r"\toprule", *head, r"\midrule"]
    lines += [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in rows]
    path.write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def header(*cols):
    return [" & ".join(cols) + r" \\"]


def readme_tables(path):
    tables, block = [], []
    for line in path.read_text().splitlines() + [""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip()[1:-1].split("|")])
        elif block:
            tables.append((tuple(block[0]), block[2:]))
            block = []
    return tables


def readme(tables, *head, nth=0):
    hits = [rows for h, rows in tables if h[: len(head)] == head]
    if len(hits) <= nth:
        raise ValueError(f"README table {head} not found")
    return hits[nth]


def half_up(value, digits):
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP)


def near(value, text, what):
    """`value` agrees with a README number up to that number's display rounding.

    A few README cells were rounded twice (0.44747 -> 0.4475 -> 0.448); those are accepted
    and reported, since the aggregate CSVs that hold the exact values are checked separately.
    """
    t = text.strip().removesuffix(" h").replace("+", "")
    digits = len(t.split(".")[1]) if "." in t else 0
    if abs(value - float(t)) <= 0.5 * 10**-digits + 1e-9:
        return
    if half_up(float(half_up(value, digits + 1)), digits) == Decimal(t):
        say("readme", f"{what}: README shows {text}, value {value:.5f} (rounded twice)")
        return
    raise ValueError(f"{what}: recomputed {value:.6f}, README {text}")


def split_pair(text):
    a, b = text.split(" (")
    return a, b.rstrip(")")


def say(name, text):
    print(f"[{name}] {text}")


# ---------------------------------------------------------------- 1. converged probe
