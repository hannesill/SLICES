"""Label-fraction, cross-database context and reduced-label positives tables."""

import statistics as st
from collections import defaultdict

from audit import DATASETS, SEEDS, TASKS, metric

from rebuttal_tables.common import (
    CLS,
    FRACTIONS,
    LABEL,
    LOS,
    bold,
    count,
    f,
    header,
    index,
    load,
    mean,
    paired,
    same,
    say,
    seeds,
    table,
    wins,
)


def label_fraction_delta(out, rows, analysis):
    reduced = [
        r
        for r in rows
        if r["experiment_class"] == "label_efficiency"
        and r["label_fraction"] < 1
        and r["model_size"] == "default"
        and r["protocol"] == "full_finetune"
    ]
    core = [
        r
        for r in rows
        if r["experiment_class"] == "core_ssl_benchmark" and r["protocol"] == "full_finetune"
    ]
    reference = {
        (r["dataset"], r["task"], r["label_fraction"]): r["mean"]
        for r in analysis["label_vs_supervised"]
        if r["paradigm"] == "mae"
    }
    reference.update(
        {
            (r["dataset"], r["task"], 1.0): r["mean"]
            for r in analysis["paired"]
            if r["protocol"] == "full_finetune" and r["comparator"] == "supervised"
        }
    )
    lines, seen = [], set()
    for d in DATASETS:
        for t in TASKS:
            cells = []
            for frac in FRACTIONS:
                source = core if frac == 1.0 else reduced
                pair = [
                    seeds(
                        source,
                        metric({"task": t}),
                        True,
                        dataset=d,
                        task=t,
                        label_fraction=frac,
                        paradigm=m,
                    )
                    for m in ("mae", "supervised")
                ]
                if pair[0] is None and pair[1] is None:
                    cells.append("--")
                    continue
                delta = st.mean(paired(*pair))
                same(delta, reference[d, t, frac], f"label-fraction delta {d} {t} {frac}")
                seen.add((d, t, frac))
                cells.append(f(delta))
            lines.append([LABEL[d], LABEL[t], *cells])
    if seen != set(reference):
        raise ValueError("Label-fraction grid differs from audit analysis")
    table(
        out / "label-fraction-delta.tex",
        header("Data", "Task", *(f"{round(100 * x)}\\%" for x in FRACTIONS)),
        lines,
        "llrrrrrr",
    )


def transfer_context(out, rows, ref, analysis):
    transfer = {
        (r["source"], r["target"], r["task"], r["paradigm"]): r for r in analysis["transfer"]
    }
    core = {(r["dataset"], r["task"], r["protocol"], r["paradigm"]): r for r in analysis["core"]}
    name = "transfer-context"
    lines = []
    for target, source in (("eicu", "miiv"), ("miiv", "eicu")):
        for t in TASKS:
            other = seeds(
                rows,
                metric({"task": t}),
                experiment_class="cross_dataset_transfer",
                source_dataset=source,
                dataset=target,
                task=t,
                paradigm="mae",
                protocol="full_finetune",
            )
            own = ref[target, t, "full_finetune", "mae"]
            sup = ref[target, t, "full_finetune", "supervised"]
            same(mean(other), transfer[source, target, t, "mae"]["mean"], "transfer mean")
            same(mean(own), core[target, t, "full_finetune", "mae"]["mean"], "core MAE mean")
            same(
                mean(sup),
                core[target, t, "full_finetune", "supervised"]["mean"],
                "core supervised mean",
            )
            higher = t != LOS
            values = [mean(other), mean(own), mean(sup)]
            best = (max if higher else min)(values)
            texts = [f(v) for v in values]
            if sorted(texts).count(f(best)) > 1:
                raise ValueError(f"Displayed tie for the best value: {target} {t}")
            lines.append(
                [LABEL[target], LABEL[t], "AUPRC" if higher else "Abs. error"]
                + [bold(x, v == best) for x, v in zip(texts, values, strict=True)]
            )
            say(
                name,
                f"{LABEL[target]} target, {LABEL[t]}: other-DB MAE better than target-DB MAE in "
                f"{wins(other, own, higher)}/5 seeds, better than supervised in "
                f"{wins(other, sup, higher)}/5; target-DB MAE better than supervised in "
                f"{wins(own, sup, higher)}/5",
            )
    table(
        out / "transfer-context.tex",
        header(
            "Target",
            "Task",
            "Metric",
            "Other-DB pretraining",
            "Target-DB pretraining",
            "Supervised",
        ),
        lines,
        "lllrrr",
    )


def low_label_positives(out, path, rows, pkg):
    name = "low-label-positives"
    if not path.is_file():
        say(name, f"WARNING: {path} not found; skipped low-label-positives.tex")
        return
    data = load(path)
    expected = {
        (r["dataset"], r["task"], r["label_fraction"])
        for r in rows
        if r["experiment_class"] == "label_efficiency"
        and r["label_fraction"] < 1
        and r["model_size"] == "default"
        and r["task"] in CLS
    }
    full = index(load(pkg / "p1_validity_split_label_parity.csv"), "dataset", "seed", "task")
    cells = defaultdict(dict)
    for r in data:
        frac = float(r["label_fraction"])
        if r["task"] not in CLS or frac >= 1:
            continue
        seed = int(r["seed"])
        labeled, positive = count(r["n_train_labeled"]), count(r["n_train_positive"])
        if seed in cells[r["dataset"], r["task"], frac]:
            raise ValueError(f"Duplicate positives row {r}")
        same(positive / labeled, float(r["train_prevalence"]), "train prevalence", 1e-6)
        whole = full[r["dataset"], str(seed), r["task"]]
        if labeled > count(whole["n_train"]) or positive > count(whole["n_train_positive"]):
            raise ValueError(f"Labeled subset exceeds the full training split: {r}")
        cells[r["dataset"], r["task"], frac][seed] = (labeled, positive)
    if set(cells) != expected or any(set(v) != SEEDS for v in cells.values()):
        raise ValueError("Low-label positives do not cover the label-efficiency grid")
    lines, fewest = [], []
    for d in DATASETS:
        for t in CLS:
            for frac in sorted(x for (dd, tt, x) in cells if (dd, tt) == (d, t)):
                c = cells[d, t, frac]
                positives = [c[s][1] for s in sorted(SEEDS)]
                lines.append(
                    [LABEL[d], LABEL[t], f"{round(100 * frac)}\\%"]
                    + [f"{v:,}" for v in positives]
                    + [f"{st.mean(positives):,.1f}", f"{st.mean(c[s][0] for s in SEEDS):,.1f}"]
                )
                fewest.append((min(positives), d, t, frac))
    table(
        out / "low-label-positives.tex",
        [
            r" & & & \multicolumn{6}{c}{Labeled training positives} & \\",
            r"\cmidrule(lr){4-9}",
            "Data & Task & Labels & "
            + " & ".join(map(str, sorted(SEEDS)))
            + r" & Mean & Labeled stays \\",
        ],
        lines,
        "lllrrrrrrr",
    )
    low = sorted(fewest)[:3]
    say(
        name,
        "fewest labeled training positives (min over seeds): "
        + "; ".join(f"{n} ({LABEL[d]}, {LABEL[t]}, {round(100 * x)}%)" for n, d, t, x in low),
    )
