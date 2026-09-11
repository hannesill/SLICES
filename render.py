#!/usr/bin/env python3
"""Recreate paper tables and vector figures from anonymous run aggregates."""

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit import DATASETS, SSL, TASKS, analyze, read_csv
from matplotlib.patches import FancyBboxPatch

LABEL = {
    "combined": "Combined",
    "miiv": "MIMIC-IV",
    "eicu": "eICU-CRD",
    "mae": "MAE",
    "jepa": "JEPA",
    "contrastive": "Contrastive",
    "supervised": "Supervised",
    "linear_probe": "Probe",
    "full_finetune": "Fine-tune",
    "mortality_24h": "24h mortality",
    "mortality_hospital": "Hospital mortality",
    "aki_kdigo": "AKI",
    "los_remaining": "Remaining LOS",
}
COLORS = {"mae": "#0072B2", "jepa": "#D55E00", "contrastive": "#009E73", "supervised": "#444444"}


def table(path, headers, rows, align):
    text = "\\begin{tabular}{" + align + "}\n\\toprule\n"
    text += " & ".join(headers) + " \\\\\n\\midrule\n"
    text += "\n".join(" & ".join(row) + " \\\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    path.write_text(text)


def f(value):
    return f"{value:.3f}"


def figures(out, analysis, rows):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )

    def save(fig, name):
        fig.savefig(
            out / (name + ".pdf"),
            metadata={
                "Author": "",
                "Creator": "SLICES result renderer",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        fig.savefig(out / (name + ".png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 1.55))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    ax.set(xlim=(-0.12, 10.12), ylim=(0, 2))
    ax.axis("off")
    blocks = [
        (
            0,
            2.05,
            "Fixed specification",
            "24 h × 84 values + masks\nPatient splits; four tasks\nThree regimes; five seeds",
        ),
        (
            2.65,
            2.05,
            "Objective interface",
            "forward(values, mask)\n→ loss + diagnostics\nMAE · JEPA · contrastive",
        ),
        (
            5.3,
            2.05,
            "Encoder transfer",
            "Save encoder + config\nProbe / fine-tune heads\nValidation-selected tests",
        ),
        (
            7.95,
            2.05,
            "Auditable results",
            "Run-level CSV + hashes\nPair seeds; reject drift\nRebuild tables and figures",
        ),
    ]
    for x, w, title, body in blocks:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.5),
                w,
                1.38,
                boxstyle="round,pad=0.06",
                facecolor="#f4f7fa",
                edgecolor="#567086",
                lw=0.8,
            )
        )
        ax.text(x + w / 2, 1.66, title, ha="center", va="center", weight="bold", fontsize=8)
        ax.text(x + w / 2, 1.1, body, ha="center", va="center", linespacing=1.55, fontsize=7.4)
        if x < 7:
            ax.annotate(
                "",
                xy=(x + w + 0.5, 1.18),
                xytext=(x + w + 0.08, 1.18),
                arrowprops={"arrowstyle": "->", "lw": 1, "color": "#567086"},
            )
    ax.text(
        5,
        0.13,
        "New objective: implement → register → synthetic smoke → "
        "compare declaration → new result revision",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    save(fig, "benchmark-overview")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), sharey=True)
    for ax, d in zip(axes, DATASETS):
        for m, marker in zip(SSL + ("supervised",), ("o", "s", "^", "D")):
            selected = [
                r
                for r in rows
                if r["dataset"] == d
                and r["task"] == "mortality_hospital"
                and r["protocol"] == "full_finetune"
                and r["paradigm"] == m
                and r["experiment_class"] in {"label_efficiency", "core_ssl_benchmark"}
            ]
            fractions = sorted(set(r["label_fraction"] for r in selected))
            means = []
            errors = []
            for fraction in fractions:
                vals = [r["test/auprc"] for r in selected if r["label_fraction"] == fraction]
                if len(vals) != 5:
                    raise ValueError("Duplicate label curve source")
                means.append(st.mean(vals))
                errors.append(st.stdev(vals))
            ax.errorbar(
                [v * 100 for v in fractions],
                means,
                yerr=errors,
                marker=marker,
                markersize=3,
                linewidth=1,
                capsize=1.5,
                color=COLORS[m],
                label=LABEL[m],
            )
        ax.set_title(LABEL[d])
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 50, 100])
        ax.set_xticklabels(["1", "5", "10", "25", "50", "100"], fontsize=6.5)
        ax.set_xlabel("Labeled training stays (%)")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Hospital-mortality AUPRC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(w_pad=1)
    save(fig, "label-efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
    cells = [(d, t) for d in DATASETS for t in TASKS[:3]]
    for ax, p in zip(axes, ("linear_probe", "full_finetune")):
        for mi, m in enumerate(("jepa", "contrastive", "supervised")):
            selected = [
                r
                for r in analysis["paired"]
                if r["protocol"] == p and r["comparator"] == m and r["task"] != "los_remaining"
            ]
            for r in selected:
                y = cells.index((r["dataset"], r["task"])) + (mi - 1) * 0.19
                ax.plot([r["minimum"], r["maximum"]], [y, y], color=COLORS[m], lw=1)
                ax.scatter(
                    r["mean"],
                    y,
                    color=COLORS[m],
                    s=15,
                    marker=("s", "^", "D")[mi],
                    label=LABEL[m] if (r["dataset"], r["task"]) == cells[0] else None,
                )
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_title(LABEL[p])
        ax.grid(axis="x", alpha=0.15)
        ax.set_xlabel("MAE − comparator AUPRC")
        ax.set_yticks(range(len(cells)))
    axes[0].set_yticklabels(
        [LABEL[d] + " / " + LABEL[t].replace("mortality", "mort.") for d, t in cells], fontsize=7
    )
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.07),
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(w_pad=1)
    save(fig, "seed-variance")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("paper-results"))
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(a.root / "results/per_seed.csv")
    result = analyze(rows)
    lookup = {(r["dataset"], r["task"], r["protocol"], r["paradigm"]): r for r in result["core"]}

    def cell(d, t, p, m, best):
        r = lookup.get((d, t, p, m))
        if r is None:
            return "--"
        text = f(r["mean"]) + " (" + f(r["sd"]) + ")"
        return r"\textbf{" + text + "}" if r["mean"] == best else text

    lines = []
    for d in DATASETS:
        for p in ("full_finetune", "linear_probe"):
            for t in TASKS[:3]:
                best = max(
                    r["mean"] for (dd, tt, pp, _), r in lookup.items() if (dd, tt, pp) == (d, t, p)
                )
                lines.append(
                    [LABEL[d], LABEL[p], LABEL[t]]
                    + [cell(d, t, p, m, best) for m in SSL + ("supervised",)]
                )
    table(
        out / "classification.tex",
        ["Data", "Protocol", "Endpoint", "MAE", "JEPA", "Contrastive", "Supervised"],
        lines,
        "lllrrrr",
    )
    lines = []
    for d in DATASETS:
        constant = next(r for r in result["constant_los"] if r["dataset"] == d)
        for p in ("full_finetune", "linear_probe"):
            best = min(
                r["mean"]
                for (dd, tt, pp, _), r in lookup.items()
                if (dd, tt, pp) == (d, "los_remaining", p)
            )
            lines.append(
                [LABEL[d], LABEL[p]]
                + [cell(d, "los_remaining", p, m, best) for m in SSL + ("supervised",)]
                + [f(constant["mean"]) + " (" + f(constant["sd"]) + ")"]
            )
    table(
        out / "los.tex",
        ["Data", "Protocol", "MAE", "JEPA", "Contrastive", "Supervised", "Constant median"],
        lines,
        "llrrrrr",
    )
    lines = []
    for d in DATASETS:
        for t in TASKS:
            vals = []
            for m in ("gru_d", "xgboost", "supervised"):
                c = [
                    r
                    for r in rows
                    if r["dataset"] == d
                    and r["task"] == t
                    and r["paradigm"] == m
                    and r["label_fraction"] == 1
                    and r["model_size"] == "default"
                    and r["experiment_class"] in ("core_ssl_benchmark", "classical_baselines")
                ]
                vals.append(
                    st.mean(r["test/mae" if t == "los_remaining" else "test/auprc"] for r in c)
                )
            best_ssl = (min if t == "los_remaining" else max)(
                [lookup[d, t, "full_finetune", m] for m in SSL], key=lambda r: r["mean"]
            )
            vals.append(best_ssl["mean"])
            best = (min if t == "los_remaining" else max)(vals)
            lines.append(
                [LABEL[d], LABEL[t]]
                + [r"\textbf{" + f(v) + "}" if v == best else f(v) for v in vals]
                + [LABEL[best_ssl["paradigm"]]]
            )
    table(
        out / "baselines.tex",
        ["Data", "Endpoint", "GRU-D", "XGBoost", "Supervised", "Best SSL", "SSL method"],
        lines,
        "llrrrrl",
    )
    lines = []
    for r in result["hp"]:
        level = r["lr"] or r["mask_ratio"]
        lines.append(
            [r["subtype"].replace("_", " "), LABEL[r["paradigm"]], level, f(r["mean"]), f(r["sd"])]
        )
    table(out / "hp.tex", ["Sweep", "Objective", "Level", "AUPRC", "SD"], lines, "lllrr")
    lines = []
    for r in result["label_vs_supervised"]:
        if r["paradigm"] == "mae" and r["label_fraction"] == min(
            x["label_fraction"] for x in result["label_vs_supervised"] if x["task"] == r["task"]
        ):
            lines.append(
                [
                    LABEL[r["dataset"]],
                    LABEL[r["task"]],
                    str(int(100 * r["label_fraction"])) + r"\%",
                    f(r["ssl_mean"]),
                    f(r["supervised_mean"]),
                    f(r["mean"]),
                    f(r["sd"]),
                ]
            )
    table(
        out / "low-label.tex",
        ["Data", "Task", "Labels", "MAE", "Supervised", r"$\Delta$ mean", r"$\Delta$ SD"],
        lines,
        "lllrrrr",
    )
    lines = []
    for r in result["test_support"]:
        lines.append(
            [
                LABEL[r["dataset"]],
                LABEL[r["task"]],
                f"{r['n_samples']:,.1f}",
                "--" if r["n_positive"] is None else f"{r['n_positive']:,.1f}",
                "--" if r["positive_ratio"] is None else f"{r['positive_ratio']:.4f}",
            ]
        )
    table(
        out / "test-support.tex",
        ["Data", "Task", "Test samples", "Positive", "Prevalence"],
        lines,
        "llrrr",
    )
    lines = []
    for source, target in (("eicu", "miiv"), ("miiv", "eicu")):
        for task in TASKS:
            group = [
                r
                for r in result["transfer"]
                if (r["source"], r["target"], r["task"]) == (source, target, task)
            ]
            best = (min if task == "los_remaining" else max)(r["mean"] for r in group)
            cells = []
            for method in SSL:
                value = next(r["mean"] for r in group if r["paradigm"] == method)
                label = f"{value:.4f}" if target == "eicu" and task == "aki_kdigo" else f(value)
                cells.append(r"\textbf{" + label + "}" if value == best else label)
            lines.append(
                [
                    LABEL[source],
                    LABEL[target],
                    LABEL[task],
                    "Abs. error" if task == "los_remaining" else "AUPRC",
                    *cells,
                ]
            )
    table(
        out / "transfer.tex",
        ["Pretrain", "Target", "Task", "Metric", "MAE", "JEPA", "Contrastive"],
        lines,
        "llllrrr",
    )
    table(
        out / "ts2vec.tex",
        ["Protocol", "TS2Vec-style", "Core contrastive"],
        [
            [LABEL[r["protocol"]], str(r["wins"]["ts2vec"]), str(r["wins"]["contrastive"])]
            for r in result["ts2vec"]
        ],
        "lrr",
    )
    figures(out, result, rows)
    from checkpoint_report import render_checkpoint

    result["checkpoint_diagnostic"] = render_checkpoint(a.root, out)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Regenerated tables, figures, and report in {out}")


if __name__ == "__main__":
    main()
