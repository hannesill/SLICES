"""Converged-probe tables (appendix table and main classification table)."""

import statistics as st
from collections import defaultdict

from audit import DATASETS, SEEDS, SSL

from rebuttal_tables.common import (
    ALTS,
    BASELINES,
    CLS,
    LABEL,
    SHORT,
    bold,
    f,
    header,
    index,
    load,
    mean,
    msd,
    near,
    paired,
    readme,
    same,
    say,
    sd,
    seeds,
    split_pair,
    table,
    wins,
)


def converged_probe(out, pkg, ref, docs):
    p1 = load(pkg / "p1_converged_probe_per_seed.csv")
    conv, frozen = {}, {}
    for d in DATASETS:
        for t in CLS:
            for m in SSL:
                key = dict(dataset=d, task=t, paradigm=m, probe="logreg_l2")
                conv[d, t, m] = seeds(p1, "converged/test_auprc", **key)
                frozen[d, t, m] = ref[d, t, "linear_probe", m]
                packaged = seeds(p1, "frozen_capped/test_auprc", **key)
                for s in SEEDS:
                    same(frozen[d, t, m][s], packaged[s], f"P1 frozen value {d} {t} {m} {s}", 1e-6)
    agg = index(
        load(pkg / "p1_converged_probe_aggregated.csv"), "dataset", "task", "probe", "paradigm"
    )
    deltas = index(
        load(pkg / "p1_paired_deltas_mae_vs_others.csv"),
        *("dataset", "task", "probe", "metric", "source", "comparison"),
    )
    ranking = index(load(pkg / "p1_objective_ranking.csv"), "dataset", "task", "probe", "metric")
    cls_rows = readme(docs, "Dataset", "Task", "MAE conv.")
    delta_rows = readme(docs, "Dataset", "Task", "MAE - contrastive")
    for d, t, m in conv:
        a = agg[d, t, "logreg_l2", m]
        same(mean(conv[d, t, m]), float(a["converged/test_auprc_mean"]), "P1 mean")
        same(sd(conv[d, t, m]), float(a["converged/test_auprc_sd"]), "P1 SD")
        same(mean(frozen[d, t, m]), float(a["frozen_capped/test_auprc_mean"]), "P1 ref", 1e-6)
        same(sd(frozen[d, t, m]), float(a["frozen_capped/test_auprc_sd"]), "P1 ref SD", 1e-6)
        change = st.mean(paired(conv[d, t, m], frozen[d, t, m]))
        same(change, float(a["delta_converged_minus_frozen/test_auprc_mean"]), "P1 change", 1e-6)
        row = next(r for r in cls_rows if r[:2] == [d, t])
        col = {"mae": 2, "jepa": 4, "contrastive": 6}[m]
        for source, offset in ((conv, 0), (frozen, 1)):
            mu, sigma = split_pair(row[col + offset])
            near(mean(source[d, t, m]), mu, f"README P1 {d} {t} {m}")
            near(sd(source[d, t, m]), sigma, f"README P1 SD {d} {t} {m}")
    for d in DATASETS:
        for t in CLS:
            row = next(r for r in delta_rows if r[:2] == [d, t])
            for alt, col in (("contrastive", 2), ("jepa", 3)):
                for source, name, text in zip(
                    (conv, frozen),
                    ("converged", "frozen_capped"),
                    split_pair(row[col]),
                    strict=True,
                ):
                    diff = paired(source[d, t, "mae"], source[d, t, alt])
                    packaged = deltas[d, t, "logreg_l2", "auprc", name, "mae_minus_" + alt]
                    same(st.mean(diff), float(packaged["delta_mean"]), "P1 paired", 1e-6)
                    same(st.stdev(diff), float(packaged["delta_sd"]), "P1 paired SD", 1e-6)
                    if wins(source[d, t, "mae"], source[d, t, alt]) != int(
                        packaged["n_seeds_mae_better"]
                    ):
                        raise ValueError("P1 paired seed count mismatch")
                    near(st.mean(diff), text, f"README P1 delta {d} {t} {alt}")
            for source, name in ((conv, "converged"), (frozen, "frozen_capped")):
                best = max(SSL, key=lambda m: mean(source[d, t, m]))
                if ranking[d, t, "logreg_l2", "auprc"]["best_" + name] != best:
                    raise ValueError("P1 objective ranking mismatch")

    lines, spans, alternatives = [], {"conv.": [], "ref.": []}, defaultdict(list)
    for d in DATASETS:
        for t in CLS:
            means = {m: mean(conv[d, t, m]) for m in SSL}
            cells = [bold(msd(conv[d, t, m]), means[m] == max(means.values())) for m in SSL]
            for name, source in (("conv.", conv), ("ref.", frozen)):
                # The stronger alternative is chosen by its mean under the same probe.
                alt = max(ALTS, key=lambda m: mean(source[d, t, m]))
                delta = st.mean(paired(source[d, t, "mae"], source[d, t, alt]))
                spans[name].append(delta)
                cells.append(f(delta))
                alternatives[f"{LABEL[d]}/{LABEL[t]}"].append(LABEL[alt])
            lines.append([LABEL[d], LABEL[t], *cells])
    table(
        out / "converged-probe.tex",
        header(
            "Data", "Endpoint", "MAE", "JEPA", "Contrastive", r"$\Delta$ conv.", r"$\Delta$ cap."
        ),
        lines,
        "llrrrrr",
    )

    name = "converged-probe"
    changes = [(t, m, x) for (d, t, m) in conv for x in paired(conv[d, t, m], frozen[d, t, m])]
    x = [c for _, _, c in changes]
    say(name, f"converged > reference capped probe in {sum(c > 0 for c in x)}/{len(x)} runs")
    say(name, f"mean change {st.mean(x):+.3f} AUPRC (range {min(x):+.3f} to {max(x):+.3f})")
    for t in CLS:
        parts = [
            f"{LABEL[m]} {st.mean(c for tt, mm, c in changes if (tt, mm) == (t, m)):+.3f}"
            for m in SSL
        ]
        say(name, f"  mean change, {LABEL[t]}: " + ", ".join(parts))
    for label, source in (("converged", conv), ("reference", frozen)):
        first = sum(
            max(SSL, key=lambda m: mean(source[d, t, m])) == "mae" for d in DATASETS for t in CLS
        )
        beats = {
            alt: sum(wins(source[d, t, "mae"], source[d, t, alt]) for d in DATASETS for t in CLS)
            for alt in ALTS
        }
        say(
            name,
            f"{label}: MAE first by mean in {first}/9 cells; seed-paired MAE > JEPA "
            f"{beats['jepa']}/45, MAE > Contrastive {beats['contrastive']}/45, "
            f"MAE beats both {sum(beats.values())}/90",
        )
    for key, values in spans.items():
        say(
            name,
            f"Delta {key} (MAE minus stronger alternative) {min(values):.3f}-{max(values):.3f}",
        )
    say(
        name,
        "stronger alternative (conv., ref.): "
        + "; ".join(f"{cell} {'/'.join(alts)}" for cell, alts in alternatives.items()),
    )


def classification_main(out, ref, p1_conv, analysis):
    """Means only: capped probe, converged probe, and fine-tuned SSL beside the
    full-label supervised baselines (supervised Transformer, GRU-D, XGBoost)."""
    core = {(r["dataset"], r["task"], r["protocol"], r["paradigm"]): r for r in analysis["core"]}
    lines = []
    for d in DATASETS:
        for t in CLS:
            cells = []
            for source in (
                {m: ref[d, t, "linear_probe", m] for m in SSL},
                {m: p1_conv[d, t, m, "logreg_l2"] for m in SSL},
                {m: ref[d, t, "full_finetune", m] for m in SSL + BASELINES},
            ):
                if any(set(v) != SEEDS for v in source.values()):
                    raise ValueError(f"Missing seeds in {d} {t}")
                means = {m: mean(v) for m, v in source.items()}
                # Bold the best mean in each block and every mean that ties it at three decimals.
                top = f(max(means.values()))
                cells += [bold(f(means[m]), f(means[m]) == top) for m in means]
            for p in ("linear_probe", "full_finetune"):
                for m in SSL + (("supervised",) if p == "full_finetune" else ()):
                    # Same means, hence same rounding, as classification.tex.
                    if mean(ref[d, t, p, m]) != core[d, t, p, m]["mean"]:
                        raise ValueError("Reference mean differs from the audit analysis")
            lines.append([LABEL[d], LABEL[t], *cells])
    table(
        out / "classification-main.tex",
        [
            r"Data & Endpoint & \multicolumn{3}{c}{Probe, capped} & "
            r"\multicolumn{3}{c}{Probe, converged} & \multicolumn{3}{c}{Fine-tune} & "
            r"\multicolumn{3}{c}{Supervised baselines} \\",
            r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
            " & & "
            + " & ".join(SHORT[m] for m in SSL + SSL + SSL)
            + r" & Transf. & GRU-D & XGBoost \\",
        ],
        lines,
        "@{}ll" + "r" * 12 + "@{}",
        pre=[r"\setlength{\tabcolsep}{3.6pt}"],
    )
