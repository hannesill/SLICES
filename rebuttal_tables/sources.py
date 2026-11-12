"""Per-source Combined table."""

import statistics as st

from audit import SEEDS, SSL, TASKS

from rebuttal_tables.common import (
    ALTS,
    LABEL,
    LOS,
    PORTIONS,
    README_NAME,
    SHORT,
    bold,
    f,
    index,
    load,
    mean,
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


def per_source(out, pkg, ref, p1_conv, docs):
    p3a = load(pkg / "p3_checkpoint_per_source_per_run.csv")
    p3b = load(pkg / "p3_local_converged_probe_per_source_per_seed.csv")
    if any(r["status"] != "ok" or r["checkpoint_sha256_verified"] != "True" for r in p3a):
        raise ValueError("P3a run without verified checkpoint")
    if {r["dataset"] for r in p3a + p3b} != {"combined"}:
        raise ValueError("P3 rows outside the Combined regime")
    agg_a = load(pkg / "p3_checkpoint_per_source_aggregated.csv")
    agg_a = index(agg_a, *("task", "paradigm", "protocol", "portion"))
    agg_b = load(pkg / "p3_local_converged_probe_per_source_aggregated.csv")
    agg_b = index(agg_b, *("task", "probe", "paradigm", "portion"))
    pd_b = load(pkg / "p3_local_converged_probe_per_source_paired_deltas.csv")
    pd_b = index(pd_b, *("task", "probe", "portion", "metric", "comparison"))
    # README P3 tables: fine-tune + supervised (classification, LOS) and converged probe.
    objectives = ("MAE", "JEPA", "Contrastive")
    cls_head = ("Task", "Portion", "n (mean)", "Prevalence")
    los_head = ("Portion", "n (mean)", "Label mean / median, days", "Metric")
    doc_ft = readme(docs, *cls_head, "Metric", *objectives, "Supervised")
    doc_los = readme(docs, *los_head, *objectives, "Supervised")
    doc_b = readme(docs, *cls_head, *objectives)
    finetuned = SSL + ("supervised",)
    name = "per-source"
    lines = []
    for t in TASKS:
        if t == LOS:
            lines.append(r"\midrule")
        col, kind = ("mae", "huber_valmae") if t == LOS else ("auprc", "logreg_l2")
        higher = t != LOS
        pick = max if higher else min
        for q in PORTIONS:
            probe = {m: seeds(p3b, col, task=t, portion=q, paradigm=m, probe=kind) for m in SSL}
            ft = {
                m: seeds(p3a, col, task=t, portion=q, paradigm=m, protocol="full_finetune")
                for m in finetuned
            }
            # Test support must be identical across runs of one seed and between both tables.
            support = {}
            for r in p3a + [r for r in p3b if r["probe"] == kind]:
                if (r["task"], r["portion"]) == (t, q):
                    key = int(r["seed"])
                    value = (r["n"], float(r["n_positive"]) if r["n_positive"] else None)
                    if support.setdefault(key, value) != value:
                        raise ValueError(f"P3 test support differs {t} {q} {key}")
            prevalence = None
            if higher:
                prevalence = st.mean(v[1] / int(v[0]) for v in support.values())
                same(
                    prevalence,
                    float(agg_a[t, "mae", "full_finetune", q]["prevalence_mean"]),
                    "P3 prevalence",
                )
            for m in SSL:
                b = agg_b[t, kind, m, q]
                same(mean(probe[m]), float(b[col + "_mean"]), "P3b mean")
                same(sd(probe[m]), float(b[col + "_sd"]), "P3b SD")
            for m in finetuned:
                a = agg_a[t, m, "full_finetune", q]
                same(mean(ft[m]), float(a[col + "_mean"]), "P3a mean")
                same(sd(ft[m]), float(a[col + "_sd"]), "P3a SD")
            for alt in ALTS:
                packaged = pd_b[t, kind, q, col, "mae_minus_" + alt]
                same(
                    st.mean(paired(probe["mae"], probe[alt])),
                    float(packaged["delta_mean"]),
                    "P3b paired",
                )
                if wins(probe["mae"], probe[alt], higher) != int(packaged["n_seeds_mae_better"]):
                    raise ValueError("P3b paired seed count mismatch")
            if higher:
                doc = next(r for r in doc_ft if r[:2] == [t, README_NAME[q]] and r[4] == "AUPRC")
                near(prevalence, doc[3], "README P3a prevalence")
                docb = next(r for r in doc_b if r[:2] == [t, README_NAME[q]])
                for i, m in enumerate(SSL):
                    mu, sigma = split_pair(docb[4 + i])
                    near(mean(probe[m]), mu, f"README P3b {t} {q} {m}")
                    near(sd(probe[m]), sigma, f"README P3b SD {t} {q} {m}")
                offset = 5
            else:
                doc = next(
                    r for r in doc_los if r[0] == README_NAME[q] and r[3].startswith("Abs. error")
                )
                offset = 4
            for i, m in enumerate(finetuned):
                mu, sigma = split_pair(doc[offset + i])
                near(mean(ft[m]), mu, f"README P3a {t} {q} {m}")
                near(sd(ft[m]), sigma, f"README P3a SD {t} {q} {m}")
            # The "all" portion reproduces the frozen overall values and the P1 probe.
            for m in finetuned:
                overall = seeds(
                    p3a, col, task=t, portion="all", paradigm=m, protocol="full_finetune"
                )
                for s in SEEDS:
                    same(
                        overall[s],
                        ref["combined", t, "full_finetune", m][s],
                        "P3a all vs per_seed",
                        1e-4,
                    )
            for m in SSL:
                overall = seeds(p3b, col, task=t, portion="all", paradigm=m, probe=kind)
                for s in SEEDS:
                    same(overall[s], p1_conv["combined", t, m, kind][s], "P3b all vs P1", 1e-6)

            best_probe = pick(mean(v) for v in probe.values())
            best_ft = pick(mean(v) for v in ft.values())
            lines.append(
                [LABEL[t], LABEL[q], "--" if prevalence is None else f(prevalence)]
                + [bold(f(mean(probe[m])), mean(probe[m]) == best_probe) for m in SSL]
                + [bold(f(mean(ft[m])), mean(ft[m]) == best_ft) for m in finetuned]
            )
            beats_probe = ", ".join(
                f"{LABEL[m]} {wins(probe['mae'], probe[m], higher)}/5" for m in ALTS
            )
            beats_ft = ", ".join(
                f"{LABEL[m]} {wins(ft['mae'], ft[m], higher)}/5" for m in ("supervised", *ALTS)
            )
            say(
                name,
                f"{LABEL[t]}, {LABEL[q]} portion ({'AUPRC' if higher else 'abs. error'}), seeds "
                f"where MAE is better: converged probe vs {beats_probe}; fine-tune vs {beats_ft}",
            )
            if not higher:
                say(
                    name,
                    f"  LOS Huber probe means, {LABEL[q]}: "
                    + ", ".join(
                        f"{LABEL[m]} {f(mean(probe[m]))}"
                        for m in sorted(SSL, key=lambda m: mean(probe[m]))
                    ),
                )
    table(
        out / "per-source.tex",
        [
            r"Task & Portion & Prev. & \multicolumn{3}{c}{Probe (converged)} & "
            r"\multicolumn{4}{c}{Fine-tune} \\",
            r"\cmidrule(lr){4-6} \cmidrule(lr){7-10}",
            " & & & " + " & ".join(SHORT[m] for m in SSL + finetuned) + r" \\",
        ],
        lines,
        "llrrrrrrrr",
    )
