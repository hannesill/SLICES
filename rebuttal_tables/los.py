"""Loss-matched remaining-LOS probe table."""

import statistics as st

from audit import DATASETS, SEEDS, SSL

from rebuttal_tables.common import (
    ALTS,
    BASELINES,
    LABEL,
    LOS,
    bold,
    f,
    header,
    index,
    load,
    mean,
    msd,
    near,
    readme,
    same,
    say,
    sd,
    seeds,
    table,
    wins,
)


def los_matched(out, pkg, ref, analysis, docs):
    p1 = load(pkg / "p1_converged_probe_per_seed.csv")
    agg = index(load(pkg / "p1_los_loss_matched.csv"), "dataset", "paradigm")
    los_doc = readme(docs, "Dataset", "Objective", "Median MAE")
    name = "los-matched"
    lines = []
    for d in DATASETS:
        probe = {}
        for kind in ("huber_valmae", "ridge_valmse"):
            for m in SSL:
                key = dict(dataset=d, task=LOS, paradigm=m, probe=kind)
                probe[kind, m] = {
                    k: seeds(p1, "converged/test_" + k, **key) for k in ("mae", "mse")
                }
                consts = [
                    seeds(p1, "converged/const_train_median_test_mae", **key),
                    seeds(p1, "converged/const_train_mean_test_mse", **key),
                ]
                if (kind, m) == ("huber_valmae", "mae"):
                    median, mean_mse = consts
                if consts != [median, mean_mse]:
                    raise ValueError("LOS constants differ between probes of one dataset")
        constant = next(r for r in analysis["constant_los"] if r["dataset"] == d)
        same(mean(median), constant["mean"], f"LOS median constant {d}", 1e-5)
        same(sd(median), constant["sd"], f"LOS median constant SD {d}", 1e-5)
        if msd(median) != f(constant["mean"]) + " (" + f(constant["sd"]) + ")":
            raise ValueError("LOS constant display differs from los.tex")
        for m in SSL:
            a = agg[d, m]
            same(mean(median), float(a["const_train_median_test_mae_mean"]), "LOS const")
            same(mean(mean_mse), float(a["const_train_mean_test_mse_mean"]), "LOS const MSE")
            doc = next(r for r in los_doc if r[0] == d and r[1].lower() == m)
            near(mean(median), doc[2], "README median constant")
            near(mean(mean_mse), doc[3], "README mean constant")
            for kind, col in (("huber_valmae", 6), ("ridge_valmse", 5)):
                v = probe[kind, m]
                for k in ("mae", "mse"):
                    same(mean(v[k]), float(a[f"{kind}/test_{k}_mean"]), f"LOS {kind} {k}")
                    same(sd(v[k]), float(a[f"{kind}/test_{k}_sd"]), f"LOS {kind} {k} SD")
                skill_mae = st.mean(1 - v["mae"][s] / median[s] for s in SEEDS)
                skill_mse = st.mean(1 - v["mse"][s] / mean_mse[s] for s in SEEDS)
                beat_mae = sum(v["mae"][s] < median[s] for s in SEEDS)
                beat_mse = sum(v["mse"][s] < mean_mse[s] for s in SEEDS)
                same(skill_mae, float(a[f"{kind}/mae_skill_vs_median_mean"]), "LOS MAE skill")
                same(skill_mse, float(a[f"{kind}/mse_skill_vs_mean_mean"]), "LOS MSE skill")
                if beat_mae != int(a[f"{kind}/n_seeds_mae_beats_median"]) or beat_mse != int(
                    a[f"{kind}/n_seeds_mse_beats_mean"]
                ):
                    raise ValueError("LOS constant seed counts mismatch")
                doc_mae, doc_mse, _ = doc[col].split(" / ")
                near(mean(v["mae"]), doc_mae, f"README {kind} MAE {d} {m}")
                near(mean(v["mse"]), doc_mse, f"README {kind} MSE {d} {m}")
                if kind == "huber_valmae":
                    say(
                        name,
                        f"{LABEL[d]} {LABEL[m]}: Huber probe beats training-median constant "
                        f"in {beat_mae}/5 seeds, mean MAE reduction {100 * skill_mae:.1f}%",
                    )
                else:
                    say(
                        name,
                        f"{LABEL[d]} {LABEL[m]}: ridge (val MSE) beats training-mean constant "
                        f"on MSE in {beat_mse}/5 seeds, mean MSE reduction {100 * skill_mse:.1f}%; "
                        f"beats training-median constant on MAE in {beat_mae}/5 seeds",
                    )
        for alt in ALTS:
            huber = wins(
                probe["huber_valmae", "mae"]["mae"], probe["huber_valmae", alt]["mae"], False
            )
            ridge = wins(
                probe["ridge_valmse", "mae"]["mse"], probe["ridge_valmse", alt]["mse"], False
            )
            say(
                name,
                f"{LABEL[d]}: MAE objective lower Huber-probe abs. error than {LABEL[alt]} in "
                f"{huber}/5 seeds, lower ridge-probe MSE in {ridge}/5",
            )
        # Fine-tuned SSL beside the full-label supervised baselines (as in Appendix Table 7).
        finetune = {m: ref[d, LOS, "full_finetune", m] for m in SSL + BASELINES}
        if any(set(v) != SEEDS for v in finetune.values()):
            raise ValueError(f"Missing seeds in {d} LOS fine-tune or baselines")
        for label, values in (
            ("Probe, abs.", {m: probe["huber_valmae", m]["mae"] for m in SSL}),
            ("Probe, sq.", {m: probe["ridge_valmse", m]["mae"] for m in SSL}),
            ("Fine-tune, sq.", finetune),
        ):
            best = min(mean(v) for v in values.values())
            cells = [
                # Centered dashes keep empty cells from running into the next column.
                bold(msd(values[m]), mean(values[m]) == best)
                if m in values
                else r"\multicolumn{1}{c}{--}"
                for m in SSL + BASELINES
            ]
            lines.append([LABEL[d], label, *cells, msd(median)])
    table(
        out / "los-matched.tex",
        header(
            "Data",
            "Probe/protocol",
            "MAE",
            "JEPA",
            "Con.",
            "Transf.",
            "GRU-D",
            "XGBoost",
            "Const.",
        ),
        lines,
        "@{}llrrrrrrr@{}",
        pre=[r"\setlength{\tabcolsep}{2.5pt}"],
    )
