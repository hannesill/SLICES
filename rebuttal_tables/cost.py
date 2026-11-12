"""Pretraining compute table."""

from audit import SSL

from rebuttal_tables.common import (
    LABEL,
    README_NAME,
    f,
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
    split_pair,
    table,
)


def compute(out, pkg, docs):
    flops = index(load(pkg / "p2_flops_per_objective.csv"), "paradigm")
    runs = load(pkg / "p2_pretrain_compute_per_run.csv")
    agg = index(load(pkg / "p2_pretrain_compute_by_paradigm_dataset.csv"), "paradigm", "dataset")
    if len(runs) != 45 or any(float(r["epochs_completed"]) != 100 for r in runs):
        raise ValueError("Pretraining compute runs incomplete")
    timing = {}
    for suffix, d in (("", "miiv"), ("_eicu", "eicu"), ("_combined", "combined")):
        for r in load(pkg / f"p2_l4_exclusive_step_timing{suffix}.csv"):
            setup = (r["dataset"], r["device"], r["timed_steps"], r["batch_size"])
            if setup != (d, "NVIDIA L4", "300", "256"):
                raise ValueError("Unexpected exclusive-L4 timing configuration")
            timing[r["paradigm"], d] = r
    doc_shared = readme(docs, "Objective", "Dataset", "W&B runtime, h (sd)")
    doc_l4 = readme(docs, "Objective", "ms/step", nth=0)
    doc_proj = readme(docs, "Objective", "miiv (205 steps/epoch)")
    order = ("miiv", "eicu", "combined")
    name = "compute"
    lines, totals = [], {}
    for m in SSL:
        fl = flops[m]
        trainable, total = int(fl["trainable_params"]), int(fl["total_params"])
        hours, projected = {}, {}
        for d in order:
            a = agg[m, d]
            for k in ("trainable_params", "total_params", "forward_flops_per_sample"):
                if a[k] != fl[k]:
                    raise ValueError("FLOP/parameter tables disagree")
            hours[d] = {
                s: v / 3600 for s, v in seeds(runs, "runtime_sec", paradigm=m, dataset=d).items()
            }
            same(mean(hours[d]), float(a["runtime_hours_mean"]), "W&B runtime hours")
            same(sd(hours[d]), float(a["runtime_hours_std"]), "W&B runtime hours SD")
            r = timing[m, d]
            projected[d] = (
                float(r["ms_per_step"]) * int(r["steps_per_epoch"]) * int(r["max_epochs"]) / 3.6e6
            )
            same(projected[d], float(r["projected_train_hours_full_schedule"]), "L4 projection")
            doc = next(x for x in doc_shared if x[0] == README_NAME[m] and x[1] == d)
            mu, sigma = split_pair(doc[2])
            near(mean(hours[d]), mu, f"README runtime {m} {d}")
            near(sd(hours[d]), sigma, f"README runtime SD {m} {d}")
        doc = next(x for x in doc_proj if x[0] == README_NAME[m])
        for d, text in zip(order, doc[1:4], strict=True):
            near(projected[d], text, f"README projection {m} {d}")
        ms_step = float(timing[m, "miiv"]["ms_per_step"])
        near(ms_step, next(x for x in doc_l4 if x[0] == README_NAME[m])[1], "README L4 ms/step")
        if total == trainable:
            params = f"{trainable:,}"
        elif m == "jepa":
            params = f"{trainable:,} + {total - trainable:,} EMA"
        else:
            raise ValueError("Unexpected non-trainable parameters")
        lines.append(
            [LABEL[m], f(float(fl["forward_flops_per_sample"]) / 1e6, 2), params, f(ms_step, 1)]
            + [x for d in order for x in (msd(hours[d], 2), f(projected[d], 2))]
        )
        rs = [r for r in runs if r["paradigm"] == m]
        totals[m] = dict(
            shared=sum(float(r["runtime_sec"]) for r in rs) / 3600,
            l4=sum(
                float(r["gradient_steps"]) * float(timing[m, r["dataset"]]["ms_per_step"])
                for r in rs
            )
            / 3.6e6,
            per_run=sum(
                float(r["runtime_sec"]) / float(r["mean_concurrent_project_runs"]) for r in rs
            )
            / 3600,
            by_cell=sum(
                5 * float(agg[m, d]["shared_gpu_hours_over_mean_concurrency"]) for d in order
            ),
            cost=float(timing[m, "miiv"]["step_cost_relative_to_mae"]),
        )
    # Per dataset: shared-GPU W&B wall-clock hours and projected idle-L4 100-epoch hours.
    table(
        out / "compute.tex",
        [
            r" & Fwd. MFLOPs & Trainable & Idle L4 & "
            + " & ".join(r"\multicolumn{2}{c}{" + LABEL[d] + " (h)}" for d in order)
            + r" \\",
            r"\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
            r"Objective & per sample & params & ms/step" + " & Shared & Idle" * 3 + r" \\",
        ],
        lines,
        "@{}lrrrrrrrrr@{}",
        pre=[r"\setlength{\tabcolsep}{4pt}"],
    )
    for m in SSL:
        t = totals[m]
        say(
            name,
            f"{LABEL[m]}, 15 runs: shared W&B wall-clock {t['shared']:.1f} h; projected idle-L4 "
            f"{t['l4']:.1f} h (gradient steps x same-dataset ms/step); wall-clock/concurrency "
            f"{t['per_run']:.1f} h summed per run ({t['by_cell']:.1f} h from cell means); "
            f"idle-L4 step cost vs MAE {t['cost']:.2f}",
        )
    total = {k: sum(t[k] for t in totals.values()) for k in ("shared", "l4", "per_run", "by_cell")}
    say(
        name,
        f"all 45 runs: shared W&B wall-clock {total['shared']:.1f} h; projected idle-L4 "
        f"{total['l4']:.1f} h; concurrency-adjusted {total['per_run']:.1f} L4-hours summed per run "
        f"({total['by_cell']:.1f} from cell means)",
    )
