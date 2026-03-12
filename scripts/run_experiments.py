#!/usr/bin/env python3
"""
Parallel experiment runner for SLICES.

Generates all experiment configurations across sprints, resolves dependencies,
and executes them in parallel with crash recovery and state persistence.

Usage:
    uv run python scripts/run_experiments.py warmup --sprint 1
    uv run python scripts/run_experiments.py run --sprint 1 --parallel 4
    uv run python scripts/run_experiments.py run --sprint 1 2 3 --parallel 6 --dry-run
    uv run python scripts/run_experiments.py status
    uv run python scripts/run_experiments.py status --sprint 1
    uv run python scripts/run_experiments.py retry --failed --parallel 4
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SSL_PARADIGMS = ["mae", "jepa", "contrastive"]
DATASETS = ["miiv", "eicu", "combined"]
TASKS = ["mortality_24h", "mortality_hospital", "aki_kdigo", "los_remaining"]
SEEDS = [42, 123, 456]
LABEL_FRACTIONS_FULL = [0.01, 0.05, 0.1, 0.25, 0.5]
LABEL_FRACTIONS_TREND = [0.1]
LR_ABLATION = [2e-4, 5e-4, 2e-3]  # 1e-3 reused from Phase 1
MASK_RATIO_ABLATION = [0.3, 0.75]  # 0.5 reused from Phase 1
TRANSFER_PAIRS = [("miiv", "eicu"), ("eicu", "miiv")]

STATE_FILE = Path("outputs/experiment_state.json")
LOG_DIR = Path("logs/runner")

# Protocol defaults
PROTO_A = {"freeze_encoder": True, "max_epochs": 50, "patience": 10, "lr": 1e-4}
PROTO_B = {"freeze_encoder": False, "max_epochs": 100, "patience": 10, "lr": 3e-4}


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
@dataclass
class Run:
    id: str
    sprint: str
    run_type: str  # "pretrain" | "finetune" | "supervised"
    paradigm: str  # "mae" | "jepa" | "contrastive" | "supervised"
    dataset: str
    seed: int
    output_dir: str
    depends_on: list[str] = field(default_factory=list)
    task: str | None = None
    label_fraction: float = 1.0
    freeze_encoder: bool | None = None
    extra_overrides: dict = field(default_factory=dict)
    # For transfer learning: dataset the encoder was pretrained on
    source_dataset: str | None = None

    def build_command(self, runs_by_id: dict[str, Run]) -> list[str]:
        """Build the subprocess command for this run."""
        if self.run_type == "pretrain":
            return self._pretrain_cmd()
        elif self.run_type == "finetune":
            return self._finetune_cmd(runs_by_id)
        elif self.run_type == "supervised":
            return self._supervised_cmd()
        else:
            raise ValueError(f"Unknown run_type: {self.run_type}")

    def _pretrain_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/pretrain.py",
            f"dataset={self.dataset}",
            f"ssl={self.paradigm}",
            f"seed={self.seed}",
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
        return cmd

    def _finetune_cmd(self, runs_by_id: dict[str, Run]) -> list[str]:
        # Find the pretrain dependency to get encoder path
        pretrain_id = [d for d in self.depends_on if d in runs_by_id]
        if not pretrain_id:
            raise ValueError(f"Finetune run {self.id} has no pretrain dependency")
        pretrain_dir = runs_by_id[pretrain_id[0]].output_dir

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/finetune.py",
            f"dataset={self.dataset}",
            f"checkpoint={pretrain_dir}/encoder.pt",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        if self.freeze_encoder is True:
            cmd += [
                "training.freeze_encoder=true",
                f"training.max_epochs={PROTO_A['max_epochs']}",
                f"training.early_stopping_patience={PROTO_A['patience']}",
                f"optimizer.lr={PROTO_A['lr']}",
            ]
        elif self.freeze_encoder is False:
            cmd += [
                "training.freeze_encoder=false",
                f"training.max_epochs={PROTO_B['max_epochs']}",
                f"training.early_stopping_patience={PROTO_B['patience']}",
                f"optimizer.lr={PROTO_B['lr']}",
            ]
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
        return cmd

    def _supervised_cmd(self) -> list[str]:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/supervised.py",
            f"dataset={self.dataset}",
            f"tasks={self.task}",
            f"seed={self.seed}",
            f"sprint={self.sprint}",
            f"hydra.run.dir={self.output_dir}",
        ]
        if self.label_fraction < 1.0:
            cmd.append(f"label_fraction={self.label_fraction}")
        for k, v in self.extra_overrides.items():
            cmd.append(f"{k}={v}")
        return cmd


# ---------------------------------------------------------------------------
# Matrix Generation
# ---------------------------------------------------------------------------
def _pretrain_key(paradigm: str, dataset: str, seed: int, extra: dict | None = None) -> str:
    """Canonical key for deduplicating pretrain runs."""
    key = f"pretrain_{paradigm}_{dataset}_seed{seed}"
    if extra:
        for k, v in sorted(extra.items()):
            short_k = k.split(".")[-1]
            short_v = str(v).replace(".", "")
            key += f"_{short_k}{short_v}"
    return key


def _output_dir(sprint: str, name: str) -> str:
    return f"outputs/sprint{sprint}/{name}"


class MatrixBuilder:
    """Generates all Run objects across sprints with pretrain deduplication."""

    def __init__(self):
        self.runs: list[Run] = []
        self.pretrain_index: dict[str, Run] = {}  # canonical_key -> Run

    def _add_pretrain(
        self, sprint: str, paradigm: str, dataset: str, seed: int, extra: dict | None = None
    ) -> Run:
        """Add a pretrain run, deduplicating by canonical key."""
        extra = extra or {}
        key = _pretrain_key(paradigm, dataset, seed, extra)
        if key in self.pretrain_index:
            return self.pretrain_index[key]

        dir_name = _pretrain_key(paradigm, dataset, seed, extra)
        run = Run(
            id=f"s{sprint}_{dir_name}",
            sprint=sprint,
            run_type="pretrain",
            paradigm=paradigm,
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, dir_name),
            extra_overrides=extra,
        )
        self.runs.append(run)
        self.pretrain_index[key] = run
        return run

    def _get_pretrain(
        self, paradigm: str, dataset: str, seed: int, extra: dict | None = None
    ) -> Run:
        """Retrieve an existing pretrain run."""
        key = _pretrain_key(paradigm, dataset, seed, extra)
        if key not in self.pretrain_index:
            raise KeyError(f"Pretrain not found: {key}")
        return self.pretrain_index[key]

    def _add_finetune(
        self,
        sprint: str,
        paradigm: str,
        dataset: str,
        seed: int,
        task: str,
        freeze: bool,
        pretrain_run: Run,
        label_fraction: float = 1.0,
        extra: dict | None = None,
        source_dataset: str | None = None,
        name_extra: dict | None = None,
    ) -> Run:
        """Add a finetune run.

        Args:
            extra: Hydra overrides added to the finetune command AND name.
            name_extra: Dict used only for directory/ID disambiguation
                (e.g. pretrain ablation params). Not passed to command.
        """
        prefix = "probe" if freeze else "finetune"
        name = f"{prefix}_{paradigm}_{task}_{dataset}_seed{seed}"
        if source_dataset:
            name += f"_from_{source_dataset}"
        if label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            name += f"_frac{frac_str}"
        # Append both extra and name_extra to the directory name
        for d in [name_extra, extra]:
            if d:
                for k, v in sorted(d.items()):
                    short_k = k.split(".")[-1]
                    short_v = str(v).replace(".", "")
                    name += f"_{short_k}{short_v}"

        run = Run(
            id=f"s{sprint}_{name}",
            sprint=sprint,
            run_type="finetune",
            paradigm=paradigm,
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, name),
            depends_on=[pretrain_run.id],
            task=task,
            label_fraction=label_fraction,
            freeze_encoder=freeze,
            extra_overrides=extra or {},
            source_dataset=source_dataset,
        )
        self.runs.append(run)
        return run

    def _add_supervised(
        self, sprint: str, dataset: str, seed: int, task: str, label_fraction: float = 1.0
    ) -> Run:
        name = f"supervised_{task}_{dataset}_seed{seed}"
        if label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            name += f"_frac{frac_str}"
        run = Run(
            id=f"s{sprint}_{name}",
            sprint=sprint,
            run_type="supervised",
            paradigm="supervised",
            dataset=dataset,
            seed=seed,
            output_dir=_output_dir(sprint, name),
            task=task,
            label_fraction=label_fraction,
        )
        self.runs.append(run)
        return run

    # --- Sprint builders ---

    def build_sprint1(self):
        """MIMIC, all tasks, Protocol B + supervised, seed=42."""
        ds, seed, sprint = "miiv", 42, "1"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)

    def build_sprint1b(self):
        """LR sensitivity, MIMIC, mortality_24h, seed=42."""
        ds, seed, task, sprint = "miiv", 42, "mortality_24h", "1b"
        for p in SSL_PARADIGMS:
            for lr in LR_ABLATION:
                extra = {"optimizer.lr": lr}
                pt = self._add_pretrain(sprint, p, ds, seed, extra)
                self._add_finetune(sprint, p, ds, seed, task, False, pt, name_extra=extra)

    def build_sprint1c(self):
        """Mask ratio sensitivity, MIMIC, mortality_24h, seed=42."""
        ds, seed, task, sprint = "miiv", 42, "mortality_24h", "1c"
        for p in SSL_PARADIGMS:
            for mr in MASK_RATIO_ABLATION:
                extra = {"ssl.mask_ratio": mr}
                pt = self._add_pretrain(sprint, p, ds, seed, extra)
                self._add_finetune(sprint, p, ds, seed, task, False, pt, name_extra=extra)

    def build_sprint2(self):
        """MIMIC Protocol A, seed=42 — reuses Sprint 1 pretrains."""
        ds, seed, sprint = "miiv", 42, "2"
        for p in SSL_PARADIGMS:
            pt = self._get_pretrain(p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, True, pt)

    def build_sprint3(self):
        """eICU, both protocols + supervised, seed=42."""
        ds, seed, sprint = "eicu", 42, "3"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
                self._add_finetune(sprint, p, ds, seed, task, True, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)

    def build_sprint4(self):
        """Combined, both protocols + supervised, seed=42."""
        ds, seed, sprint = "combined", 42, "4"
        for p in SSL_PARADIGMS:
            pt = self._add_pretrain(sprint, p, ds, seed)
            for task in TASKS:
                self._add_finetune(sprint, p, ds, seed, task, False, pt)
                self._add_finetune(sprint, p, ds, seed, task, True, pt)
        for task in TASKS:
            self._add_supervised(sprint, ds, seed, task)

    def build_sprint5(self):
        """Seeds 123, 456 for datasets miiv, eicu, combined."""
        sprint = "5"
        for seed in [123, 456]:
            for ds in DATASETS:
                for p in SSL_PARADIGMS:
                    pt = self._add_pretrain(sprint, p, ds, seed)
                    for task in TASKS:
                        self._add_finetune(sprint, p, ds, seed, task, False, pt)
                        self._add_finetune(sprint, p, ds, seed, task, True, pt)
                for task in TASKS:
                    self._add_supervised(sprint, ds, seed, task)

    def build_sprint6(self):
        """Label efficiency ablation — reuses Phase 1 encoders."""
        sprint = "6"
        for seed in SEEDS:
            for ds in DATASETS:
                for p in SSL_PARADIGMS:
                    pt = self._get_pretrain(p, ds, seed)
                    # mortality_24h gets full sweep
                    for frac in LABEL_FRACTIONS_FULL:
                        self._add_finetune(sprint, p, ds, seed, "mortality_24h", False, pt, frac)
                        self._add_finetune(sprint, p, ds, seed, "mortality_24h", True, pt, frac)
                    # Other tasks get trend check
                    for task in TASKS[1:]:
                        for frac in LABEL_FRACTIONS_TREND:
                            self._add_finetune(sprint, p, ds, seed, task, False, pt, frac)
                            self._add_finetune(sprint, p, ds, seed, task, True, pt, frac)
                # Supervised label efficiency
                for frac in LABEL_FRACTIONS_FULL:
                    self._add_supervised(sprint, ds, seed, "mortality_24h", frac)
                for task in TASKS[1:]:
                    for frac in LABEL_FRACTIONS_TREND:
                        self._add_supervised(sprint, ds, seed, task, frac)

    def build_sprint7(self):
        """Cross-dataset transfer — Protocol B only (full finetune)."""
        sprint = "7"
        for seed in SEEDS:
            for source_ds, target_ds in TRANSFER_PAIRS:
                for p in SSL_PARADIGMS:
                    pt = self._get_pretrain(p, source_ds, seed)
                    for task in TASKS:
                        self._add_finetune(
                            sprint, p, target_ds, seed, task, False, pt, source_dataset=source_ds
                        )

    def build_sprint8(self):
        """LR + mask ablations, seeds 123, 456."""
        sprint = "8"
        for seed in [123, 456]:
            for p in SSL_PARADIGMS:
                # LR ablation
                for lr in LR_ABLATION:
                    extra = {"optimizer.lr": lr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint, p, "miiv", seed, "mortality_24h", False, pt, name_extra=extra
                    )
                # Mask ratio ablation
                for mr in MASK_RATIO_ABLATION:
                    extra = {"ssl.mask_ratio": mr}
                    pt = self._add_pretrain(sprint, p, "miiv", seed, extra)
                    self._add_finetune(
                        sprint, p, "miiv", seed, "mortality_24h", False, pt, name_extra=extra
                    )

    def build_all(self) -> list[Run]:
        """Build full experiment matrix. Order matters for dedup."""
        self.build_sprint1()
        self.build_sprint1b()
        self.build_sprint1c()
        self.build_sprint2()
        self.build_sprint3()
        self.build_sprint4()
        self.build_sprint5()
        self.build_sprint6()
        self.build_sprint7()
        self.build_sprint8()
        return self.runs


def generate_all_runs() -> list[Run]:
    builder = MatrixBuilder()
    return builder.build_all()


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"version": 1, "runs": {}}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_run_status(state: dict, run_id: str) -> str:
    return state["runs"].get(run_id, {}).get("status", "pending")


def set_run_status(state: dict, run_id: str, status: str, **kwargs):
    if run_id not in state["runs"]:
        state["runs"][run_id] = {}
    entry = state["runs"][run_id]
    entry["status"] = status
    entry.update(kwargs)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def recover_stale_running(state: dict):
    """Reset running entries whose PIDs are dead back to pending."""
    for run_id, info in state["runs"].items():
        if info.get("status") == "running":
            pid = info.get("pid")
            if pid is None or not is_pid_alive(pid):
                info["status"] = "pending"
                info.pop("pid", None)
                print(f"  Recovered stale run: {run_id}")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def run_scheduler(runs: list[Run], state: dict, parallel: int, dry_run: bool):
    runs_by_id = {r.id: r for r in runs}
    active: dict[str, subprocess.Popen] = {}  # run_id -> Popen
    shutting_down = False

    def handle_signal(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        for rid, proc in active.items():
            try:
                proc.terminate()
            except OSError:
                pass
        # Wait up to 30s for children
        deadline = time.time() + 30
        for rid, proc in list(active.items()):
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Reset interrupted runs to pending
        for rid in active:
            set_run_status(state, rid, "pending")
        save_state(state)
        print("State saved. Interrupted runs reset to pending.")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _print_dry_run(runs, runs_by_id)
        return

    print(f"Scheduler started (parallel={parallel})")
    print(f"State file: {STATE_FILE}")
    print()

    while not shutting_down:
        # 1. Poll active processes
        finished = []
        for rid, proc in active.items():
            ret = proc.poll()
            if ret is not None:
                finished.append((rid, ret))

        # 2. Handle completions
        for rid, exit_code in finished:
            proc = active.pop(rid)
            fh = getattr(proc, "_log_fh", None)
            if fh:
                fh.close()
            run = runs_by_id[rid]
            now = datetime.now(timezone.utc).isoformat()

            elapsed = _format_elapsed(state, rid, now)

            if exit_code == 0:
                # For pretrain, verify encoder.pt exists
                if run.run_type == "pretrain":
                    encoder_path = Path(run.output_dir) / "encoder.pt"
                    if not encoder_path.exists():
                        print(f"  FAILED {rid}: exit 0 but encoder.pt " f"missing {elapsed}")
                        set_run_status(state, rid, "failed", finished_at=now, exit_code=exit_code)
                        _propagate_failure(rid, runs, state)
                        continue
                print(f"  DONE {rid} {elapsed}")
                set_run_status(state, rid, "completed", finished_at=now, exit_code=0)
            else:
                print(f"  FAILED {rid} (exit {exit_code}) {elapsed}")
                set_run_status(state, rid, "failed", finished_at=now, exit_code=exit_code)
                _propagate_failure(rid, runs, state)

        # 3. Find ready runs
        ready = []
        for r in runs:
            status = get_run_status(state, r.id)
            if status != "pending":
                continue
            if r.id in active:
                continue
            # Check all dependencies completed
            deps_ok = all(get_run_status(state, d) == "completed" for d in r.depends_on)
            if deps_ok:
                ready.append(r)

        # 4. Launch runs up to parallel limit
        slots = parallel - len(active)
        for r in ready[:slots]:
            cmd = r.build_command(runs_by_id)
            log_file = LOG_DIR / f"{r.id}.log"
            now = datetime.now(timezone.utc).isoformat()

            print(f"  START {r.id}")
            log_fh = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Store log handle for cleanup
            proc._log_fh = log_fh  # type: ignore[attr-defined]
            active[r.id] = proc
            set_run_status(
                state, r.id, "running", started_at=now, pid=proc.pid, log_file=str(log_file)
            )

        # 5. Save state
        save_state(state)

        # 6. Exit condition
        if not active and not ready:
            break

        time.sleep(2)

    # Close any remaining log handles
    for proc in active.values():
        fh = getattr(proc, "_log_fh", None)
        if fh:
            fh.close()

    # Print summary
    print()
    _print_summary(runs, state)


def _format_elapsed(state: dict, run_id: str, now_iso: str) -> str:
    """Format elapsed time as (Xh Ym Zs)."""
    started = state["runs"].get(run_id, {}).get("started_at")
    if not started:
        return ""
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(now_iso)
        secs = int((t1 - t0).total_seconds())
        if secs < 60:
            return f"({secs}s)"
        elif secs < 3600:
            return f"({secs // 60}m{secs % 60:02d}s)"
        else:
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            return f"({h}h{m:02d}m{s:02d}s)"
    except (ValueError, TypeError):
        return ""


def _propagate_failure(failed_id: str, runs: list[Run], state: dict):
    """Skip runs that depend on a failed run (transitive)."""
    queue = [failed_id]
    while queue:
        fid = queue.pop(0)
        for r in runs:
            if fid in r.depends_on and get_run_status(state, r.id) == "pending":
                set_run_status(state, r.id, "skipped", reason=f"dependency {fid} failed")
                print(f"  SKIP {r.id} (dep {fid} failed)")
                queue.append(r.id)


def _print_dry_run(runs: list[Run], runs_by_id: dict[str, Run]):
    """Print all runs and their commands without executing."""
    print(f"DRY RUN: {len(runs)} runs\n")
    for r in runs:
        deps = ", ".join(r.depends_on) if r.depends_on else "(none)"
        cmd = " ".join(r.build_command(runs_by_id))
        print(f"[{r.id}]")
        print(f"  sprint={r.sprint}  type={r.run_type}  deps={deps}")
        print(f"  dir={r.output_dir}")
        print(f"  cmd: {cmd}")
        print()


# ---------------------------------------------------------------------------
# Status / Retry
# ---------------------------------------------------------------------------
def print_status(sprint_filter: list[str] | None = None):
    all_runs = generate_all_runs()
    state = load_state()

    # Group by sprint
    sprints: dict[str, list[Run]] = {}
    for r in all_runs:
        sprints.setdefault(r.sprint, []).append(r)

    sprint_keys = sorted(sprints.keys(), key=_sprint_sort_key)
    if sprint_filter:
        sprint_keys = [s for s in sprint_keys if s in sprint_filter]

    print(
        f"{'Sprint':>6} | {'Total':>5} | {'Done':>4} | {'Run':>3} | "
        f"{'Fail':>4} | {'Pend':>4} | {'Skip':>4}"
    )
    print("-" * 52)

    totals = {"total": 0, "completed": 0, "running": 0, "failed": 0, "pending": 0, "skipped": 0}

    for s in sprint_keys:
        runs = sprints[s]
        counts = {
            "total": len(runs),
            "completed": 0,
            "running": 0,
            "failed": 0,
            "pending": 0,
            "skipped": 0,
        }
        for r in runs:
            status = get_run_status(state, r.id)
            if status in counts:
                counts[status] += 1
            else:
                counts["pending"] += 1  # unknown → pending
        print(
            f"{s:>6} | {counts['total']:>5} | {counts['completed']:>4} | "
            f"{counts['running']:>3} | {counts['failed']:>4} | "
            f"{counts['pending']:>4} | {counts['skipped']:>4}"
        )
        for k in totals:
            totals[k] += counts[k]

    print("-" * 52)
    print(
        f"{'TOTAL':>6} | {totals['total']:>5} | {totals['completed']:>4} | "
        f"{totals['running']:>3} | {totals['failed']:>4} | "
        f"{totals['pending']:>4} | {totals['skipped']:>4}"
    )


def _print_summary(runs: list[Run], state: dict):
    """Print completion summary for a scheduler run."""
    counts = {"completed": 0, "failed": 0, "skipped": 0, "pending": 0}
    for r in runs:
        s = get_run_status(state, r.id)
        counts[s] = counts.get(s, 0) + 1
    total = len(runs)
    print(
        f"Summary: {counts['completed']}/{total} completed, "
        f"{counts['failed']} failed, {counts['skipped']} skipped, "
        f"{counts['pending']} pending"
    )


def _sprint_sort_key(s: str) -> tuple[int, str]:
    """Sort sprints: 1, 1b, 2, 3, ..."""
    num = ""
    suffix = ""
    for c in s:
        if c.isdigit():
            num += c
        else:
            suffix += c
    return (int(num) if num else 0, suffix)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_run(args):
    all_runs = generate_all_runs()
    state = load_state()
    recover_stale_running(state)

    # Filter to requested sprints
    sprints = [str(s) for s in args.sprint]
    runs = [r for r in all_runs if r.sprint in sprints]

    # Also include dependency runs from earlier sprints (pretrain reuse)
    run_ids = {r.id for r in runs}
    deps_needed = set()
    for r in runs:
        for d in r.depends_on:
            if d not in run_ids:
                deps_needed.add(d)
    if deps_needed:
        all_by_id = {r.id: r for r in all_runs}
        extra = [all_by_id[d] for d in deps_needed if d in all_by_id]
        runs = extra + runs
        if extra and not args.dry_run:
            print(f"Including {len(extra)} dependency run(s) from earlier " f"sprints")

    if not runs:
        print(f"No runs found for sprint(s): {', '.join(sprints)}")
        return

    print(f"Sprint(s) {', '.join(sprints)}: {len(runs)} runs")
    run_scheduler(runs, state, args.parallel, args.dry_run)


def cmd_status(args):
    sprint_filter = [str(s) for s in args.sprint] if args.sprint else None
    print_status(sprint_filter)


def cmd_retry(args):
    all_runs = generate_all_runs()
    state = load_state()
    recover_stale_running(state)

    runs_to_retry = []
    for r in all_runs:
        status = get_run_status(state, r.id)
        if args.failed and status == "failed":
            set_run_status(state, r.id, "pending")
            runs_to_retry.append(r)
        elif args.skipped and status == "skipped":
            set_run_status(state, r.id, "pending")
            runs_to_retry.append(r)

    if not runs_to_retry:
        print("No runs to retry.")
        return

    # Include their dependencies that are completed (already fine) or pending
    all_by_id = {r.id: r for r in all_runs}
    retry_ids = {r.id for r in runs_to_retry}
    deps_needed = set()
    for r in runs_to_retry:
        for d in r.depends_on:
            if d not in retry_ids:
                deps_needed.add(d)
    extra = [all_by_id[d] for d in deps_needed if d in all_by_id]
    runs_to_retry = extra + runs_to_retry

    save_state(state)
    print(f"Retrying {len(runs_to_retry)} runs")
    run_scheduler(runs_to_retry, state, args.parallel, dry_run=False)


def cmd_warmup(args):
    """Pre-build tensor caches for requested sprints.

    Instantiates ICUDataModule for each unique (dataset, task, seed, label_fraction)
    combination sequentially. This populates the tensor cache so that parallel
    experiment runs can load preprocessed data from disk instead of each process
    independently loading and converting raw parquet files (which causes OOM).
    """
    all_runs = generate_all_runs()

    # Filter to requested sprints
    sprints = [str(s) for s in args.sprint]
    runs = [r for r in all_runs if r.sprint in sprints]

    # Also include dependency runs
    run_ids = {r.id for r in runs}
    deps_needed = set()
    for r in runs:
        for d in r.depends_on:
            if d not in run_ids:
                deps_needed.add(d)
    if deps_needed:
        all_by_id = {r.id: r for r in all_runs}
        runs = [all_by_id[d] for d in deps_needed if d in all_by_id] + runs

    # Collect unique (dataset, task, seed, label_fraction) combos
    # task=None for pretrain, task=<name> for finetune/supervised
    combos: dict[tuple, None] = {}  # ordered set via dict
    for r in runs:
        if r.run_type == "pretrain":
            key = (r.dataset, None, r.seed, 1.0)
        else:
            key = (r.dataset, r.task, r.seed, r.label_fraction)
        combos[key] = None

    print(f"Warming up tensor caches for sprint(s) {', '.join(sprints)}")
    print(f"  {len(combos)} unique (dataset, task, seed, label_fraction) combinations\n")

    # Import here to avoid loading heavy deps when not needed
    from slices.data.datamodule import ICUDataModule

    for i, (dataset, task, seed, label_fraction) in enumerate(combos, 1):
        processed_dir = f"data/processed/{dataset}"
        task_str = task or "(pretrain/no task)"
        frac_str = f", frac={label_fraction}" if label_fraction < 1.0 else ""
        print(f"[{i}/{len(combos)}] {dataset} / {task_str} / seed={seed}{frac_str}")

        try:
            dm = ICUDataModule(
                processed_dir=processed_dir,
                task_name=task,
                batch_size=1,  # doesn't matter, we just need setup()
                num_workers=0,
                seed=seed,
                label_fraction=label_fraction,
            )
            dm.setup()
            print(f"  -> Cached ({len(dm.dataset):,} samples)")
            # Free memory before next combo
            del dm
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print("\nWarmup complete. Tensor caches saved to data/processed/<dataset>/.tensor_cache/")
    print("You can now run experiments in parallel without OOM.")


def main():
    parser = argparse.ArgumentParser(description="SLICES parallel experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run experiments for given sprints")
    p_run.add_argument("--sprint", nargs="+", required=True, help="Sprint(s) to run (e.g. 1 1b 2)")
    p_run.add_argument("--parallel", type=int, default=4, help="Max parallel jobs (default: 4)")
    p_run.add_argument("--dry-run", action="store_true", help="Print runs without executing")

    # status
    p_status = sub.add_parser("status", help="Show experiment status")
    p_status.add_argument("--sprint", nargs="*", default=None, help="Filter by sprint(s)")

    # retry
    p_retry = sub.add_parser("retry", help="Retry failed/skipped runs")
    p_retry.add_argument("--failed", action="store_true", help="Retry failed runs")
    p_retry.add_argument("--skipped", action="store_true", help="Retry skipped runs")
    p_retry.add_argument("--parallel", type=int, default=4, help="Max parallel jobs (default: 4)")

    # warmup
    p_warmup = sub.add_parser(
        "warmup", help="Pre-build tensor caches to avoid OOM during parallel runs"
    )
    p_warmup.add_argument(
        "--sprint", nargs="+", required=True, help="Sprint(s) to warmup (e.g. 1 1b 2)"
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retry":
        cmd_retry(args)
    elif args.command == "warmup":
        cmd_warmup(args)


if __name__ == "__main__":
    main()
