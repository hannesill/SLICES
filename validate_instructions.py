#!/usr/bin/env python3
"""Validate documented commands in a new ZIP extraction and virtual environment.

Agent-operated acceptance check, not independent human reuse. Executes no clinical
training. Printed credentialed recipes are composed with --cfg job only.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import time
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-budget-seconds", type=int, default=240)
    args = parser.parse_args()
    if not 1 <= args.synthetic_budget_seconds <= 300:
        raise ValueError("Synthetic budget must be between 1 and 300 seconds")
    if args.out.exists():
        raise ValueError("Validation output exists; use a fresh directory")
    args.out.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe ZIP member")
        archive.extractall(args.out)
    root = (args.out / "slices-supplement").resolve()
    if (root / "framework/.venv").exists() or (root / ".git").exists():
        raise ValueError("Archive contains an environment or Git checkout")
    env = dict(os.environ)
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_MODE="disabled")
    receipt = {
        "archive_sha256": digest(args.archive),
        "validator_sha256": digest(Path(__file__)),
        "validation_type": (
            "agent; fresh extraction and venv on the same host; not independent human reuse"
        ),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cache_policy": (
            "uv distribution and interpreter caches may be reused; no venv or source copied"
        ),
        "assistance": "No source/config changes or fallback commands during this acceptance run",
        "synthetic_budget_seconds": args.synthetic_budget_seconds,
        "synthetic_wall_seconds": 0.0,
        "commands": [],
        "instructions_sha256": {
            name: digest(root / name) for name in ("README.md", "TS2VEC_CASE.md", "TRAINING.md")
        },
    }

    def save():
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(name, command, cwd=root, synthetic=False, timeout=180):
        if synthetic:
            remaining = int(args.synthetic_budget_seconds - receipt["synthetic_wall_seconds"])
            if remaining < 1:
                raise ValueError("Cumulative synthetic budget exhausted")
            timeout = min(120, remaining)
        start = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
            status = process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            status = 124
        elapsed = time.monotonic() - start
        (args.out / f"{name}.txt").write_text(output)
        receipt["commands"].append(
            {
                "name": name,
                "command": command,
                "cwd": str(cwd.relative_to(root)) or ".",
                "returncode": status,
                "wall_seconds": elapsed,
                "timeout_seconds": timeout,
                "synthetic": synthetic,
            }
        )
        if synthetic:
            receipt["synthetic_wall_seconds"] += elapsed
        save()
        print(json.dumps(receipt["commands"][-1]), flush=True)
        if status != 0:
            raise RuntimeError(
                f"Documented command failed: {name}; see its log. No fallback applied."
            )
        return output

    # These shell lines appear verbatim in the packaged README/case/training guides.
    commands = [
        ("audit", "uv run --no-project python audit.py", False),
        ("install", "uv sync --project framework --frozen --extra viz --python 3.12", False),
        ("report", "uv run --no-project python extension_report.py --out extension-report", False),
        (
            "extension",
            "uv run --project framework python extension.py --out extension-output",
            True,
        ),
        ("smoke", "uv run --project framework python smoke.py", True),
        ("render", "uv run --project framework python render.py --out paper/generated", False),
    ]
    guides = "\n".join(
        (root / name).read_text() for name in ("README.md", "TS2VEC_CASE.md", "TRAINING.md")
    )
    for name, command, synthetic in commands:
        if command not in guides:
            raise ValueError(f"Command absent from packaged instructions: {command}")
        run(name, shlex.split(command), synthetic=synthetic)
    generated = json.loads((root / "MANIFEST.json").read_text())["files"]
    equality = {
        name: digest(root / name) == info["sha256"]
        for name, info in generated.items()
        if name.startswith("paper/generated/")
    }
    receipt["generated_byte_equality"] = equality
    if not all(equality.values()):
        raise ValueError("Regenerated manuscript assets differ from the packaged reference")
    for protocol in ("linear_probe", "full_finetune"):
        command = (
            "uv run --no-project python training_recipe.py --tree framework --objective ts2vec "
            f"--dataset miiv --task mortality_hospital --seed 42 --protocol {protocol}"
        )
        if command not in guides:
            raise ValueError("Recipe absent from instructions")
        recipe = run(f"recipe-{protocol}", shlex.split(command))
        lines = [line for line in recipe.splitlines() if line.startswith("uv ")]
        if len(lines) != 2:
            raise ValueError("Unexpected recipe output")
        for i, line in enumerate(lines):
            run(f"config-{protocol}-{i}", shlex.split(line) + ["--cfg", "job"], root / "framework")
    run(
        "preprocessing-dry-run",
        shlex.split(
            "uv run --project ../framework python "
            "../framework/scripts/preprocessing/build_processed_data.py "
            "--datasets miiv eicu --combined --dry-run"
        ),
        root / "historical",
    )
    run(
        "latex",
        shlex.split("latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex"),
        root / "paper",
    )
    log = (root / "paper/main.log").read_text()
    page = re.search(r"ML4H-MAIN-END-PAGE:(\d+)", log)
    if not page or int(page[1]) > 8:
        raise ValueError("Main text exceeds eight pages or lacks the boundary marker")
    if re.search(r"Overfull \\[hv]box|LaTeX Warning: (?:Reference|Citation).*undefined", log):
        raise ValueError("PDF has overflow or unresolved citations/references")
    receipt["main_pages"] = int(page[1])
    receipt["paper_pdf_sha256"] = digest(root / "paper/main.pdf")
    receipt["status"] = "pass; complete PDF still requires visual inspection"
    save()


if __name__ == "__main__":
    main()
