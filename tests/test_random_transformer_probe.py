"""Tests for the cached random Transformer linear-probe diagnostic runner."""

from __future__ import annotations

from scripts.diagnostics import run_cached_random_transformer_probe as cached_runner


def test_cached_runner_dry_run_limit_builds_expected_jobs():
    args = cached_runner.parse_args(["--dry-run", "--limit", "2", "--max-epochs", "1"])
    jobs = cached_runner.build_jobs(args)

    assert len(jobs) == 2
    assert jobs[0].dataset == "miiv"
    assert jobs[0].task == "mortality_24h"
    assert jobs[0].seed == 42
    assert jobs[0].output_dir.as_posix() == (
        "outputs/diagnostics/random_transformer_probe_cached/miiv/" "mortality_24h/seed42"
    )
    assert jobs[1].seed == 123


def test_cached_runner_completion_marker_requires_marker(monkeypatch, tmp_path):
    args = cached_runner.parse_args(
        ["--datasets", "combined", "--tasks", "mortality_24h", "--seeds", "42"]
    )
    job = cached_runner.build_jobs(args)[0]
    monkeypatch.setattr(cached_runner, "REPO_ROOT", tmp_path)

    assert not cached_runner._job_complete(job)

    marker = tmp_path / job.output_dir / cached_runner.COMPLETION_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n")

    assert cached_runner._job_complete(job)


def test_cached_runner_accepts_submission_flags():
    args = cached_runner.parse_args(
        ["--datasets", "combined", "--tasks", "mortality_24h", "--seeds", "42"]
    )
    assert args.datasets == ["combined"]

    args = cached_runner.parse_args(
        [
            "--datasets",
            "eicu",
            "--tasks",
            "mortality_24h",
            "--seeds",
            "42",
            "--accelerator",
            "auto",
            "--devices",
            "1",
            "--no-wandb",
            "--skip-existing",
        ]
    )
    jobs = cached_runner.build_jobs(args)

    assert args.wandb is False
    assert args.skip_existing is True
    assert len(jobs) == 1
    assert jobs[0].dataset == "eicu"
    assert jobs[0].output_dir.as_posix() == (
        "outputs/diagnostics/random_transformer_probe_cached/eicu/" "mortality_24h/seed42"
    )


def test_cached_runner_compose_cfg_is_self_contained():
    args = cached_runner.parse_args(
        [
            "--datasets",
            "eicu",
            "--tasks",
            "mortality_24h",
            "--seeds",
            "42",
            "--max-epochs",
            "1",
        ]
    )
    job = cached_runner.build_jobs(args)[0]

    cfg = cached_runner._compose_cfg(args, job)

    assert cfg.paradigm == "random_transformer"
    assert cfg.experiment_class == "random_probe_diagnostic"
    assert cfg.experiment_subtype == "random_transformer_cached_linear_probe"
    assert cfg.revision == "random-probe-cached-v1"
    assert cfg.training.max_epochs == 1
    assert cfg.scheduler.T_max == 1
    assert cfg.task.head_type == "linear"
    assert cfg.logging.use_wandb is False
