"""Offline FLOPs measurement for each SSL paradigm.

Instantiates each SSL objective with the same configs used in training runs,
profiles a single forward pass using PyTorch's FlopCounterMode, and estimates
total training FLOPs by multiplying per-step FLOPs by gradient steps from W&B.

Usage:
    uv run python scripts/measure_flops.py
    uv run python scripts/measure_flops.py --no-wandb          # skip W&B query
    uv run python scripts/measure_flops.py --n-features 84     # override feature dim
    uv run python scripts/measure_flops.py --sparsity 0.7      # fraction missing
"""

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from slices.constants import SEQ_LENGTH_HOURS
from slices.models.encoders.factory import build_encoder
from slices.models.pretraining.factory import build_ssl_objective, get_ssl_config_class
from torch.utils.flop_counter import FlopCounterMode

# ---------------------------------------------------------------------------
# Paradigm configs — mirror the YAML defaults used in actual runs
# ---------------------------------------------------------------------------

DEFAULT_N_FEATURES = 84

ENCODER_CONFIG = {
    "d_input": DEFAULT_N_FEATURES,
    "d_model": 64,
    "max_seq_length": SEQ_LENGTH_HOURS,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 256,
    "dropout": 0.1,
    "activation": "gelu",
    "prenorm": True,
    "layer_norm_eps": 1e-5,
    "use_positional_encoding": True,
    "pooling": "none",  # Required for SSL pretraining
    "obs_aware": True,  # All SSL paradigms use obs-aware tokenization
}

# SSL configs — exact values from configs/ssl/*.yaml
SSL_CONFIGS: Dict[str, dict] = {
    "mae": {
        "mask_ratio": 0.5,
        "decoder_d_model": 64,
        "decoder_n_layers": 2,
        "decoder_n_heads": 4,
        "decoder_d_ff": 256,
        "decoder_dropout": 0.1,
    },
    "jepa": {
        "mask_ratio": 0.5,
        "mask_strategy": "block",
        "mask_n_blocks": 3,
        "predictor_d_model": 32,
        "predictor_n_layers": 2,
        "predictor_n_heads": 4,
        "predictor_d_ff": 128,
        "predictor_dropout": 0.1,
        "momentum_base": 0.999,
        "momentum_final": 1.0,
        "loss_type": "mse",
    },
    "contrastive": {
        "mode": "instance",
        "mask_ratio": 0.5,
        "proj_hidden_dim": 256,
        "proj_output_dim": 64,
        "temperature": 0.07,
        "complementary_masks": True,
    },
    "ts2vec": {
        "mask_ratio": 0.5,
        "noise_scale": 0.01,
        "crop_ratio": 1.0,
        "proj_hidden_dim": 256,
        "proj_output_dim": 64,
        "temperature": 0.05,
        "n_hierarchical_scales": 4,
    },
}


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def measure_forward_flops(
    model: nn.Module,
    x: torch.Tensor,
    obs_mask: torch.Tensor,
) -> int:
    """Measure forward-pass FLOPs for an SSL objective."""
    model.eval()
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            model(x, obs_mask)
        return flop_counter.get_total_flops()


def build_objective(name: str, encoder_config: dict) -> nn.Module:
    """Build an SSL objective with a fresh encoder."""
    encoder = build_encoder("transformer", encoder_config)
    ssl_config_cls = get_ssl_config_class(name)
    ssl_config = ssl_config_cls(**SSL_CONFIGS[name])
    return build_ssl_objective(encoder, ssl_config)


def format_flops(flops: int) -> str:
    """Format FLOPs as human-readable string."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:,} FLOPs"


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def query_wandb_runs(entity: str, project: str, revision: str | None = None) -> Dict[str, dict]:
    """Query W&B for pretrain run gradient steps and wall-clock time.

    Returns dict keyed by SSL name -> {grad_steps, wall_clock_s, n_runs}.
    """
    import wandb

    api = wandb.Api(timeout=300)
    path = f"{entity}/{project}" if entity else project

    # Server-side filter: only finished pretrain runs from the selected corpus.
    required_tags = ["phase:pretrain"]
    if revision:
        required_tags.append(f"revision:{revision}")
    filters = {"tags": {"$all": required_tags}, "state": "finished"}
    runs = api.runs(path, filters=filters, order="-created_at")

    results: Dict[str, list] = {}
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary_metrics or {})

        ssl_name = config.get("ssl", {}).get("name") or config.get("ssl_name")
        if not ssl_name:
            continue

        grad_steps = summary.get("train/gradient_steps") or summary.get("trainer/global_step")
        wall_clock = summary.get("train/wall_clock_seconds")

        if grad_steps is None:
            continue

        if ssl_name not in results:
            results[ssl_name] = []
        results[ssl_name].append(
            {
                "grad_steps": int(grad_steps),
                "wall_clock_s": float(wall_clock) if wall_clock is not None else None,
                "run_name": run.name,
            }
        )

    # Aggregate per paradigm
    aggregated = {}
    for name, run_list in results.items():
        steps = [r["grad_steps"] for r in run_list]
        clocks = [r["wall_clock_s"] for r in run_list if r["wall_clock_s"] is not None]
        aggregated[name] = {
            "avg_grad_steps": sum(steps) / len(steps),
            "avg_wall_clock_s": sum(clocks) / len(clocks) if clocks else None,
            "n_runs": len(run_list),
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs per SSL paradigm")
    parser.add_argument(
        "--n-features",
        type=int,
        default=DEFAULT_N_FEATURES,
        help=f"Number of input features (d_input; default: {DEFAULT_N_FEATURES})",
    )
    parser.add_argument(
        "--seq-length", type=int, default=SEQ_LENGTH_HOURS, help="Sequence length (T)"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument(
        "--sparsity", type=float, default=0.7, help="Fraction of missing values (0-1)"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B query")
    parser.add_argument(
        "--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY", ""), help="W&B entity"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "slices-thesis"),
        help="W&B project",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=os.environ.get("SLICES_REVISION", "thesis-v1"),
        help="Revision tag to query in W&B (use empty string to disable)",
    )
    args = parser.parse_args()

    # Build encoder config with CLI overrides
    encoder_config = {
        **ENCODER_CONFIG,
        "d_input": args.n_features,
        "max_seq_length": args.seq_length,
    }

    # Create synthetic input with realistic sparsity
    B, T, D = args.batch_size, args.seq_length, args.n_features
    torch.manual_seed(42)
    x = torch.randn(B, T, D)
    obs_mask = (torch.rand(B, T, D) > args.sparsity).bool()
    # Ensure at least one observation per timestep to avoid degenerate cases
    for b in range(B):
        for t in range(T):
            if not obs_mask[b, t].any():
                obs_mask[b, t, 0] = True

    paradigms = ["mae", "jepa", "contrastive", "ts2vec"]

    print(f"Input shape: ({B}, {T}, {D}), sparsity: {args.sparsity:.0%}")
    print(f"Avg observations per sample: {obs_mask.float().sum() / B:.0f} / {T * D}")
    print()

    # Query W&B if requested
    wandb_data = {}
    if not args.no_wandb:
        try:
            revision = args.revision or None
            target = (
                f"{args.wandb_entity}/{args.wandb_project}"
                if args.wandb_entity
                else args.wandb_project
            )
            print(f"Querying W&B target: {target}, revision: {revision or 'unfiltered'}")
            wandb_data = query_wandb_runs(args.wandb_entity, args.wandb_project, revision)
            print(f"Found W&B data for: {list(wandb_data.keys())}")
            print()
        except Exception as e:
            print(f"W&B query failed ({e}), continuing without it.\n")

    # Measure each paradigm
    rows = []
    for name in paradigms:
        objective = build_objective(name, encoder_config)
        trainable, total = count_parameters(objective)

        fwd_flops = measure_forward_flops(objective, x.clone(), obs_mask.clone())

        # Backward ≈ 2× forward; total step = fwd + bwd = 3× fwd
        # Exception: JEPA target encoder is forward-only (no gradient)
        if name == "jepa":
            # The FlopCounterMode already captures both online + target encoder forward.
            # But backward only applies to online encoder + predictor (not target).
            # Approximate: target encoder ≈ online encoder forward FLOPs.
            # So: total_step ≈ fwd_flops (online+target) + 2 × (fwd_flops - target_fwd)
            # Since target ≈ online encoder, target_fwd ≈ fwd_flops * (encoder_share).
            # Simpler: fwd already includes target. bwd ≈ 2 × fwd_without_target.
            # We estimate target encoder as ~40% of total forward (encoder is the bulk).
            # Conservative: just use 3× as upper bound like others.
            step_flops = fwd_flops * 3
        else:
            step_flops = fwd_flops * 3

        wb = wandb_data.get(name, {})
        avg_steps = wb.get("avg_grad_steps")
        avg_clock = wb.get("avg_wall_clock_s")
        n_runs = wb.get("n_runs", 0)

        total_flops = step_flops * avg_steps if avg_steps else None

        rows.append(
            {
                "name": name,
                "trainable": trainable,
                "total_params": total,
                "fwd_flops": fwd_flops,
                "step_flops": step_flops,
                "avg_steps": avg_steps,
                "total_flops": total_flops,
                "avg_clock": avg_clock,
                "n_runs": n_runs,
            }
        )

    # Print results table
    cols = [
        f"{'Paradigm':<14}",
        f"{'Params':>10}",
        f"{'FLOPs/step (fwd)':>20}",
        f"{'FLOPs/step (fwd+bwd)':>22}",
        f"{'Grad steps':>12}",
        f"{'Total FLOPs':>14}",
        f"{'Wall-clock':>12}",
        f"{'Runs':>5}",
    ]
    header = " ".join(cols)
    print(header)
    print("-" * len(header))

    for r in rows:
        steps_str = f"{r['avg_steps']:.0f}" if r["avg_steps"] else "N/A"
        total_str = format_flops(r["total_flops"]) if r["total_flops"] else "N/A"
        clock_str = f"{r['avg_clock']:.0f}s" if r["avg_clock"] else "N/A"
        runs_str = str(r["n_runs"]) if r["n_runs"] else "-"

        print(
            f"{r['name']:<14} "
            f"{format_params(r['trainable']):>10} "
            f"{format_flops(r['fwd_flops']):>20} "
            f"{format_flops(r['step_flops']):>22} "
            f"{steps_str:>12} "
            f"{total_str:>14} "
            f"{clock_str:>12} "
            f"{runs_str:>5}"
        )

    # Also print raw numbers for programmatic use
    print("\n--- Raw values (for tables/plots) ---")
    raw_cols = [
        f"{'Paradigm':<14}",
        f"{'Trainable':>12}",
        f"{'FWD FLOPs':>14}",
        f"{'Step FLOPs':>14}",
        f"{'Avg Steps':>12}",
        f"{'Total FLOPs':>18}",
    ]
    print(" ".join(raw_cols))
    for r in rows:
        print(
            f"{r['name']:<14} "
            f"{r['trainable']:>12,} "
            f"{r['fwd_flops']:>14,} "
            f"{r['step_flops']:>14,} "
            f"{r['avg_steps'] or 0:>12.0f} "
            f"{r['total_flops'] or 0:>18,.0f}"
        )


if __name__ == "__main__":
    main()
