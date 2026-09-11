#!/usr/bin/env python3
# ruff: noqa: E402 -- establish the hard timeout before importing numerical libraries
"""Synthetic-only CPU test of objectives, transfer, extension, and negative contracts.

Run from supplement root: uv run --project framework python smoke.py
Hard timeout 120 seconds. No data download, training services, or credentials.
"""

import signal

signal.alarm(120)

import copy
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(
    0, str(ROOT / "framework/src" if (ROOT / "framework").exists() else ROOT.parents[1] / "src")
)

import torch
from omegaconf import OmegaConf

from slices.benchmark.contract import compare_contracts
from slices.models.encoders import TransformerConfig, TransformerEncoder
from slices.models.pretraining.base import BaseSSLObjective, SSLConfig
from slices.models.pretraining.factory import (
    CONFIG_REGISTRY,
    SSL_REGISTRY,
    build_ssl_objective,
    get_ssl_config_class,
)
from slices.training.checkpoint_loading import load_encoder_weights


class ExampleObjective(BaseSSLObjective):
    """Toy extension illustrating the interface; no scientific result claim."""

    def forward(self, x, obs_mask):
        tokens, valid, _ = self.encoder.tokenize(x, obs_mask)
        z = self.encoder.encode(tokens, padding_mask=valid)
        loss = z.square().mean()
        return loss, {"example_loss": loss.detach()}


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    x = torch.randn(8, 24, 84)
    mask = torch.rand_like(x) > 0.85
    mask[:, 0, :] = False
    x = x.masked_fill(~mask, 0)
    base = TransformerConfig(
        d_input=84,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_seq_length=24,
        obs_aware=True,
        pooling="none",
    )
    SSL_REGISTRY["example"] = ExampleObjective
    CONFIG_REGISTRY["example"] = SSLConfig
    results = []
    with tempfile.TemporaryDirectory() as temp:
        for name in ("mae", "jepa", "contrastive", "ts2vec", "example"):
            encoder = TransformerEncoder(copy.deepcopy(base))
            config = get_ssl_config_class(name)()
            config.name = name
            objective = build_ssl_objective(encoder, config)
            before = {k: v.clone() for k, v in encoder.state_dict().items()}
            optimizer = torch.optim.AdamW(objective.parameters(), lr=1e-3)
            loss, _ = objective(x, mask)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f"{name}: nonfinite/nonscalar loss")
            loss.backward()
            if not any(
                p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in encoder.parameters()
            ):
                raise ValueError(f"{name}: encoder has no finite nonzero gradient")
            optimizer.step()
            if all(torch.equal(before[k], v) for k, v in encoder.state_dict().items()):
                raise ValueError(f"{name}: encoder update absent")
            path = Path(temp) / "encoder.pt"
            saved = {k: v.clone() for k, v in encoder.state_dict().items()}
            torch.save(
                {
                    "version": 3,
                    "encoder_state_dict": encoder.state_dict(),
                    "encoder_config": {"name": "transformer", **asdict(base)},
                    "d_input": 84,
                    "missing_token": None,
                },
                path,
            )
            cfg = OmegaConf.create({"encoder": {"name": "transformer", "pooling": "mean"}})
            transferred = load_encoder_weights(encoder, str(path), cfg, False).eval()
            for key, value in saved.items():
                if not torch.equal(transferred.state_dict()[key], value):
                    raise ValueError("Transfer changed saved encoder weights")
            frozen = copy.deepcopy(transferred.state_dict())
            for p in transferred.parameters():
                p.requires_grad_(False)
            head = torch.nn.Linear(64, 2)
            hopt = torch.optim.AdamW(head.parameters(), lr=1e-3)
            z = transferred(x, mask=mask, padding_mask=mask.any(-1))
            if z.shape != (8, 64) or not torch.isfinite(z).all():
                raise ValueError("Invalid transferred pooled representation")
            torch.nn.functional.cross_entropy(head(z), torch.arange(8) % 2).backward()
            hopt.step()
            if any(not torch.equal(v, transferred.state_dict()[k]) for k, v in frozen.items()):
                raise ValueError("Frozen encoder changed")
            # Full adaptation must propagate a finite gradient to the transferred encoder.
            for p in transferred.parameters():
                p.requires_grad_(True)
            z = transferred(x, mask=mask, padding_mask=mask.any(-1))
            torch.nn.functional.cross_entropy(head(z), torch.arange(8) % 2).backward()
            if not any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in transferred.parameters()
            ):
                raise ValueError("Fine-tuning does not reach the encoder")
            results.append({"objective": name, "loss": float(loss.detach()), "transfer": "pass"})
    spec_path = (
        ROOT / "benchmark.json"
        if (ROOT / "benchmark.json").exists()
        else ROOT.parents[1] / "review/benchmark.json"
    )
    spec = json.loads(spec_path.read_text())
    for path, value in [
        ("feature_names", list(reversed(spec["feature_names"]))),
        ("pretraining_epochs", 99),
        ("checkpoint_policy", "best_ssl_loss"),
    ]:
        candidate = copy.deepcopy(spec)
        candidate[path] = value
        if not compare_contracts(spec, candidate):
            raise ValueError(f"Comparator failed to reject changed {path}")
    candidate = copy.deepcopy(spec)
    candidate["objective"] = "example"
    if compare_contracts(spec, candidate):
        raise ValueError("Valid objective extension was rejected")
    print(
        json.dumps(
            {
                "status": "pass",
                "device": "cpu",
                "synthetic_shape": list(x.shape),
                "objectives": results,
                "seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )
    signal.alarm(0)


if __name__ == "__main__":
    main()
