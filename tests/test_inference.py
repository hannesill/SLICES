"""Tests for shared inference utilities."""

import torch
from slices.eval.inference import run_inference


class _DummyRegressionModel(torch.nn.Module):
    def forward(self, timeseries: torch.Tensor, mask: torch.Tensor):
        del timeseries, mask
        return {"probs": torch.tensor([[0.25], [0.75]], dtype=torch.float32)}


def test_run_inference_squeezes_singleton_regression_outputs():
    model = _DummyRegressionModel()
    dataloader = [
        {
            "timeseries": torch.zeros(2, 3, 4),
            "mask": torch.ones(2, 3, 4, dtype=torch.bool),
            "label": torch.tensor([1.0, 2.0]),
            "stay_id": torch.tensor([10, 11]),
        }
    ]

    predictions, labels, stay_ids = run_inference(model, dataloader)

    assert predictions.shape == (2,)
    assert labels.shape == (2,)
    assert torch.allclose(predictions, torch.tensor([0.25, 0.75]))
    assert torch.allclose(labels, torch.tensor([1.0, 2.0]))
    assert stay_ids == [10, 11]


class _DummySingletonModel(torch.nn.Module):
    def forward(self, timeseries: torch.Tensor, mask: torch.Tensor):
        del timeseries, mask
        return {"probs": torch.tensor([[0.5]], dtype=torch.float32)}


def test_run_inference_handles_singleton_batches_without_scalar_squeeze():
    model = _DummySingletonModel()
    dataloader = [
        {
            "timeseries": torch.zeros(1, 3, 4),
            "mask": torch.ones(1, 3, 4, dtype=torch.bool),
            "label": torch.tensor([1.0]),
            "stay_id": torch.tensor([10]),
        }
    ]

    predictions, labels, stay_ids = run_inference(model, dataloader)

    assert predictions.shape == (1,)
    assert labels.shape == (1,)
    assert torch.allclose(predictions, torch.tensor([0.5]))
    assert torch.allclose(labels, torch.tensor([1.0]))
    assert stay_ids == [10]
