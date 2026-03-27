"""Shared inference utilities for model evaluation.

Provides a reusable function for collecting predictions, labels, and stay IDs
from a test dataloader — used by both in-training fairness evaluation and
the standalone post-run fairness script.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Run inference on a dataloader, collecting predictions, labels, and stay IDs.

    Args:
        model: Model that accepts (timeseries, mask) and returns a dict
            with a 'probs' key. Must already be in eval mode.
        dataloader: DataLoader yielding batches with 'timeseries', 'mask',
            'label', and 'stay_id' keys.
        device: Device to run inference on.

    Returns:
        Tuple of (predictions, labels, stay_ids) where:
        - predictions: (N,) tensor of probabilities (positive class for binary)
        - labels: (N,) tensor of ground truth labels
        - stay_ids: list of N stay ID integers
    """
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_stay_ids: list[int] = []

    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(
                batch["timeseries"].to(device),
                batch["mask"].to(device),
            )
        probs = outputs["probs"]
        # For binary classification with 2-class output, take positive class
        if probs.dim() > 1 and probs.shape[1] == 2:
            all_preds.append(probs[:, 1].cpu())
        else:
            all_preds.append(probs.cpu())
        all_labels.append(batch["label"].cpu())
        all_stay_ids.extend(
            batch["stay_id"].tolist()
            if isinstance(batch["stay_id"], torch.Tensor)
            else batch["stay_id"]
        )

    predictions = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return predictions, labels, all_stay_ids
