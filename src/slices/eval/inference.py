"""Shared inference utilities for model evaluation.

Provides a reusable function for collecting predictions, labels, and stay IDs
from a test dataloader — used by both in-training fairness evaluation and
the standalone post-run fairness script.
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
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
            all_preds.append(probs.reshape(-1).cpu())
        all_labels.append(batch["label"].reshape(-1).cpu())
        all_stay_ids.extend(
            batch["stay_id"].tolist()
            if isinstance(batch["stay_id"], torch.Tensor)
            else batch["stay_id"]
        )

    predictions = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return predictions, labels, all_stay_ids


def extract_tabular_features(dataset, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Extract the tabular summary features used by the XGBoost baseline."""
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    ts = dataset._timeseries_tensor[idx_tensor]  # (N, T, D)
    mask = dataset._mask_tensor[idx_tensor]  # (N, T, D) bool

    n_samples, n_timesteps, n_features = ts.shape
    mask_float = mask.float()

    obs_count = mask_float.sum(dim=1)
    obs_frac = obs_count / n_timesteps

    masked_ts = ts * mask_float
    feat_sum = masked_ts.sum(dim=1)
    safe_count = obs_count.clamp(min=1)
    feat_mean = feat_sum / safe_count

    diff_sq = ((ts - feat_mean.unsqueeze(1)) ** 2) * mask_float
    feat_var = diff_sq.sum(dim=1) / safe_count
    feat_std = torch.sqrt(feat_var)

    zeros = ts.new_zeros(n_samples, n_features)
    ts_for_min = ts.clone()
    ts_for_min[~mask] = float("inf")
    raw_min = ts_for_min.min(dim=1).values
    feat_min = torch.where(obs_count > 0, raw_min, zeros)

    ts_for_max = ts.clone()
    ts_for_max[~mask] = float("-inf")
    raw_max = ts_for_max.max(dim=1).values
    feat_max = torch.where(obs_count > 0, raw_max, zeros)

    first_idx = mask.float().argmax(dim=1)
    last_idx = n_timesteps - 1 - mask.flip(dims=[1]).float().argmax(dim=1)

    feat_first = ts.gather(1, first_idx.unsqueeze(1)).squeeze(1)
    feat_last = ts.gather(1, last_idx.unsqueeze(1)).squeeze(1)

    no_obs = obs_count == 0
    feat_mean = torch.nan_to_num(feat_mean, nan=0.0)
    feat_first[no_obs] = 0.0
    feat_last[no_obs] = 0.0

    features = torch.stack(
        [feat_mean, feat_std, feat_min, feat_max, feat_first, feat_last, obs_count, obs_frac],
        dim=-1,
    )
    x = features.reshape(n_samples, n_features * 8).numpy()

    if dataset._labels_tensor is not None:
        y = dataset._labels_tensor[idx_tensor].numpy()
    else:
        y = np.zeros(n_samples, dtype=np.float32)

    return x, y


def load_xgboost_model(task_type: str, model_path: str | Path):
    """Load a saved XGBoost baseline model from disk."""
    from xgboost import XGBClassifier, XGBRegressor

    model = XGBRegressor() if task_type == "regression" else XGBClassifier()
    model.load_model(str(model_path))
    return model


def run_xgboost_inference(
    model_path: str | Path,
    task_type: str,
    dataset,
    indices: list[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Run inference with a saved XGBoost baseline over a dataset subset."""
    model = load_xgboost_model(task_type, model_path)
    features, labels = extract_tabular_features(dataset, indices)

    if task_type == "regression":
        predictions = model.predict(features)
    else:
        predictions = model.predict_proba(features)[:, 1]

    stay_ids = [dataset.stay_ids[i] for i in indices]
    return (
        torch.tensor(predictions, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        stay_ids,
    )
