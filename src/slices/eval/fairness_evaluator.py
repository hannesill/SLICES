"""Fairness evaluation framework for clinical prediction models.

Computes per-group AUROC, worst-group AUROC, demographic parity, and
equalized odds across protected attributes (gender, age group, race).

This module builds on the lower-level fairness utilities in fairness.py
to provide a high-level evaluator that works with static patient data
and produces structured reports suitable for W&B logging.

Protected attributes:
- gender (M/F) — available in all datasets
- age_group (18-44, 45-64, 65-79, 80+) — available in all datasets
- race — available in MIMIC + eICU only
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch
from torchmetrics import AUROC

from slices.eval.fairness import (
    demographic_parity_difference,
    equalized_odds_difference,
)

logger = logging.getLogger(__name__)


class FairnessEvaluator:
    """Compute fairness metrics from model predictions + demographics.

    Metrics per attribute:
    - Per-group AUROC
    - Worst-group AUROC
    - Demographic parity difference
    - Equalized odds difference

    Example:
        >>> evaluator = FairnessEvaluator(static_df)
        >>> report = evaluator.evaluate(predictions, labels, stay_ids)
        >>> evaluator.print_report(report)
    """

    AGE_BINS = [(18, 44), (45, 64), (65, 79), (80, float("inf"))]
    AGE_LABELS = ["18-44", "45-64", "65-79", "80+"]

    def __init__(
        self,
        static_df: pl.DataFrame,
        protected_attributes: Optional[List[str]] = None,
        min_subgroup_size: int = 50,
        task_type: str = "binary",
    ) -> None:
        """Initialize FairnessEvaluator.

        Args:
            static_df: Polars DataFrame with static patient data.
                Must contain 'stay_id' column. May contain 'gender', 'age', 'race'.
            protected_attributes: List of attributes to evaluate.
                Defaults to ["gender", "age_group"].
            min_subgroup_size: Minimum samples for a subgroup to be included.
            task_type: Task type ("binary", "multiclass", "regression").
                Determines which per-group metrics are computed.
        """
        self.static_df = static_df
        self.protected_attributes = protected_attributes or ["gender", "age_group"]
        self.min_subgroup_size = min_subgroup_size
        self.task_type = task_type
        self._available_attributes = self._detect_available_attributes()

    def _detect_available_attributes(self) -> List[str]:
        """Auto-detect which attributes are available in the data.

        Returns:
            List of available attribute names from the requested attributes.
        """
        available = []
        if "gender" in self.static_df.columns:
            available.append("gender")
        if "age" in self.static_df.columns:
            available.append("age_group")  # Derived from age
        if "race" in self.static_df.columns:
            available.append("race")
        return [a for a in self.protected_attributes if a in available]

    def _bin_age(self, ages: torch.Tensor) -> torch.Tensor:
        """Bin continuous age into groups.

        Args:
            ages: Tensor of ages.

        Returns:
            Tensor of group indices (0-3).
        """
        groups = torch.zeros_like(ages, dtype=torch.long)
        for i, (low, high) in enumerate(self.AGE_BINS):
            groups[(ages >= low) & (ages <= high)] = i
        return groups

    def _encode_attribute(
        self,
        stay_ids: List[int],
        attribute: str,
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        """Encode attribute as integer group IDs.

        Args:
            stay_ids: List of stay IDs to look up.
            attribute: Attribute name ("gender", "age_group", "race").

        Returns:
            Tuple of (group_ids tensor, mapping from int -> group name).
        """
        # Build stay_id -> row lookup
        static_dict = {row["stay_id"]: row for row in self.static_df.to_dicts()}

        if attribute == "age_group":
            ages = []
            for sid in stay_ids:
                row = static_dict.get(sid, {})
                age = row.get("age")
                ages.append(float(age) if age is not None else -1.0)

            ages_tensor = torch.tensor(ages)
            group_ids = self._bin_age(ages_tensor)
            # Mark missing ages as -1
            group_ids[ages_tensor < 0] = -1
            group_names = {i: label for i, label in enumerate(self.AGE_LABELS)}
            group_names[-1] = "unknown"
            return group_ids, group_names

        elif attribute == "gender":
            # Map gender values to integers
            unique_vals = set()
            raw_vals = []
            for sid in stay_ids:
                row = static_dict.get(sid, {})
                val = row.get("gender")
                raw_vals.append(val)
                if val is not None:
                    unique_vals.add(val)

            val_to_id = {v: i for i, v in enumerate(sorted(unique_vals))}
            group_ids = torch.tensor(
                [val_to_id.get(v, -1) if v is not None else -1 for v in raw_vals], dtype=torch.long
            )
            group_names = {i: str(v) for v, i in val_to_id.items()}
            group_names[-1] = "unknown"
            return group_ids, group_names

        elif attribute == "race":
            unique_vals = set()
            raw_vals = []
            for sid in stay_ids:
                row = static_dict.get(sid, {})
                val = row.get("race")
                raw_vals.append(val)
                if val is not None:
                    unique_vals.add(val)

            val_to_id = {v: i for i, v in enumerate(sorted(unique_vals))}
            group_ids = torch.tensor(
                [val_to_id.get(v, -1) if v is not None else -1 for v in raw_vals], dtype=torch.long
            )
            group_names = {i: str(v) for v, i in val_to_id.items()}
            group_names[-1] = "unknown"
            return group_ids, group_names

        else:
            raise ValueError(f"Unknown attribute: {attribute}")

    def evaluate(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        stay_ids: List[int],
    ) -> Dict[str, Any]:
        """Compute fairness report across all available attributes.

        For binary tasks: per-group AUROC, demographic parity, equalized odds.
        For regression tasks: per-group MSE, MAE, R2.

        Args:
            predictions: Model predictions, shape (N,).
                Probabilities for binary tasks, continuous values for regression.
            labels: Ground truth labels, shape (N,).
            stay_ids: List of stay IDs corresponding to each sample.

        Returns:
            Structured dict with per-attribute results, loggable to W&B.
        """
        report: Dict[str, Any] = {}

        for attr in self._available_attributes:
            group_ids, group_names = self._encode_attribute(stay_ids, attr)

            # Get unique valid groups (exclude -1 = unknown)
            unique_groups = [g for g in group_ids.unique().tolist() if g >= 0]

            # Filter groups below min_subgroup_size
            valid_groups = []
            for g in unique_groups:
                group_mask = group_ids == g
                if group_mask.sum().item() >= self.min_subgroup_size:
                    valid_groups.append(g)

            if len(valid_groups) < 2:
                logger.warning(
                    "Attribute '%s': fewer than 2 groups with >= %d samples, skipping",
                    attr,
                    self.min_subgroup_size,
                )
                continue

            group_sizes = {group_names[g]: int((group_ids == g).sum().item()) for g in valid_groups}

            if self.task_type == "regression":
                report[attr] = self._evaluate_regression(
                    predictions, labels, group_ids, group_names, valid_groups, group_sizes
                )
            else:
                report[attr] = self._evaluate_binary(
                    predictions, labels, group_ids, group_names, valid_groups, group_sizes
                )

        return report

    def _evaluate_binary(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor,
        group_names: Dict[int, str],
        valid_groups: List[int],
        group_sizes: Dict[str, int],
    ) -> Dict[str, Any]:
        """Compute binary classification fairness metrics."""
        per_group_auroc: Dict[str, float] = {}
        auroc_values = []

        for g in valid_groups:
            group_mask = group_ids == g
            g_preds = predictions[group_mask]
            g_labels = labels[group_mask].long()

            if g_labels.unique().numel() < 2:
                per_group_auroc[group_names[g]] = float("nan")
                continue

            auroc_metric = AUROC(task="binary")
            auroc_val = auroc_metric(g_preds, g_labels).item()
            per_group_auroc[group_names[g]] = auroc_val
            auroc_values.append(auroc_val)

        worst_group_auroc = min(auroc_values) if auroc_values else float("nan")

        valid_mask = torch.zeros_like(group_ids, dtype=torch.bool)
        for g in valid_groups:
            valid_mask |= group_ids == g

        dp_diff = demographic_parity_difference(predictions[valid_mask], group_ids[valid_mask])
        eo_diff = equalized_odds_difference(
            labels[valid_mask], predictions[valid_mask], group_ids[valid_mask]
        )

        return {
            "per_group_auroc": per_group_auroc,
            "worst_group_auroc": worst_group_auroc,
            "demographic_parity_diff": dp_diff,
            "equalized_odds_diff": eo_diff,
            "n_valid_groups": len(valid_groups),
            "group_sizes": group_sizes,
        }

    def _evaluate_regression(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor,
        group_names: Dict[int, str],
        valid_groups: List[int],
        group_sizes: Dict[str, int],
    ) -> Dict[str, Any]:
        """Compute regression fairness metrics (per-group MSE, MAE, R2)."""
        per_group_mse: Dict[str, float] = {}
        per_group_mae: Dict[str, float] = {}
        per_group_r2: Dict[str, float] = {}
        mse_values = []

        for g in valid_groups:
            group_mask = group_ids == g
            g_preds = predictions[group_mask].float()
            g_labels = labels[group_mask].float()

            residuals = g_preds - g_labels
            mse = (residuals**2).mean().item()
            mae = residuals.abs().mean().item()

            # R2 = 1 - SS_res / SS_tot
            ss_res = (residuals**2).sum().item()
            ss_tot = ((g_labels - g_labels.mean()) ** 2).sum().item()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

            name = group_names[g]
            per_group_mse[name] = mse
            per_group_mae[name] = mae
            per_group_r2[name] = r2
            mse_values.append(mse)

        worst_group_mse = max(mse_values) if mse_values else float("nan")

        return {
            "per_group_mse": per_group_mse,
            "per_group_mae": per_group_mae,
            "per_group_r2": per_group_r2,
            "worst_group_mse": worst_group_mse,
            "n_valid_groups": len(valid_groups),
            "group_sizes": group_sizes,
        }

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted fairness report to console.

        Args:
            report: Report dictionary from evaluate().
        """
        print("\n" + "=" * 60)
        print("Fairness Evaluation Report")
        print("=" * 60)

        for attr, metrics in report.items():
            print(f"\n  Attribute: {attr}")
            print("  " + "-" * 40)

            if "per_group_auroc" in metrics:
                # Binary classification report
                print("  Per-group AUROC:")
                for group, auroc in metrics["per_group_auroc"].items():
                    size = metrics["group_sizes"].get(group, "?")
                    if isinstance(auroc, float) and auroc != auroc:  # NaN check
                        print(f"    {group} (n={size}): N/A (single class)")
                    else:
                        print(f"    {group} (n={size}): {auroc:.4f}")

                wg = metrics["worst_group_auroc"]
                if isinstance(wg, float) and wg == wg:  # Not NaN
                    print(f"  Worst-group AUROC: {wg:.4f}")
                else:
                    print("  Worst-group AUROC: N/A")
                print(f"  Demographic parity diff: {metrics['demographic_parity_diff']:.4f}")
                eo = metrics["equalized_odds_diff"]
                if isinstance(eo, float) and eo == eo:
                    print(f"  Equalized odds diff: {eo:.4f}")
                else:
                    print("  Equalized odds diff: N/A (undefined for some groups)")

            elif "per_group_mse" in metrics:
                # Regression report
                print("  Per-group MSE / MAE / R2:")
                for group in metrics["per_group_mse"]:
                    size = metrics["group_sizes"].get(group, "?")
                    mse = metrics["per_group_mse"][group]
                    mae = metrics["per_group_mae"][group]
                    r2 = metrics["per_group_r2"][group]
                    print(f"    {group} (n={size}): MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

                wg = metrics["worst_group_mse"]
                if isinstance(wg, float) and wg == wg:
                    print(f"  Worst-group MSE: {wg:.4f}")
                else:
                    print("  Worst-group MSE: N/A")

        print("\n" + "=" * 60)
