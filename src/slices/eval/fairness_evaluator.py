"""Fairness evaluation framework for clinical prediction models.

Computes per-group AUROC, worst-group AUROC, demographic parity, and
equalized odds across protected attributes (gender, age group, race).

Race handling follows the thesis plan:
- race/ethnicity is evaluated only on MIMIC-IV rows
- raw race strings are mapped into the canonical five-bin schema
  (White, Black, Hispanic, Asian, Other)
- subgroup inclusion thresholds are enforced on unique patients, not stays
"""

import logging
import re
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAveragePrecision

from slices.eval.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)

logger = logging.getLogger(__name__)


def flatten_fairness_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested fairness report into W&B-safe ``fairness/*`` keys."""

    flat: dict[str, Any] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, Real) and not isinstance(value, bool):
            flat[prefix] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                _flatten(f"{prefix}/{sub_key}", sub_value)

    for attr, metrics in report.items():
        _flatten(f"fairness/{attr}", metrics)

    return flat


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
    RACE_LABELS = ["White", "Black", "Hispanic", "Asian", "Other"]

    def __init__(
        self,
        static_df: pl.DataFrame,
        protected_attributes: Optional[List[str]] = None,
        min_subgroup_size: int = 50,
        task_type: str = "binary",
        dataset_name: Optional[str] = None,
    ) -> None:
        """Initialize FairnessEvaluator.

        Args:
            static_df: Polars DataFrame with static patient data.
                Must contain 'stay_id' column. May contain 'gender', 'age', 'race'.
            protected_attributes: List of attributes to evaluate.
                Defaults to ["gender", "age_group"].
            min_subgroup_size: Minimum patients for a subgroup to be included.
            task_type: Task type ("binary", "multiclass", "regression").
                Determines which per-group metrics are computed.
            dataset_name: Dataset identifier ("miiv", "eicu", "combined", ...).
                Used for dataset-specific subgroup rules such as MIMIC-only race
                analysis. When omitted, the evaluator falls back to columns
                available in ``static_df`` for backwards compatibility.
        """
        self.static_df = static_df
        self.protected_attributes = protected_attributes or ["gender", "age_group"]
        self.min_subgroup_size = min_subgroup_size
        self.task_type = task_type
        self.dataset_name = dataset_name.lower() if dataset_name is not None else None
        self._static_dict = {row["stay_id"]: row for row in self.static_df.to_dicts()}
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
        if "race" in self.static_df.columns and self._race_analysis_available():
            available.append("race")
        return [a for a in self.protected_attributes if a in available]

    def _race_analysis_available(self) -> bool:
        """Return whether race fairness should be evaluated for this dataset."""
        if self.dataset_name == "eicu":
            return False
        if self.dataset_name == "combined":
            if "source_dataset" not in self.static_df.columns:
                return False
            source_values = {
                str(value).strip().lower()
                for value in self.static_df["source_dataset"].drop_nulls().to_list()
            }
            return any("miiv" in value or "mimic" in value for value in source_values)
        return True

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
    ) -> Tuple[torch.Tensor, Dict[int, str], List[Any]]:
        """Encode attribute as integer group IDs.

        Args:
            stay_ids: List of stay IDs to look up.
            attribute: Attribute name ("gender", "age_group", "race").

        Returns:
            Tuple of:
            - group_ids tensor aligned to ``stay_ids``
            - mapping from int -> group name
            - patient_ids aligned to ``stay_ids`` for patient-threshold logic
        """
        patient_ids = []
        rows = []
        for sid in stay_ids:
            row = self._static_dict.get(sid, {})
            rows.append(row)
            patient_ids.append(row.get("patient_id", sid))

        if attribute == "age_group":
            ages = []
            for row in rows:
                age = row.get("age")
                ages.append(float(age) if age is not None else -1.0)

            ages_tensor = torch.tensor(ages)
            group_ids = self._bin_age(ages_tensor)
            # Mark missing ages as -1
            group_ids[ages_tensor < 0] = -1
            group_names = {i: label for i, label in enumerate(self.AGE_LABELS)}
            group_names[-1] = "unknown"
            return group_ids, group_names, patient_ids

        elif attribute == "gender":
            # Map gender values to integers
            unique_vals = set()
            raw_vals = []
            for row in rows:
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
            return group_ids, group_names, patient_ids

        elif attribute == "race":
            canonical_to_id = {label: i for i, label in enumerate(self.RACE_LABELS)}
            canonical_vals = []
            for row in rows:
                canonical_vals.append(self._canonicalize_race(row))

            group_ids = torch.tensor(
                [
                    canonical_to_id.get(value, -1) if value is not None else -1
                    for value in canonical_vals
                ],
                dtype=torch.long,
            )
            group_names = {i: label for label, i in canonical_to_id.items()}
            group_names[-1] = "unknown"
            return group_ids, group_names, patient_ids

        else:
            raise ValueError(f"Unknown attribute: {attribute}")

    def _canonicalize_race(self, row: Dict[str, Any]) -> Optional[str]:
        """Map raw race strings into the planned five-bin schema."""
        if not self._row_is_miiv_for_race(row):
            return None

        raw_value = row.get("race")
        if raw_value is None:
            return None

        text = str(raw_value).strip()
        if not text:
            return None

        normalized = re.sub(r"[/_\\-]+", " ", text.upper())
        normalized = re.sub(r"\s+", " ", normalized).strip()

        missing_markers = ("UNKNOWN", "DECLIN", "UNABLE", "PATIENT REFUSED", "NOT SPECIFIED")
        if any(marker in normalized for marker in missing_markers):
            return None
        if "HISPANIC" in normalized or "LATINO" in normalized:
            return "Hispanic"
        if "BLACK" in normalized or "AFRICAN" in normalized:
            return "Black"
        if "ASIAN" in normalized:
            return "Asian"
        if "WHITE" in normalized:
            return "White"
        return "Other"

    def _row_is_miiv_for_race(self, row: Dict[str, Any]) -> bool:
        """Return whether a row should participate in race fairness analysis."""
        if self.dataset_name == "eicu":
            return False
        source_dataset = row.get("source_dataset")
        if source_dataset is not None:
            source_value = str(source_dataset).strip().lower()
            return "miiv" in source_value or "mimic" in source_value
        if self.dataset_name == "combined":
            return False
        return True

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
            group_ids, group_names, patient_ids = self._encode_attribute(stay_ids, attr)

            # Get unique valid groups (exclude -1 = unknown)
            unique_groups = [g for g in group_ids.unique().tolist() if g >= 0]

            # Filter groups below the patient-count threshold from the thesis plan
            valid_groups = []
            group_sizes = {}
            group_sample_sizes = {}
            for g in unique_groups:
                group_mask = group_ids == g
                patient_count = len(
                    {
                        patient_ids[i]
                        for i, is_member in enumerate(group_mask.tolist())
                        if is_member and patient_ids[i] is not None
                    }
                )
                sample_count = int(group_mask.sum().item())
                group_sizes[group_names[g]] = patient_count
                group_sample_sizes[group_names[g]] = sample_count
                if patient_count >= self.min_subgroup_size:
                    valid_groups.append(g)

            if len(valid_groups) < 2:
                logger.warning(
                    "Attribute '%s': fewer than 2 groups with >= %d patients, skipping",
                    attr,
                    self.min_subgroup_size,
                )
                continue

            valid_group_sizes = {group_names[g]: group_sizes[group_names[g]] for g in valid_groups}
            valid_group_sample_sizes = {
                group_names[g]: group_sample_sizes[group_names[g]] for g in valid_groups
            }

            if self.task_type == "regression":
                report[attr] = self._evaluate_regression(
                    predictions,
                    labels,
                    group_ids,
                    group_names,
                    valid_groups,
                    valid_group_sizes,
                    valid_group_sample_sizes,
                )
            else:
                report[attr] = self._evaluate_binary(
                    predictions,
                    labels,
                    group_ids,
                    group_names,
                    valid_groups,
                    valid_group_sizes,
                    valid_group_sample_sizes,
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
        group_sample_sizes: Dict[str, int],
    ) -> Dict[str, Any]:
        """Compute binary classification fairness metrics."""
        per_group_auroc: Dict[str, float] = {}
        per_group_auprc: Dict[str, float] = {}
        auroc_values = []
        auprc_values = []

        for g in valid_groups:
            group_mask = group_ids == g
            g_preds = predictions[group_mask]
            g_labels = labels[group_mask].long()

            if g_labels.unique().numel() < 2:
                per_group_auroc[group_names[g]] = float("nan")
                per_group_auprc[group_names[g]] = float("nan")
                continue

            auroc_metric = AUROC(task="binary")
            auroc_val = auroc_metric(g_preds, g_labels).item()
            per_group_auroc[group_names[g]] = auroc_val
            auroc_values.append(auroc_val)

            auprc_metric = BinaryAveragePrecision()
            auprc_val = auprc_metric(g_preds, g_labels).item()
            per_group_auprc[group_names[g]] = auprc_val
            auprc_values.append(auprc_val)

        worst_group_auroc = min(auroc_values) if auroc_values else float("nan")
        worst_group_auprc = min(auprc_values) if auprc_values else float("nan")
        auroc_gap = (
            (max(auroc_values) - min(auroc_values)) if len(auroc_values) >= 2 else float("nan")
        )
        auprc_gap = (
            (max(auprc_values) - min(auprc_values)) if len(auprc_values) >= 2 else float("nan")
        )

        valid_mask = torch.zeros_like(group_ids, dtype=torch.bool)
        for g in valid_groups:
            valid_mask |= group_ids == g

        dp_diff = demographic_parity_difference(predictions[valid_mask], group_ids[valid_mask])
        eo_diff = equalized_odds_difference(
            labels[valid_mask], predictions[valid_mask], group_ids[valid_mask]
        )
        di_ratio = disparate_impact_ratio(predictions[valid_mask], group_ids[valid_mask])

        return {
            "per_group_auroc": per_group_auroc,
            "per_group_auprc": per_group_auprc,
            "worst_group_auroc": worst_group_auroc,
            "worst_group_auprc": worst_group_auprc,
            "auroc_gap": auroc_gap,
            "auprc_gap": auprc_gap,
            "demographic_parity_diff": dp_diff,
            "equalized_odds_diff": eo_diff,
            "disparate_impact_ratio": di_ratio,
            "n_valid_groups": len(valid_groups),
            "group_sizes": group_sizes,
            "group_sample_sizes": group_sample_sizes,
        }

    def _evaluate_regression(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor,
        group_names: Dict[int, str],
        valid_groups: List[int],
        group_sizes: Dict[str, int],
        group_sample_sizes: Dict[str, int],
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
            "group_sample_sizes": group_sample_sizes,
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
                    patient_size = metrics["group_sizes"].get(group, "?")
                    sample_size = metrics.get("group_sample_sizes", {}).get(group, "?")
                    if isinstance(auroc, float) and auroc != auroc:  # NaN check
                        print(
                            f"    {group} (patients={patient_size}, stays={sample_size}): "
                            "N/A (single class)"
                        )
                    else:
                        print(
                            f"    {group} (patients={patient_size}, stays={sample_size}): "
                            f"{auroc:.4f}"
                        )

                print("  Per-group AUPRC:")
                for group, auprc in metrics.get("per_group_auprc", {}).items():
                    patient_size = metrics["group_sizes"].get(group, "?")
                    sample_size = metrics.get("group_sample_sizes", {}).get(group, "?")
                    if isinstance(auprc, float) and auprc != auprc:
                        print(
                            f"    {group} (patients={patient_size}, stays={sample_size}): "
                            "N/A (single class)"
                        )
                    else:
                        print(
                            f"    {group} (patients={patient_size}, stays={sample_size}): "
                            f"{auprc:.4f}"
                        )

                wg = metrics["worst_group_auroc"]
                if isinstance(wg, float) and wg == wg:  # Not NaN
                    print(f"  Worst-group AUROC: {wg:.4f}")
                else:
                    print("  Worst-group AUROC: N/A")
                wg_auprc = metrics.get("worst_group_auprc", float("nan"))
                if isinstance(wg_auprc, float) and wg_auprc == wg_auprc:
                    print(f"  Worst-group AUPRC: {wg_auprc:.4f}")
                else:
                    print("  Worst-group AUPRC: N/A")
                auroc_gap = metrics.get("auroc_gap", float("nan"))
                if isinstance(auroc_gap, float) and auroc_gap == auroc_gap:
                    print(f"  AUROC gap (max-min): {auroc_gap:.4f}")
                auprc_gap = metrics.get("auprc_gap", float("nan"))
                if isinstance(auprc_gap, float) and auprc_gap == auprc_gap:
                    print(f"  AUPRC gap (max-min): {auprc_gap:.4f}")
                print(f"  Demographic parity diff: {metrics['demographic_parity_diff']:.4f}")
                eo = metrics["equalized_odds_diff"]
                if isinstance(eo, float) and eo == eo:
                    print(f"  Equalized odds diff: {eo:.4f}")
                else:
                    print("  Equalized odds diff: N/A (undefined for some groups)")
                di = metrics.get("disparate_impact_ratio", float("nan"))
                if isinstance(di, float) and di == di:
                    print(f"  Disparate impact ratio: {di:.4f}")
                    if di < 0.8:
                        print("    -> Below 0.8 threshold (adverse impact)")
                else:
                    print("  Disparate impact ratio: N/A")

            elif "per_group_mse" in metrics:
                # Regression report
                print("  Per-group MSE / MAE / R2:")
                for group in metrics["per_group_mse"]:
                    patient_size = metrics["group_sizes"].get(group, "?")
                    sample_size = metrics.get("group_sample_sizes", {}).get(group, "?")
                    mse = metrics["per_group_mse"][group]
                    mae = metrics["per_group_mae"][group]
                    r2 = metrics["per_group_r2"][group]
                    print(
                        f"    {group} (patients={patient_size}, stays={sample_size}): "
                        f"MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}"
                    )

                wg = metrics["worst_group_mse"]
                if isinstance(wg, float) and wg == wg:
                    print(f"  Worst-group MSE: {wg:.4f}")
                else:
                    print("  Worst-group MSE: N/A")

        print("\n" + "=" * 60)
