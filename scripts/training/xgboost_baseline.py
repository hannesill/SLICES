"""XGBoost baseline for ICU prediction tasks.

Non-neural baseline that extracts hand-crafted tabular features from ICU time
series and trains XGBoost. Reuses ICUDataModule for data loading and splits,
and logs metrics to W&B in the same format as neural baselines.

Example usage:
    uv run python scripts/training/xgboost_baseline.py dataset=miiv
    uv run python scripts/training/xgboost_baseline.py dataset=eicu tasks=mortality_hospital
    uv run python scripts/training/xgboost_baseline.py dataset=miiv logging.use_wandb=false
"""

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from slices.data.datamodule import ICUDataModule
from slices.eval.inference import extract_tabular_features
from xgboost import XGBClassifier, XGBRegressor


def _xgboost_eval_metric(task_type: str) -> str:
    """Return the eval metric aligned with the benchmark primary metric."""
    return "mae" if task_type == "regression" else "aucpr"


def _add_wandb_tag(tags: list[str], tag: str | None) -> None:
    """Append a W&B tag once, ignoring unset values."""
    if not tag:
        return
    if tag not in tags:
        tags.append(tag)


def _resolve_scale_pos_weight(scale_pos_weight, y_train) -> float | None:
    """Resolve XGBoost's native positive-class weight.

    ``balanced`` intentionally follows XGBoost's standard binary weighting:
    ``n_negative / n_positive``. This differs from the neural sqrt-balanced
    class weights, whose effective positive/negative ratio is milder.
    """
    if scale_pos_weight is None:
        return None
    if scale_pos_weight == "balanced":
        y_train = np.asarray(y_train)
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        return n_neg / max(n_pos, 1.0)
    return float(scale_pos_weight)


def _binary_ece(y_true, y_pred_proba, n_bins: int = 15) -> float:
    """Compute L1 expected calibration error with uniform probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    if y_true.size == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_pred_proba, bin_edges[1:-1], right=False)

    ece = 0.0
    for bin_idx in range(n_bins):
        in_bin = bin_ids == bin_idx
        if not np.any(in_bin):
            continue
        bin_weight = in_bin.mean()
        confidence = y_pred_proba[in_bin].mean()
        accuracy = y_true[in_bin].mean()
        ece += bin_weight * abs(confidence - accuracy)
    return float(ece)


def _build_wandb_tags(cfg: DictConfig) -> list[str] | None:
    """Build W&B tags in parity with neural training runs."""
    tags = list(cfg.logging.get("wandb_tags", []))
    if cfg.get("experiment_class") is not None:
        _add_wandb_tag(tags, f"experiment_class:{cfg.experiment_class}")
    if cfg.get("experiment_subtype") is not None:
        _add_wandb_tag(tags, f"experiment_subtype:{cfg.experiment_subtype}")
    if cfg.get("revision") is not None:
        _add_wandb_tag(tags, f"revision:{cfg.revision}")
    if cfg.get("rerun_reason") is not None:
        tag = f"rerun-reason:{cfg.rerun_reason}"
        if len(tag) > 64:
            tag = tag[:61] + "..."
        _add_wandb_tag(tags, tag)
    if cfg.get("phase") is not None:
        _add_wandb_tag(tags, f"phase:{cfg.phase}")
    if cfg.get("dataset") is not None:
        _add_wandb_tag(tags, f"dataset:{cfg.dataset}")
    if cfg.get("paradigm") is not None:
        _add_wandb_tag(tags, f"paradigm:{cfg.paradigm}")
    task_name = cfg.get("task", {}).get("task_name")
    if task_name is not None:
        _add_wandb_tag(tags, f"task:{task_name}")
    if cfg.get("seed") is not None:
        _add_wandb_tag(tags, f"seed:{cfg.seed}")
    if cfg.get("protocol") is not None:
        _add_wandb_tag(tags, f"protocol:{cfg.protocol}")
    if cfg.get("label_fraction", 1.0) < 1.0:
        _add_wandb_tag(tags, f"label_fraction:{cfg.label_fraction}")
        _add_wandb_tag(tags, "ablation:label-efficiency")
    return tags or None


@hydra.main(version_base=None, config_path="../../configs", config_name="xgboost")
def main(cfg: DictConfig) -> None:
    """Train XGBoost baseline."""
    print("=" * 80)
    print("XGBoost Baseline")
    print("=" * 80)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # =========================================================================
    # 1. Setup DataModule
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. Setting up DataModule")
    print("=" * 80)

    from slices.training.utils import (
        report_and_validate_train_label_support,
        validate_data_prerequisites,
    )

    task_name = cfg.task.get("task_name", "mortality_24h")
    task_type = cfg.task.get("task_type", "binary")

    # Validate data prerequisites including label freshness
    validate_data_prerequisites(
        cfg.data.processed_dir,
        cfg.dataset,
        task_names=[task_name],
        task_configs=[cfg.task],
    )

    datamodule = ICUDataModule(
        processed_dir=cfg.data.processed_dir,
        task_name=task_name,
        batch_size=cfg.training.batch_size,
        num_workers=0,  # feature extraction is single-threaded
        seed=cfg.seed,
        label_fraction=cfg.get("label_fraction", 1.0),
    )
    datamodule.setup()

    print(f"\n  Task: {task_name} ({task_type})")
    print(f"  Train: {len(datamodule.train_indices)} stays")
    print(f"  Val:   {len(datamodule.val_indices)} stays")
    print(f"  Test:  {len(datamodule.test_indices)} stays")

    report_and_validate_train_label_support(
        datamodule=datamodule,
        task_name=task_name,
        task_type=task_type,
        dataset=cfg.dataset,
        seed=cfg.seed,
        label_fraction=cfg.get("label_fraction", 1.0),
        min_train_positives=cfg.get("min_train_positives", 3),
    )

    # =========================================================================
    # 2. Extract Tabular Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Extracting tabular features")
    print("=" * 80)

    X_train, y_train = extract_tabular_features(datamodule.dataset, datamodule.train_indices)
    X_val, y_val = extract_tabular_features(datamodule.dataset, datamodule.val_indices)
    X_test, y_test = extract_tabular_features(datamodule.dataset, datamodule.test_indices)

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # =========================================================================
    # 3. Train XGBoost
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. Training XGBoost")
    print("=" * 80)

    xgb_cfg = cfg.xgboost
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 20)
    n_jobs = xgb_cfg.get("n_jobs", 4)
    if n_jobs is None:
        n_jobs = 4
    common_params = {
        "n_estimators": xgb_cfg.n_estimators,
        "max_depth": xgb_cfg.max_depth,
        "learning_rate": xgb_cfg.learning_rate,
        "subsample": xgb_cfg.subsample,
        "colsample_bytree": xgb_cfg.colsample_bytree,
        "min_child_weight": xgb_cfg.min_child_weight,
        "early_stopping_rounds": early_stopping_rounds,
        "random_state": cfg.seed,
        "n_jobs": n_jobs,
    }

    resolved_scale_pos_weight = None
    if task_type == "regression":
        model = XGBRegressor(
            **common_params,
            eval_metric=_xgboost_eval_metric(task_type),
        )
    else:
        requested_scale_pos_weight = xgb_cfg.get("scale_pos_weight", None)
        resolved_scale_pos_weight = _resolve_scale_pos_weight(requested_scale_pos_weight, y_train)
        if resolved_scale_pos_weight is not None:
            was_struct = OmegaConf.is_struct(cfg)
            OmegaConf.set_struct(cfg, False)
            cfg.xgboost.scale_pos_weight = resolved_scale_pos_weight
            OmegaConf.set_struct(cfg, was_struct)
            if requested_scale_pos_weight == "balanced":
                print(
                    "  scale_pos_weight (native balanced n_neg/n_pos): "
                    f"{resolved_scale_pos_weight:.6g}"
                )
            else:
                print(f"  scale_pos_weight: {resolved_scale_pos_weight:.6g}")
        model = XGBClassifier(
            **common_params,
            scale_pos_weight=resolved_scale_pos_weight,
            eval_metric=_xgboost_eval_metric(task_type),
        )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # =========================================================================
    # 4. Evaluate
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. Evaluating on test set")
    print("=" * 80)

    metrics = {}

    if task_type == "regression":
        y_pred = model.predict(X_test)
        metrics["test/mse"] = mean_squared_error(y_test, y_pred)
        metrics["test/mae"] = mean_absolute_error(y_test, y_pred)
        metrics["test/r2"] = r2_score(y_test, y_pred)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        metrics["test/auroc"] = roc_auc_score(y_test, y_pred_proba)
        metrics["test/auprc"] = average_precision_score(y_test, y_pred_proba)
        metrics["test/accuracy"] = accuracy_score(y_test, y_pred_class)
        metrics["test/brier_score"] = brier_score_loss(y_test, y_pred_proba)
        metrics["test/ece"] = _binary_ece(y_test, y_pred_proba)
        if resolved_scale_pos_weight is not None:
            metrics["xgboost/scale_pos_weight"] = resolved_scale_pos_weight

    print("\n  Test Results:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

    # =========================================================================
    # 5. Optional Fairness Evaluation
    # =========================================================================
    fairness_cfg = cfg.get("eval", {}).get("fairness", {})
    fairness_report = None
    if fairness_cfg.get("enabled", False):
        print("\n" + "=" * 80)
        print("5. Fairness Evaluation")
        print("=" * 80)

        from slices.eval.fairness_evaluator import FairnessEvaluator, flatten_fairness_report

        # Get stay_ids for test set
        test_stay_ids = [datamodule.dataset.stay_ids[i] for i in datamodule.test_indices]

        evaluator = FairnessEvaluator(
            static_df=datamodule.dataset.static_df,
            protected_attributes=list(
                fairness_cfg.get("protected_attributes", ["gender", "age_group"])
            ),
            min_subgroup_size=fairness_cfg.get("min_subgroup_size", 50),
            task_type=task_type,
            dataset_name=getattr(getattr(datamodule, "processed_dir", None), "name", None),
        )

        if task_type == "regression":
            predictions_tensor = torch.tensor(y_pred, dtype=torch.float32)
        else:
            predictions_tensor = torch.tensor(y_pred_proba, dtype=torch.float32)
        labels_tensor = torch.tensor(y_test, dtype=torch.float32)

        fairness_report = evaluator.evaluate(predictions_tensor, labels_tensor, test_stay_ids)
        evaluator.print_report(fairness_report)

    # =========================================================================
    # 6. Save Model
    # =========================================================================
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model.json"
    model.save_model(str(model_path))
    print(f"\n  Model saved to: {model_path}")

    # =========================================================================
    # 7. Log to W&B
    # =========================================================================
    if cfg.logging.get("use_wandb", False):
        import wandb

        run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity", None),
            name=cfg.logging.get("run_name", f"xgboost_{cfg.dataset}_{task_name}"),
            group=cfg.logging.get("wandb_group", f"xgboost_{cfg.dataset}_{task_name}"),
            tags=_build_wandb_tags(cfg),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        run.summary.update(metrics)
        if fairness_report:
            run.summary.update(flatten_fairness_report(fairness_report))
        wandb.finish()

    print("\n" + "=" * 80)
    print("XGBoost Baseline Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
