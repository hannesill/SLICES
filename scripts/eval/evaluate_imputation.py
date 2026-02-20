"""Evaluate SSL encoder quality via imputation reconstruction.

Loads a pretrained encoder checkpoint, applies masking strategies to held-out
test data, and measures reconstruction quality (NRMSE per feature, MAE overall).

Example usage:
    # Evaluate MAE checkpoint with all strategies
    uv run python scripts/eval/evaluate_imputation.py \
        checkpoint=outputs/pretrain/encoder.pt \
        data.processed_dir=data/processed/mimic-iv

    # Evaluate full pretrain checkpoint (MAE with decoder)
    uv run python scripts/eval/evaluate_imputation.py \
        pretrain_checkpoint=outputs/pretrain/ssl-last.ckpt \
        data.processed_dir=data/processed/mimic-iv

    # Single strategy
    uv run python scripts/eval/evaluate_imputation.py \
        checkpoint=outputs/encoder.pt \
        masking.strategies=[random]
"""

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from slices.data.datamodule import ICUDataModule
from slices.eval.imputation import ImputationEvaluator


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluate_imputation",
)
def main(cfg: DictConfig) -> None:
    """Run imputation evaluation.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("Imputation Evaluation")
    print("=" * 80)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # =========================================================================
    # 1. Setup DataModule (no task labels needed)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. Setting up DataModule")
    print("=" * 80)

    datamodule = ICUDataModule(
        processed_dir=cfg.data.processed_dir,
        task_name=None,  # No labels needed for imputation
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.data.get("num_workers", 4),
        seed=cfg.seed,
    )
    datamodule.setup()

    d_input = datamodule.get_feature_dim()
    feature_names = datamodule.dataset.get_feature_names()

    print(f"\n  Feature dimension: {d_input}")
    print(f"  Sequence length: {datamodule.get_seq_length()}")

    # =========================================================================
    # 2. Load Encoder
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Loading Encoder")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrain_ckpt = cfg.get("pretrain_checkpoint", None)
    encoder_ckpt = cfg.get("checkpoint", None)

    if pretrain_ckpt:
        print(f"\n  Loading MAE checkpoint: {pretrain_ckpt}")
        evaluator = ImputationEvaluator.from_mae_checkpoint(
            pretrain_ckpt, device=device, feature_names=feature_names
        )
    elif encoder_ckpt:
        print(f"\n  Loading encoder checkpoint: {encoder_ckpt}")
        evaluator = ImputationEvaluator.from_encoder_checkpoint(
            encoder_ckpt, d_input=d_input, device=device, feature_names=feature_names
        )
    else:
        raise ValueError(
            "Must provide either 'checkpoint' (encoder.pt) or "
            "'pretrain_checkpoint' (full .ckpt file)"
        )

    # =========================================================================
    # 3. Train decoder (if not MAE)
    # =========================================================================
    if not pretrain_ckpt:
        print("\n" + "=" * 80)
        print("3. Training Reconstruction Decoder")
        print("=" * 80)

        recon_cfg = cfg.get("reconstruction_head", {})
        evaluator.train_decoder(
            dataloader=datamodule.train_dataloader(),
            max_epochs=recon_cfg.get("max_epochs", 10),
            lr=recon_cfg.get("lr", 1e-3),
        )

    # =========================================================================
    # 4. Evaluate
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. Running Imputation Evaluation")
    print("=" * 80)

    strategies = cfg.masking.strategies
    mask_ratio = cfg.masking.mask_ratio
    test_loader = datamodule.test_dataloader()

    # Setup W&B logger if configured
    logger = None
    if cfg.logging.get("use_wandb", False):
        try:
            import wandb

            wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.get("wandb_entity", None),
                config=OmegaConf.to_container(cfg, resolve=True),
                job_type="imputation_eval",
            )
            logger = wandb
        except ImportError:
            print("  W&B not available, skipping logging")

    all_results = {}
    for strategy in strategies:
        print(f"\n  Strategy: {strategy} (mask_ratio={mask_ratio})")
        print("  " + "-" * 40)

        results = evaluator.evaluate(test_loader, mask_strategy=strategy, mask_ratio=mask_ratio)
        all_results[strategy] = results

        print(f"  NRMSE overall: {results['nrmse_overall']:.4f}")
        print(f"  MAE overall:   {results['mae_overall']:.4f}")

        # Print per-feature NRMSE (top 5 worst)
        sorted_features = sorted(
            results["nrmse_per_feature"].items(), key=lambda x: x[1], reverse=True
        )
        print("\n  Worst 5 features (NRMSE):")
        for name, nrmse in sorted_features[:5]:
            print(f"    {name}: {nrmse:.4f}")

        # Log to W&B
        if logger:
            for name, nrmse in results["nrmse_per_feature"].items():
                logger.log({f"imputation/{strategy}/nrmse/{name}": nrmse})
            logger.log(
                {
                    f"imputation/{strategy}/nrmse_overall": results["nrmse_overall"],
                    f"imputation/{strategy}/mae_overall": results["mae_overall"],
                }
            )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Imputation Evaluation Summary")
    print("=" * 80)

    print(f"\n  {'Strategy':<20} {'NRMSE':>10} {'MAE':>10}")
    print("  " + "-" * 42)
    for strategy, results in all_results.items():
        print(
            f"  {strategy:<20} {results['nrmse_overall']:>10.4f} "
            f"{results['mae_overall']:>10.4f}"
        )

    if logger:
        wandb.finish()

    print(f"\n  Output directory: {cfg.output_dir}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
