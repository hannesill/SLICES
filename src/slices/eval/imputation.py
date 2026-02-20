"""Imputation evaluation framework for SSL encoder quality assessment.

Evaluates reconstruction quality of SSL embeddings by measuring how well
masked values can be reconstructed. This is fundamentally different from
label-based tasks — it directly measures the encoder's ability to capture
temporal and feature-level patterns in the ICU time-series data.

Three masking strategies:
- random: Mask 15% of observed values uniformly at random
- feature_block: Mask entire features for the full window
- temporal_block: Mask contiguous hour blocks across all features

For MAE models: use encoder + decoder to reconstruct.
For non-MAE models: train lightweight linear decoder (d_model -> d_input).

Metrics: NRMSE per feature, MAE overall.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ImputationEvaluator:
    """Evaluate reconstruction quality of SSL embeddings.

    Three masking strategies:
    - random: mask a fraction of observed values uniformly at random
    - feature_block: mask entire features for the full window
    - temporal_block: mask contiguous hour blocks across all features

    For MAE models: use encoder + decoder to reconstruct.
    For non-MAE models: train lightweight linear decoder (d_model -> d_input).

    Metrics: NRMSE per feature, MAE overall.

    Example:
        >>> evaluator = ImputationEvaluator.from_encoder_checkpoint(
        ...     "outputs/encoder.pt", d_input=35
        ... )
        >>> evaluator.train_decoder(train_loader, max_epochs=10)
        >>> results = evaluator.evaluate(test_loader, mask_strategy="random")
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        d_input: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize ImputationEvaluator.

        Args:
            encoder: Pretrained encoder module.
            decoder: Optional decoder for reconstruction. If None and d_input
                is provided, creates a lightweight linear decoder.
            d_input: Input feature dimension. Required if decoder is None.
            feature_names: Optional list of feature names for per-feature reporting.
        """
        self.encoder = encoder
        self.feature_names = feature_names
        self.device = next(encoder.parameters()).device

        if decoder is not None:
            self.decoder = decoder
        elif d_input is not None:
            d_model = encoder.get_output_dim() if hasattr(encoder, "get_output_dim") else d_input
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_input),
            )
        else:
            raise ValueError("Either decoder or d_input must be provided.")

        self.decoder = self.decoder.to(self.device)

    @classmethod
    def from_mae_checkpoint(
        cls,
        ckpt_path: str,
        device: str = "cpu",
        feature_names: Optional[List[str]] = None,
    ) -> "ImputationEvaluator":
        """Load MAE encoder + decoder for direct reconstruction.

        Loads a full pretraining checkpoint (.ckpt) containing an MAE
        objective with encoder and decoder.

        Args:
            ckpt_path: Path to Lightning pretrain checkpoint (.ckpt).
            device: Device to load model onto.
            feature_names: Optional feature names for per-feature reporting.

        Returns:
            ImputationEvaluator with MAE encoder and decoder.
        """
        from slices.models.encoders import build_encoder
        from slices.models.pretraining.mae import MAEConfig, MAEDecoder

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]

        # Rebuild encoder from hyperparameters
        hyper_params = checkpoint["hyper_parameters"]["config"]
        encoder_cfg = dict(hyper_params["encoder"])
        encoder_name = encoder_cfg.pop("name")
        # Force pooling=none for per-timestep reconstruction
        encoder_cfg["pooling"] = "none"
        encoder = build_encoder(encoder_name, encoder_cfg)

        # Load encoder weights
        encoder_state_dict = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        encoder.load_state_dict(encoder_state_dict)

        # Rebuild decoder
        ssl_cfg = hyper_params.get("ssl", {})
        d_encoder = encoder.get_output_dim()
        d_input = encoder.config.d_input
        mae_config = MAEConfig(**{k: v for k, v in ssl_cfg.items() if k != "name"})
        decoder = MAEDecoder(d_encoder, d_input, mae_config)

        # Load decoder weights
        decoder_state_dict = {
            k.replace("ssl_objective.decoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("ssl_objective.decoder.")
        }
        decoder.load_state_dict(decoder_state_dict)

        encoder = encoder.to(device).eval()
        decoder = decoder.to(device).eval()

        return cls(
            encoder=encoder,
            decoder=decoder,
            d_input=d_input,
            feature_names=feature_names,
        )

    @classmethod
    def from_encoder_checkpoint(
        cls,
        ckpt_path: str,
        d_input: int,
        device: str = "cpu",
        feature_names: Optional[List[str]] = None,
    ) -> "ImputationEvaluator":
        """Load encoder + create simple linear reconstruction head.

        Loads an encoder checkpoint (.pt file, v3 format) and creates
        a lightweight decoder for reconstruction evaluation.

        Args:
            ckpt_path: Path to encoder checkpoint (.pt).
            d_input: Input feature dimension.
            device: Device to load model onto.
            feature_names: Optional feature names for per-feature reporting.

        Returns:
            ImputationEvaluator with encoder and linear decoder.
        """
        from slices.models.encoders import build_encoder

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

        if isinstance(checkpoint, dict) and "encoder_config" in checkpoint:
            encoder_config = dict(checkpoint["encoder_config"])
            encoder_name = encoder_config.pop("name")
            # Force pooling=none for per-timestep reconstruction
            encoder_config["pooling"] = "none"
            encoder = build_encoder(encoder_name, encoder_config)
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        else:
            raise ValueError(
                "Checkpoint does not contain encoder_config. "
                "Use a v3+ checkpoint from SSLPretrainModule.save_encoder()."
            )

        encoder = encoder.to(device).eval()

        return cls(
            encoder=encoder,
            decoder=None,
            d_input=d_input,
            feature_names=feature_names,
        )

    def apply_mask(
        self,
        timeseries: torch.Tensor,
        mask: torch.Tensor,
        strategy: str = "random",
        mask_ratio: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking strategy for imputation evaluation.

        Args:
            timeseries: Input tensor of shape (B, T, D).
            mask: Observation mask of shape (B, T, D), True=observed.
            strategy: Masking strategy ("random", "feature_block", "temporal_block").
            mask_ratio: Fraction of observed values to mask.

        Returns:
            Tuple of:
            - masked_input: Input with masked positions zeroed out (B, T, D)
            - eval_mask: Boolean mask of positions to evaluate (B, T, D),
              True = position was masked for evaluation
            - original_values: Original values at masked positions (B, T, D)
        """
        B, T, D = timeseries.shape
        eval_mask = torch.zeros_like(mask, dtype=torch.bool)

        if strategy == "random":
            # Mask random observed positions
            observed = mask.clone()
            random_vals = torch.rand_like(timeseries)
            eval_mask = observed & (random_vals < mask_ratio)

        elif strategy == "feature_block":
            # Mask entire features for the full window
            n_features_to_mask = max(1, int(D * mask_ratio))
            for b in range(B):
                feature_indices = torch.randperm(D)[:n_features_to_mask]
                for f in feature_indices:
                    eval_mask[b, :, f] = mask[b, :, f]

        elif strategy == "temporal_block":
            # Mask contiguous hour blocks across all features
            n_hours_to_mask = max(1, int(T * mask_ratio))
            for b in range(B):
                start = torch.randint(0, max(1, T - n_hours_to_mask + 1), (1,)).item()
                end = min(start + n_hours_to_mask, T)
                eval_mask[b, start:end, :] = mask[b, start:end, :]

        else:
            raise ValueError(
                f"Unknown masking strategy: '{strategy}'. "
                "Available: random, feature_block, temporal_block"
            )

        # Zero out masked positions in input
        masked_input = timeseries.clone()
        masked_input[eval_mask] = 0.0

        return masked_input, eval_mask, timeseries

    def train_decoder(
        self,
        dataloader: DataLoader,
        max_epochs: int = 10,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """Train lightweight decoder for non-MAE models.

        Freezes encoder and trains only the decoder to reconstruct
        observed values from encoder representations.

        Args:
            dataloader: DataLoader providing batches with 'timeseries' and 'mask'.
            max_epochs: Maximum training epochs.
            lr: Learning rate for decoder training.

        Returns:
            Dictionary with training loss history.
        """
        self.encoder.eval()
        self.decoder.train()

        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

        losses = []
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                timeseries = batch["timeseries"].to(self.device)
                mask = batch["mask"].to(self.device)

                with torch.no_grad():
                    encoder_out = self.encoder(timeseries, mask=mask)

                reconstruction = self.decoder(encoder_out)

                # Loss only on observed positions
                loss_mask = mask.float()
                loss = (
                    (reconstruction - timeseries) ** 2 * loss_mask
                ).sum() / loss_mask.sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            logger.info("Decoder training epoch %d/%d, loss: %.6f", epoch + 1, max_epochs, avg_loss)

        self.decoder.eval()
        return {"train_losses": losses}

    def evaluate(
        self,
        dataloader: DataLoader,
        mask_strategy: str = "random",
        mask_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """Evaluate reconstruction quality.

        Args:
            dataloader: DataLoader providing batches with 'timeseries' and 'mask'.
            mask_strategy: Masking strategy to use.
            mask_ratio: Fraction of observed values to mask.

        Returns:
            Dictionary with:
            - nrmse_per_feature: Dict mapping feature name/index to NRMSE
            - mae_overall: Overall mean absolute error on masked positions
            - nrmse_overall: Overall normalized RMSE on masked positions
        """
        self.encoder.eval()
        self.decoder.eval()

        # Accumulators for per-feature and overall metrics
        feature_squared_errors: Dict[int, List[float]] = {}
        feature_abs_errors: Dict[int, List[float]] = {}
        feature_values: Dict[int, List[float]] = {}

        with torch.no_grad():
            for batch in dataloader:
                timeseries = batch["timeseries"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Apply masking
                masked_input, eval_mask, original = self.apply_mask(
                    timeseries, mask, strategy=mask_strategy, mask_ratio=mask_ratio
                )

                # Encode and decode
                encoder_out = self.encoder(masked_input, mask=mask & ~eval_mask)
                reconstruction = self.decoder(encoder_out)

                # Compute errors only on eval_mask positions
                for d in range(timeseries.shape[2]):
                    feat_eval = eval_mask[:, :, d]
                    if feat_eval.sum() == 0:
                        continue

                    recon_vals = reconstruction[:, :, d][feat_eval]
                    orig_vals = original[:, :, d][feat_eval]

                    se = ((recon_vals - orig_vals) ** 2).cpu().tolist()
                    ae = ((recon_vals - orig_vals).abs()).cpu().tolist()
                    ov = orig_vals.cpu().tolist()

                    feature_squared_errors.setdefault(d, []).extend(se)
                    feature_abs_errors.setdefault(d, []).extend(ae)
                    feature_values.setdefault(d, []).extend(ov)

        # Compute metrics
        nrmse_per_feature = {}
        all_squared_errors = []
        all_abs_errors = []

        for d in sorted(feature_squared_errors.keys()):
            se = torch.tensor(feature_squared_errors[d])
            ae = torch.tensor(feature_abs_errors[d])
            vals = torch.tensor(feature_values[d])

            rmse = se.mean().sqrt().item()
            feature_std = vals.std().item() if len(vals) > 1 else 1.0
            nrmse = rmse / max(feature_std, 1e-8)

            name = (
                self.feature_names[d]
                if self.feature_names and d < len(self.feature_names)
                else str(d)
            )
            nrmse_per_feature[name] = nrmse

            all_squared_errors.extend(se.tolist())
            all_abs_errors.extend(ae.tolist())

        # Overall metrics
        if all_squared_errors:
            all_se = torch.tensor(all_squared_errors)
            all_ae = torch.tensor(all_abs_errors)
            mae_overall = all_ae.mean().item()
            rmse_overall = all_se.mean().sqrt().item()

            # Overall NRMSE: normalize by std of all original values
            all_vals = []
            for vals_list in feature_values.values():
                all_vals.extend(vals_list)
            overall_std = torch.tensor(all_vals).std().item() if len(all_vals) > 1 else 1.0
            nrmse_overall = rmse_overall / max(overall_std, 1e-8)
        else:
            mae_overall = 0.0
            nrmse_overall = 0.0

        return {
            "nrmse_per_feature": nrmse_per_feature,
            "mae_overall": mae_overall,
            "nrmse_overall": nrmse_overall,
            "mask_strategy": mask_strategy,
            "mask_ratio": mask_ratio,
        }
