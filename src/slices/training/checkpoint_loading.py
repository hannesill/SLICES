"""Checkpoint loading utilities for encoder weights.

Handles loading pretrained encoder weights from various checkpoint formats
(.pt v1/v2/v3, .ckpt Lightning checkpoints), encoder type inference,
and conditional wrapping with EncoderWithMissingToken.
Extracted from FineTuneModule for modularity.
"""

import logging
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from slices.models.encoders import (
    EncoderWithMissingToken,
    GRUDEncoder,
    ObservationTransformerEncoder,
    SMARTEncoder,
    TransformerEncoder,
    build_encoder,
)

logger = logging.getLogger(__name__)


def infer_encoder_type(state_dict: Dict[str, Any]) -> Optional[str]:
    """Infer encoder type from state_dict keys.

    Used for v2 checkpoints that don't include encoder_config.

    Args:
        state_dict: Encoder state dictionary.

    Returns:
        Inferred encoder name or None if unknown.
    """
    keys = set(state_dict.keys())

    # GRU-D encoder has gru_cell and decay parameters
    if any("gru_cell" in k for k in keys) and any("W_gamma_x" in k for k in keys):
        return "gru_d"

    # SMART encoder has distinctive keys
    if any("embedder" in k or "blocks" in k or "seq_att" in k for k in keys):
        return "smart"

    # Observation transformer has value_proj and feature_embed (not input_proj)
    if any("value_proj" in k for k in keys) and any("feature_embed" in k for k in keys):
        return "observation_transformer"

    # Standard transformer has input_proj and layers
    if any("input_proj" in k or "layers" in k for k in keys):
        return "transformer"

    return None


def wrap_encoder_with_missing_token(
    encoder: nn.Module,
    missing_token: Optional[torch.Tensor],
) -> nn.Module:
    """Conditionally wrap encoder with EncoderWithMissingToken.

    Some encoders handle missingness intrinsically and should not be wrapped:
    - ObservationTransformerEncoder: only tokenizes observed values
    - SMARTEncoder: MLPEmbedder jointly embeds (value, mask) pairs
    - TransformerEncoder with obs_aware=True: obs_proj handles missingness

    Args:
        encoder: The encoder module to potentially wrap.
        missing_token: Optional pretrained missing token. If None,
            a random token will be initialized.

    Returns:
        The encoder, possibly wrapped with EncoderWithMissingToken.
    """
    # Skip for encoders that handle missingness intrinsically
    if isinstance(encoder, (ObservationTransformerEncoder, SMARTEncoder, GRUDEncoder)):
        logger.info(
            "Skipping EncoderWithMissingToken wrapper for %s "
            "(handles missingness intrinsically)",
            type(encoder).__name__,
        )
        return encoder

    if isinstance(encoder, TransformerEncoder) and getattr(encoder.config, "obs_aware", False):
        logger.info(
            "Skipping EncoderWithMissingToken wrapper for obs_aware "
            "TransformerEncoder (forward() handles mask via obs_proj)"
        )
        return encoder

    d_input = encoder.config.d_input

    if missing_token is not None:
        encoder = EncoderWithMissingToken(
            encoder=encoder,
            d_input=d_input,
            missing_token=missing_token,
            init_missing_token=False,
        )
        logger.info("Wrapped encoder with pretrained missing token")
    else:
        encoder = EncoderWithMissingToken(
            encoder=encoder,
            d_input=d_input,
            missing_token=None,
            init_missing_token=True,
        )
        logger.info("Wrapped encoder with randomly initialized missing token")

    return encoder


def load_encoder_weights(
    encoder: nn.Module,
    path: str,
    config: DictConfig,
    use_missing_token: bool,
) -> nn.Module:
    """Load encoder weights from .pt file.

    Handles multiple checkpoint formats:
    - v3+: Contains encoder_config for automatic architecture detection
    - v2: Contains encoder_state_dict, missing_token but no config
    - v1/legacy: Raw state_dict

    For v3+, the encoder is rebuilt from saved config, ensuring architecture
    matches between pretraining and finetuning.

    Args:
        encoder: The encoder module to load weights into (may be rebuilt).
        path: Path to encoder checkpoint.
        config: Full Hydra configuration (for pooling override).
        use_missing_token: Whether to wrap with EncoderWithMissingToken.

    Returns:
        The encoder with loaded weights, possibly wrapped with missing token.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If state dict keys don't match encoder architecture.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    # Detect checkpoint format
    if isinstance(checkpoint, dict) and "version" in checkpoint:
        version = checkpoint["version"]
        state_dict = checkpoint["encoder_state_dict"]
        missing_token = checkpoint.get("missing_token", None)

        # Version 3+: Rebuild encoder from saved config
        if version >= 3 and "encoder_config" in checkpoint:
            encoder_config = dict(checkpoint["encoder_config"])
            encoder_name = encoder_config.pop("name")
            # Override pooling with finetuning config's value.
            # SSL pretraining uses pooling='none' but finetuning needs
            # pooling='mean' or 'query' to get a single representation.
            # Pooling doesn't affect learned weights, just output aggregation.
            finetune_pooling = config.encoder.get("pooling", "mean")
            ckpt_pooling = encoder_config.get("pooling", "none")
            if ckpt_pooling != finetune_pooling:
                encoder_config["pooling"] = finetune_pooling
                logger.info(
                    "Overriding pooling: %s -> %s (finetuning requires aggregated output)",
                    ckpt_pooling,
                    finetune_pooling,
                )
            encoder = build_encoder(encoder_name, encoder_config)
            logger.info(
                "Rebuilt encoder from checkpoint config: %s (d_model=%s)",
                encoder_name,
                encoder_config.get("d_model", "N/A"),
            )
        elif version == 2:
            # v2 checkpoint: try to infer encoder type from state_dict keys
            inferred_encoder = infer_encoder_type(state_dict)
            config_encoder = config.encoder.name

            if inferred_encoder and inferred_encoder != config_encoder:
                raise RuntimeError(
                    f"Encoder architecture mismatch!\n"
                    f"  Checkpoint appears to be: {inferred_encoder}\n"
                    f"  Config specifies: {config_encoder}\n\n"
                    f"Fix: Add 'encoder={inferred_encoder}' to your command:\n"
                    f"  uv run python scripts/training/finetune.py \\\n"
                    f"      checkpoint={path} \\\n"
                    f"      encoder={inferred_encoder} \\\n"
                    f"      task.task_name=...\n\n"
                    f"Or re-run pretraining to create a v3 checkpoint with embedded config."
                )

        encoder.load_state_dict(state_dict)
        logger.info("Loaded encoder weights from: %s (format v%d)", path, version)

        # Wrap encoder with missing token if available and enabled
        if use_missing_token:
            encoder = wrap_encoder_with_missing_token(encoder, missing_token)
    else:
        # Old format (raw state_dict)
        warnings.warn(
            f"Loading old-format checkpoint from {path}. "
            "Missing token will be randomly initialized. "
            "Re-save the encoder using SSLPretrainModule.save_encoder() "
            "to include the learned missing_token.",
            UserWarning,
        )
        encoder.load_state_dict(checkpoint)
        logger.info("Loaded encoder weights from: %s (legacy format)", path)

        # Wrap encoder with random missing token if enabled
        if use_missing_token:
            encoder = wrap_encoder_with_missing_token(encoder, missing_token=None)

    return encoder


def load_from_pretrain_checkpoint(
    encoder: nn.Module,
    path: str,
    config: DictConfig,
    use_missing_token: bool,
) -> nn.Module:
    """Load encoder from full pretraining checkpoint (.ckpt).

    Also extracts the missing_token from the SSL objective if present
    and wraps the encoder with EncoderWithMissingToken.

    The encoder architecture is auto-detected from the checkpoint's
    hyperparameters (saved by save_hyperparameters() during pretraining),
    and the encoder is rebuilt with the correct architecture before
    loading weights.

    Args:
        encoder: The encoder module to load weights into (may be rebuilt).
        path: Path to Lightning checkpoint.
        config: Full Hydra configuration (for pooling override).
        use_missing_token: Whether to wrap with EncoderWithMissingToken.

    Returns:
        The encoder with loaded weights, possibly wrapped with missing token.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        KeyError: If checkpoint doesn't contain 'state_dict'.
        RuntimeError: If no encoder weights found in checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {path} does not contain 'state_dict' key. "
            "Is this a valid PyTorch Lightning checkpoint?"
        )

    state_dict = checkpoint["state_dict"]

    # Extract encoder weights (prefixed with "encoder.")
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key[8:]] = value  # Remove "encoder." prefix

    if not encoder_state_dict:
        raise RuntimeError(
            f"No encoder weights found in checkpoint {path}. "
            "Expected keys prefixed with 'encoder.' in state_dict."
        )

    # Auto-detect encoder architecture from checkpoint hyperparameters
    if "hyper_parameters" in checkpoint:
        hyper_params = checkpoint["hyper_parameters"]
        if "config" in hyper_params and "encoder" in hyper_params["config"]:
            ckpt_encoder_cfg = hyper_params["config"]["encoder"]
            ckpt_encoder_name = ckpt_encoder_cfg.get("name")

            # Always rebuild encoder from checkpoint config to ensure dimensions match.
            # This handles both encoder name mismatches (e.g., transformer vs smart)
            # and parameter mismatches (e.g., d_model=64 vs d_model=32).
            encoder_config_dict = {k: v for k, v in ckpt_encoder_cfg.items() if k != "name"}
            # Override pooling with finetuning config's value.
            # SSL pretraining uses pooling='none' but finetuning needs
            # pooling='mean' or 'query' to get a single representation.
            finetune_pooling = config.encoder.get("pooling", "mean")
            ckpt_pooling = encoder_config_dict.get("pooling", "none")
            if ckpt_pooling != finetune_pooling:
                encoder_config_dict["pooling"] = finetune_pooling
                logger.info(
                    "Overriding pooling: %s -> %s (finetuning requires aggregated output)",
                    ckpt_pooling,
                    finetune_pooling,
                )
            encoder = build_encoder(ckpt_encoder_name, encoder_config_dict)
            logger.info(
                "Rebuilt encoder from checkpoint config: %s (d_model=%s)",
                ckpt_encoder_name,
                encoder_config_dict.get("d_model", "N/A"),
            )
    else:
        # No hyper_parameters - infer encoder type from state_dict keys
        inferred_encoder = infer_encoder_type(encoder_state_dict)
        config_encoder = config.encoder.name

        if inferred_encoder and inferred_encoder != config_encoder:
            raise RuntimeError(
                f"Encoder architecture mismatch!\n"
                f"  Checkpoint appears to be: {inferred_encoder}\n"
                f"  Config specifies: {config_encoder}\n\n"
                f"Fix: Add 'encoder={inferred_encoder}' to your command:\n"
                f"  uv run python scripts/training/finetune.py \\\n"
                f"      pretrain_checkpoint={path} \\\n"
                f"      encoder={inferred_encoder} \\\n"
                f"      task.task_name=...\n"
            )

    encoder.load_state_dict(encoder_state_dict)
    logger.info("Loaded encoder from pretrain checkpoint: %s", path)

    # Extract missing_token from SSL objective if present
    missing_token = None
    if "ssl_objective.missing_token" in state_dict:
        missing_token = state_dict["ssl_objective.missing_token"]

    # Wrap encoder with missing token if enabled
    if use_missing_token:
        encoder = wrap_encoder_with_missing_token(encoder, missing_token)

    return encoder
