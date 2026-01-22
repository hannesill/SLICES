"""Encoder architectures for time-series data."""

from .base import BaseEncoder, EncoderConfig
from .factory import build_encoder, get_encoder_config_class
from .linear import LinearConfig, LinearEncoder
from .transformer import TransformerConfig, TransformerEncoder

__all__ = [
    "BaseEncoder",
    "EncoderConfig",
    "LinearConfig",
    "LinearEncoder",
    "TransformerConfig",
    "TransformerEncoder",
    "build_encoder",
    "get_encoder_config_class",
]
