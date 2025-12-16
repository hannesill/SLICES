"""Self-supervised learning objectives."""

from .base import BaseSSLObjective, SSLConfig
from .factory import build_ssl_objective, get_ssl_config_class
from .mae import MAEConfig, MAEObjective

__all__ = [
    "BaseSSLObjective",
    "SSLConfig",
    "MAEConfig",
    "MAEObjective",
    "build_ssl_objective",
    "get_ssl_config_class",
]
