"""Self-supervised learning objectives."""

from .base import BaseSSLObjective, SSLConfig
from .contrastive import ContrastiveConfig, ContrastiveObjective
from .factory import build_ssl_objective, get_ssl_config_class
from .jepa import JEPAConfig, JEPAObjective
from .mae import MAEConfig, MAEObjective
from .smart import SMARTObjective, SMARTSSLConfig
from .ts2vec import TS2VecConfig, TS2VecObjective

__all__ = [
    "BaseSSLObjective",
    "SSLConfig",
    "ContrastiveConfig",
    "ContrastiveObjective",
    "JEPAConfig",
    "JEPAObjective",
    "MAEConfig",
    "MAEObjective",
    "SMARTObjective",
    "SMARTSSLConfig",
    "TS2VecConfig",
    "TS2VecObjective",
    "build_ssl_objective",
    "get_ssl_config_class",
]
