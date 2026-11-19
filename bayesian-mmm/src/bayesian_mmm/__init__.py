"""bayesian_mmm - a transparent, end-to-end Bayesian Marketing Mix Model built on PyMC."""

from .config import load_config
from .data import MMMData, load_dataframe, prepare_data
from .model import BayesianMMM
from .optimizer import optimize_budget
from .pipeline import build_model, run_pipeline

__all__ = [
    "BayesianMMM",
    "MMMData",
    "build_model",
    "load_config",
    "load_dataframe",
    "optimize_budget",
    "prepare_data",
    "run_pipeline",
]
__version__ = "1.0.0"
