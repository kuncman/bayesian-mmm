"""Configuration loading with defaults and validation.

The YAML file only needs to contain the keys you want to change - everything else
falls back to ``DEFAULT_CONFIG`` below.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .transforms import ADSTOCK_TYPES, SATURATION_TYPES

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "path": "data/mmm_synthetic.csv",
        "date_col": "date",
        "target_col": "revenue",
        "channel_cols": [],
        "control_cols": [],
        "holdout_periods": 12,
        "lift_tests_path": None,
        "ground_truth_path": None,
    },
    "model": {
        "adstock": {"type": "geometric", "l_max": 8, "normalize": True},
        "saturation": {"type": "hill"},
        "trend": {"type": "piecewise", "n_changepoints": 5, "changepoint_range": 0.8},
        "seasonality": {"yearly_fourier_order": 3, "period_days": 365.25},
        "likelihood": "studentt",
        "media_prior": "roi",
        "priors": {
            "intercept_mu": 0.5,
            "intercept_sigma": 0.3,
            "trend_slope_sigma": 0.3,
            "changepoint_scale": 0.05,
            "fourier_sigma": 0.2,
            "control_sigma": 0.2,
            "noise_sigma": 0.1,
            "adstock_alpha": [2.0, 3.0],
            "adstock_theta": [2.0, 1.0],
            "adstock_shape": [2.0, 2.0],
            "adstock_scale": [2.0, 4.0],
            "sat_half": [3.0, 6.0],
            "sat_slope": [3.0, 2.0],
            "sat_lam": [3.0, 1.0],
            "beta_sigma": 2.0,
            "roi_mu": 0.0,
            "roi_sigma": 0.8,
            "channel_roi_mu": {},
            "channel_roi_sigma": {},
        },
    },
    "sampler": {
        "draws": 1000,
        "tune": 1500,
        "chains": 4,
        "cores": None,
        "target_accept": 0.95,
        "random_seed": 42,
        "nuts_sampler": "pymc",
        "prior_samples": 500,
    },
    "optimizer": {
        "budget": None,
        "reference_periods": 52,
        "lower_bound_multiplier": 0.5,
        "upper_bound_multiplier": 1.5,
        "horizon_periods": 13,
    },
    "output": {"dir": "outputs", "save_trace": True, "dpi": 130},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def validate_config(cfg: dict) -> dict:
    """Raise a readable error for the most common configuration mistakes."""
    d, m = cfg["data"], cfg["model"]
    if not d["channel_cols"]:
        raise ValueError("config.data.channel_cols must list at least one media channel")
    if m["adstock"]["type"] not in ADSTOCK_TYPES:
        raise ValueError(f"model.adstock.type must be one of {ADSTOCK_TYPES}")
    if m["saturation"]["type"] not in SATURATION_TYPES:
        raise ValueError(f"model.saturation.type must be one of {SATURATION_TYPES}")
    if m["trend"]["type"] not in ("none", "linear", "piecewise"):
        raise ValueError("model.trend.type must be 'none', 'linear' or 'piecewise'")
    if m["likelihood"] not in ("normal", "studentt"):
        raise ValueError("model.likelihood must be 'normal' or 'studentt'")
    if m["media_prior"] not in ("roi", "spend_share"):
        raise ValueError("model.media_prior must be 'roi' or 'spend_share'")
    if int(m["adstock"]["l_max"]) < 1:
        raise ValueError("model.adstock.l_max must be >= 1")
    overlap = set(d["channel_cols"]) & set(d["control_cols"])
    if overlap:
        raise ValueError(f"Columns cannot be both channels and controls: {overlap}")
    return cfg


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config, merge it over the defaults and validate it."""
    user_cfg: dict = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return validate_config(cfg)
