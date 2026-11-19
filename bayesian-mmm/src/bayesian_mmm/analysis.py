"""Post-fit analysis: decomposition, ROI, marginal ROI, response curves, carry-over."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import transforms as tf
from .model import BayesianMMM

CI = (0.03, 0.97)  # 94% equal-tailed credible interval used throughout


def _ci(x: np.ndarray, axis: int = 0):
    lo, hi = np.quantile(x, CI, axis=axis)
    return x.mean(axis=axis), lo, hi


def reference_spend(mmm: BayesianMMM, periods: int) -> np.ndarray:
    """Average per-period spend over the last ``periods`` periods (the "current" plan)."""
    m = mmm.data.media
    return m[-periods:].mean(axis=0)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def decomposition(mmm: BayesianMMM, max_draws: int = 1000) -> pd.DataFrame:
    """Posterior-mean contribution of every component for every period (orig. units)."""
    d = mmm.data
    comps = mmm.forward(mmm.posterior(max_draws))
    df = pd.DataFrame(index=d.dates)
    df.index.name = "date"
    df["actual"] = d.y
    df["fitted"] = comps["mu"].mean(0)
    df["intercept"] = comps["intercept"].mean(0)
    df["trend"] = comps["trend"].mean(0)
    df["seasonality"] = comps["seasonality"].mean(0)
    for k, name in enumerate(d.control_names):
        df[f"control:{name}"] = comps["controls"][:, :, k].mean(0)
    for c, name in enumerate(d.channels):
        df[f"media:{name}"] = comps["media"][:, :, c].mean(0)
    df["is_holdout"] = np.arange(d.n_periods) >= d.n_train
    return df


def contribution_shares(decomp: pd.DataFrame) -> pd.DataFrame:
    """Total contribution and share of the fitted outcome, per component (train window)."""
    tr = decomp[~decomp["is_holdout"]]
    cols = [c for c in tr.columns if c not in ("actual", "fitted", "is_holdout")]
    tot = tr[cols].sum()
    return pd.DataFrame({"total_contribution": tot, "share_of_fitted_pct": tot / tr["fitted"].sum() * 100})


# ---------------------------------------------------------------------------
# Channel-level KPIs
# ---------------------------------------------------------------------------


def adstock_mean_lag(mmm: BayesianMMM, params: dict) -> np.ndarray:
    """Mean carry-over lag (periods) per draw and channel: sum_l l * w_l / sum_l w_l."""
    w = tf.adstock_weights(mmm.adstock_type, params, mmm.l_max, True, xp=np)  # (D, L, C)
    lags = np.arange(mmm.l_max)[None, :, None]
    return (w * lags).sum(axis=1)


def channel_summary(mmm: BayesianMMM, reference_periods: int = 13, max_draws: int = 1000) -> pd.DataFrame:
    """One row per channel with spend, contribution, ROI, marginal ROI and carry-over."""
    d = mmm.data
    post = mmm.posterior(max_draws)
    comps = mmm.forward(post)
    tr = d.train_slice()

    spend = d.media[tr].sum(0)  # (C,)
    contrib = comps["media"][:, tr, :].sum(1)  # (D, C)
    roi = contrib / spend
    total_media = contrib.sum(1, keepdims=True)

    ref = reference_spend(mmm, reference_periods)
    eps = 0.01
    base_resp = mmm.steady_state_response(post, ref)  # (D, C)
    up_resp = mmm.steady_state_response(post, ref * (1 + eps))
    mroi = (up_resp - base_resp) / np.where(ref * eps > 0, ref * eps, np.nan)

    lag = adstock_mean_lag(mmm, post)

    rows = []
    for c, ch in enumerate(d.channels):
        r_m, r_lo, r_hi = _ci(roi[:, c])
        m_m, m_lo, m_hi = _ci(mroi[:, c])
        c_m, c_lo, c_hi = _ci(contrib[:, c])
        l_m, l_lo, l_hi = _ci(lag[:, c])
        row = {
            "channel": ch,
            "spend": spend[c],
            "spend_share_pct": spend[c] / spend.sum() * 100,
            "contribution": c_m,
            "contribution_lo": c_lo,
            "contribution_hi": c_hi,
            "contribution_share_pct": float((contrib[:, c] / total_media[:, 0]).mean() * 100),
            "roi": r_m,
            "roi_lo": r_lo,
            "roi_hi": r_hi,
            "prob_roi_gt_1": float((roi[:, c] > 1).mean()),
            "reference_spend_per_period": ref[c],
            "marginal_roi": m_m,
            "marginal_roi_lo": m_lo,
            "marginal_roi_hi": m_hi,
            "mean_carryover_lag": l_m,
            "mean_carryover_lag_lo": l_lo,
            "mean_carryover_lag_hi": l_hi,
        }
        if "sat_half" in post:
            row["half_saturation_spend"] = float((post["sat_half"][:, c] * d.media_scale[c]).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("channel")


def response_curves(
    mmm: BayesianMMM, n_points: int = 60, max_multiple: float = 2.0, max_draws: int = 1000
) -> pd.DataFrame:
    """Steady-state response (contribution per period) vs. spend per period, per channel."""
    d = mmm.data
    post = mmm.posterior(max_draws)
    top = d.media[: d.n_train].max(0) * max_multiple
    grid = np.linspace(0, 1, n_points)[:, None] * top[None, :]  # (G, C)
    resp = mmm.steady_state_response(post, grid)  # (D, G, C)
    mean, lo, hi = _ci(resp)
    rows = []
    for c, ch in enumerate(d.channels):
        for g in range(n_points):
            rows.append(
                {"channel": ch, "spend": grid[g, c], "response": mean[g, c], "lo": lo[g, c], "hi": hi[g, c]}
            )
    return pd.DataFrame(rows)


def adstock_curves(mmm: BayesianMMM, max_draws: int = 1000) -> pd.DataFrame:
    """Posterior adstock weights per lag and channel."""
    post = mmm.posterior(max_draws)
    w = tf.adstock_weights(mmm.adstock_type, post, mmm.l_max, True, xp=np)
    mean, lo, hi = _ci(w)
    rows = []
    for c, ch in enumerate(mmm.data.channels):
        for lag in range(mmm.l_max):
            rows.append({"channel": ch, "lag": lag, "weight": mean[lag, c], "lo": lo[lag, c], "hi": hi[lag, c]})
    return pd.DataFrame(rows)


def prior_vs_posterior(mmm: BayesianMMM, var: str = "roi") -> pd.DataFrame:
    """Compare prior and posterior for a channel-level parameter (did the data teach us anything?)."""
    from .model import get_group

    if mmm.prior_idata is None:
        return pd.DataFrame()
    prior = get_group(mmm.prior_idata, "prior")
    if var not in prior:
        return pd.DataFrame()
    pr = prior[var].values.reshape(-1, mmm.data.n_channels)
    po = mmm.posterior()[var]
    # ROI has a LogNormal prior, so learning is measured on the log scale
    tr = np.log if var == "roi" else (lambda a: a)
    rows = []
    for c, ch in enumerate(mmm.data.channels):
        rows.append(
            {
                "channel": ch,
                "prior_mean": pr[:, c].mean(),
                "prior_sd": pr[:, c].std(),
                "posterior_mean": po[:, c].mean(),
                "posterior_sd": po[:, c].std(),
                # 1 - posterior_var/prior_var: 0 = learned nothing, 1 = fully determined by data
                "shrinkage": 1 - tr(po[:, c]).var() / tr(pr[:, c]).var(),
            }
        )
    return pd.DataFrame(rows).set_index("channel")


# ---------------------------------------------------------------------------
# Ground-truth recovery (synthetic data only)
# ---------------------------------------------------------------------------


def parameter_recovery(mmm: BayesianMMM, truth_path: str, summary: pd.DataFrame) -> pd.DataFrame:
    """Compare estimates with the true values used to simulate the data."""
    truth = json.loads(open(truth_path, encoding="utf-8").read())
    post = mmm.posterior()
    d = mmm.data
    rows = []
    for c, ch in enumerate(d.channels):
        t = truth["channels"].get(ch)
        if t is None:
            continue
        checks = [("roi", t["roi"], summary.loc[ch, ["roi", "roi_lo", "roi_hi"]].to_numpy())]
        if "adstock_alpha" in post and mmm.adstock_type == "geometric":
            checks.append(("adstock_alpha", t["alpha"], _ci(post["adstock_alpha"][:, c])))
        if "sat_half" in post:
            checks.append(("half_saturation_spend", t["half_sat"], _ci(post["sat_half"][:, c] * d.media_scale[c])))
        if "sat_slope" in post:
            checks.append(("hill_slope", t["slope"], _ci(post["sat_slope"][:, c])))
        for name, true, (m, lo, hi) in checks:
            rows.append(
                {
                    "channel": ch,
                    "parameter": name,
                    "true": true,
                    "estimate": float(m),
                    "lo": float(lo),
                    "hi": float(hi),
                    "covered_by_94_ci": bool(lo <= true <= hi),
                }
            )
    return pd.DataFrame(rows)
