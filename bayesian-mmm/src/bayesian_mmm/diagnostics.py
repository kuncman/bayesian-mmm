"""MCMC convergence diagnostics and model-fit metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import BayesianMMM, get_group, has_group

# Thresholds follow Vehtari et al. (2021), "Rank-normalization, folding, and localization"
RHAT_MAX = 1.01
ESS_MIN = 400
BFMI_MIN = 0.3


def parameter_summary(mmm: BayesianMMM) -> pd.DataFrame:
    """Posterior summary table (mean, sd, credible interval, ESS, R-hat) via ArviZ."""
    import arviz as az

    names = [n for n in mmm.parameter_names if n in get_group(mmm.idata, "posterior")]
    return az.summary(mmm.idata, var_names=names)


def _divergences(idata) -> int:
    ss = get_group(idata, "sample_stats")
    for key in ("diverging", "divergences"):
        if key in ss:
            return int(np.asarray(ss[key].values).astype(bool).sum())
    return 0


def _bfmi(idata) -> np.ndarray:
    """Energy Bayesian fraction of missing information, per chain."""
    ss = get_group(idata, "sample_stats")
    if "energy" not in ss:
        return np.array([np.nan])
    e = ss["energy"].values  # (chain, draw)
    return np.square(np.diff(e, axis=1)).mean(axis=1) / e.var(axis=1)


def convergence_report(mmm: BayesianMMM) -> dict:
    """Return the key convergence numbers and a pass/fail verdict for each."""
    summ = parameter_summary(mmm)
    rhat = pd.to_numeric(summ["r_hat"], errors="coerce")
    ess_b = pd.to_numeric(summ["ess_bulk"], errors="coerce")
    ess_t = pd.to_numeric(summ["ess_tail"], errors="coerce")
    div = _divergences(mmm.idata)
    bfmi = _bfmi(mmm.idata)
    n_draws = int(np.prod(get_group(mmm.idata, "posterior")["intercept"].shape))
    rep = {
        "n_posterior_draws": n_draws,
        "divergences": div,
        "divergence_rate": div / max(n_draws, 1),
        "max_r_hat": float(rhat.max()),
        "worst_r_hat_param": str(rhat.idxmax()),
        "min_ess_bulk": float(ess_b.min()),
        "min_ess_tail": float(ess_t.min()),
        "min_bfmi": float(np.nanmin(bfmi)),
    }
    rep["checks"] = {
        "no_divergences": div == 0,
        "r_hat_ok": rep["max_r_hat"] <= RHAT_MAX,
        "ess_ok": min(rep["min_ess_bulk"], rep["min_ess_tail"]) >= ESS_MIN,
        "bfmi_ok": rep["min_bfmi"] >= BFMI_MIN,
    }
    rep["converged"] = all(rep["checks"].values())
    return rep


# ---------------------------------------------------------------------------
# Fit metrics
# ---------------------------------------------------------------------------


def _metrics(y: np.ndarray, pred_draws: np.ndarray, mu_draws: np.ndarray) -> dict:
    yhat = mu_draws.mean(axis=0)
    resid = y - yhat
    lo, hi = np.quantile(pred_draws, [0.03, 0.97], axis=0)
    lo50, hi50 = np.quantile(pred_draws, [0.25, 0.75], axis=0)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    # Continuous ranked probability score (sample estimator), lower is better
    sample = pred_draws[: min(len(pred_draws), 200)]  # pairwise term is O(n^2) in memory
    crps = np.mean(np.abs(sample - y).mean(0) - 0.5 * np.abs(sample[:, None] - sample[None]).mean((0, 1)))
    return {
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mape_pct": float(np.mean(np.abs(resid / y)) * 100),
        "nrmse": float(np.sqrt(np.mean(resid**2)) / (y.max() - y.min())),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "coverage_94_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "coverage_50_pct": float(np.mean((y >= lo50) & (y <= hi50)) * 100),
        "crps": float(crps),
    }


def fit_metrics(mmm: BayesianMMM, max_draws: int = 1000) -> dict:
    """In-sample and holdout accuracy & calibration of the posterior predictive."""
    d = mmm.data
    post = mmm.posterior(max_draws)
    comps = mmm.forward(post)
    pred = mmm.predictive_samples(post, comps["mu"])
    tr = d.train_slice()
    out = {"train": _metrics(d.y[tr], pred[:, tr], comps["mu"][:, tr])}
    if d.n_holdout:
        ho = slice(d.n_train, d.n_periods)
        out["holdout"] = _metrics(d.y[ho], pred[:, ho], comps["mu"][:, ho])
    return out


def prior_predictive_check(mmm: BayesianMMM) -> dict:
    """How plausible is the prior? Share of observed points inside the prior 94% band."""
    if mmm.prior_idata is None or not has_group(mmm.prior_idata, "prior_predictive"):
        return {}
    pp = get_group(mmm.prior_idata, "prior_predictive")["y"].values
    pp = pp.reshape(-1, pp.shape[-1]) * mmm.data.y_scale
    y = mmm.data.y[: mmm.data.n_train]
    lo, hi = np.quantile(pp, [0.03, 0.97], axis=0)
    return {
        "observed_inside_prior_94_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "prior_predictive_mean": float(pp.mean()),
        "observed_mean": float(y.mean()),
    }
