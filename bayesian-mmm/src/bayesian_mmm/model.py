"""The Bayesian Marketing Mix Model.

Model (all quantities on the *scaled* target, y / max(y_train)):

    y_t  ~ StudentT(nu, mu_t, sigma)            (or Normal)
    mu_t = intercept
         + trend_t                               (linear or piecewise-linear)
         + seasonality_t                         (yearly Fourier terms)
         + sum_k gamma_k * z_{t,k}               (control variables)
         + sum_c beta_c * Sat_c( Adstock_c(x_{.,c}) )_t

The media contribution of channel ``c`` is ``beta_c * Sat_c(Adstock_c(x_c))``.
Adstock captures carry-over (an ad today still works next week); saturation
captures diminishing returns (doubling spend does not double sales).

Two ways to put a prior on media effectiveness are offered:

* ``roi``         - a LogNormal prior directly on each channel's ROI (as in Google
                     Meridian). ``beta_c`` is then derived deterministically. This is
                     the most intuitive prior to elicit from the business.
* ``spend_share`` - HalfNormal prior on ``beta_c`` with a scale proportional to the
                     channel's share of spend (as in PyMC-Marketing / LightweightMMM).

Optional lift tests (geo-experiments, conversion-lift studies) add an extra
likelihood term that pins the saturation curve to experimentally measured
incremental outcomes.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from . import transforms as tf
from .data import MMMData

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers that work with both arviz.InferenceData (PyMC 5) and xarray.DataTree (PyMC 6)
# ---------------------------------------------------------------------------


def get_group(idata, name: str):
    """Return an inference-data group as an ``xarray.Dataset``."""
    g = idata[name]
    return g.to_dataset() if hasattr(g, "to_dataset") else g


def has_group(idata, name: str) -> bool:
    try:
        get_group(idata, name)
        return True
    except (KeyError, AttributeError):
        return False


def stack_draws(ds, var: str) -> np.ndarray:
    """Flatten (chain, draw, ...) -> (chain*draw, ...)."""
    v = ds[var].values
    return v.reshape((-1,) + v.shape[2:])


class BayesianMMM:
    """Build, fit and query a Bayesian MMM."""

    def __init__(self, cfg: dict, data: MMMData, lift_tests: pd.DataFrame | None = None):
        self.cfg = cfg
        self.data = data
        self.lift_tests = lift_tests
        mc = cfg["model"]
        self.adstock_type = mc["adstock"]["type"]
        self.l_max = int(mc["adstock"]["l_max"])
        self.normalize = bool(mc["adstock"].get("normalize", True))
        self.saturation_type = mc["saturation"]["type"]
        self.trend_type = mc["trend"]["type"]
        self.likelihood = mc["likelihood"]
        self.media_prior = mc["media_prior"]
        self.priors = mc["priors"]
        self.model: pm.Model | None = None
        self.idata = None
        self.prior_idata = None
        self._posterior_cache: dict | None = None

    # ------------------------------------------------------------------ names
    @property
    def adstock_params(self) -> list[str]:
        return {
            "geometric": ["adstock_alpha"],
            "delayed": ["adstock_alpha", "adstock_theta"],
            "weibull": ["adstock_shape", "adstock_scale"],
        }[self.adstock_type]

    @property
    def saturation_params(self) -> list[str]:
        return {
            "hill": ["sat_half", "sat_slope"],
            "logistic": ["sat_lam"],
            "michaelis_menten": ["sat_half"],
        }[self.saturation_type]

    @property
    def parameter_names(self) -> list[str]:
        names = ["intercept"]
        if self.trend_type in ("linear", "piecewise"):
            names.append("trend_slope")
        if self.trend_type == "piecewise" and self.data.n_changepoints:
            names.append("trend_delta")
        if self.data.fourier_order:
            names.append("gamma_fourier")
        if self.data.control_names:
            names.append("gamma_control")
        names += self.adstock_params + self.saturation_params
        if self.media_prior == "roi":
            names.append("roi")
        names += ["beta", "sigma"]
        if self.likelihood == "studentt":
            names.append("nu")
        return names

    # ------------------------------------------------------------- features
    def _trend_design(self, t: np.ndarray) -> np.ndarray:
        """Piecewise-linear design matrix A[t, j] = max(0, t - s_j)."""
        s = self.data.changepoints()
        return np.maximum(0.0, t[:, None] - s[None, :])

    def _lagged_media(self, media_raw: np.ndarray) -> np.ndarray:
        return tf.build_lagged_media(self.data.scale_media(media_raw), self.l_max)

    # ----------------------------------------------------------------- build
    def _ab(self, key):
        a, b = self.priors[key]
        return float(a), float(b)

    def _roi_prior_params(self):
        mu = np.full(self.data.n_channels, float(self.priors["roi_mu"]))
        sd = np.full(self.data.n_channels, float(self.priors["roi_sigma"]))
        for i, ch in enumerate(self.data.channels):
            if ch in (self.priors.get("channel_roi_mu") or {}):
                # user supplies the *median ROI*; LogNormal mu is on the log scale
                mu[i] = np.log(float(self.priors["channel_roi_mu"][ch]))
            if ch in (self.priors.get("channel_roi_sigma") or {}):
                sd[i] = float(self.priors["channel_roi_sigma"][ch])
        return mu, sd

    def build(self) -> pm.Model:
        d = self.data
        tr = d.train_slice()
        p = self.priors

        coords = {
            "date": d.dates[tr],
            "channel": d.channels,
            "lag": np.arange(self.l_max),
        }
        if d.control_names:
            coords["control"] = d.control_names
        if d.fourier_order:
            coords["fourier"] = d.fourier_names()
        if self.trend_type == "piecewise" and d.n_changepoints:
            coords["changepoint"] = np.arange(d.n_changepoints)

        x_lag = self._lagged_media(d.media)[:, tr, :]
        t = d.time_index()[tr]
        y = d.y_scaled[tr]

        with pm.Model(coords=coords) as model:
            # ---------------- data containers (named so they can be inspected)
            x_lag_d = pm.Data("media_lagged", x_lag)

            # ---------------- baseline
            intercept = pm.Normal("intercept", mu=p["intercept_mu"], sigma=p["intercept_sigma"])
            mu = intercept * pt.ones(len(y))

            if self.trend_type in ("linear", "piecewise"):
                slope = pm.Normal("trend_slope", 0.0, p["trend_slope_sigma"])
                trend = slope * t
                if self.trend_type == "piecewise" and d.n_changepoints:
                    delta = pm.Laplace(
                        "trend_delta", 0.0, p["changepoint_scale"], dims="changepoint"
                    )
                    trend = trend + pt.dot(self._trend_design(t), delta)
                trend = pm.Deterministic("trend_component", trend, dims="date")
                mu = mu + trend

            if d.fourier_order:
                g_f = pm.Normal("gamma_fourier", 0.0, p["fourier_sigma"], dims="fourier")
                season = pm.Deterministic(
                    "seasonality_component", pt.dot(d.fourier_features()[tr], g_f), dims="date"
                )
                mu = mu + season

            if d.control_names:
                g_c = pm.Normal("gamma_control", 0.0, p["control_sigma"], dims="control")
                ctrl = pm.Deterministic(
                    "control_contribution",
                    d.scaled_controls()[tr] * g_c[None, :],
                    dims=("date", "control"),
                )
                mu = mu + ctrl.sum(axis=1)

            # ---------------- media: adstock
            params = {}
            if self.adstock_type in ("geometric", "delayed"):
                params["adstock_alpha"] = pm.Beta(
                    "adstock_alpha", *self._ab("adstock_alpha"), dims="channel"
                )
            if self.adstock_type == "delayed":
                a, b = self._ab("adstock_theta")
                params["adstock_theta"] = pm.Gamma("adstock_theta", alpha=a, beta=b, dims="channel")
            if self.adstock_type == "weibull":
                a, b = self._ab("adstock_shape")
                params["adstock_shape"] = pm.Gamma("adstock_shape", alpha=a, beta=b, dims="channel")
                a, b = self._ab("adstock_scale")
                params["adstock_scale"] = pm.Gamma("adstock_scale", alpha=a, beta=b, dims="channel")

            w = tf.adstock_weights(self.adstock_type, params, self.l_max, self.normalize, xp=pt)
            pm.Deterministic("adstock_weights", w, dims=("lag", "channel"))
            adstocked = tf.apply_adstock(x_lag_d, w, xp=pt)

            # ---------------- media: saturation
            if self.saturation_type in ("hill", "michaelis_menten"):
                a, b = self._ab("sat_half")
                params["sat_half"] = pm.Gamma("sat_half", alpha=a, beta=b, dims="channel")
            if self.saturation_type == "hill":
                a, b = self._ab("sat_slope")
                params["sat_slope"] = pm.Gamma("sat_slope", alpha=a, beta=b, dims="channel")
            if self.saturation_type == "logistic":
                a, b = self._ab("sat_lam")
                params["sat_lam"] = pm.Gamma("sat_lam", alpha=a, beta=b, dims="channel")

            saturated = tf.saturate(self.saturation_type, adstocked, params, xp=pt)

            # ---------------- media: effect size
            if self.media_prior == "roi":
                roi_mu, roi_sd = self._roi_prior_params()
                roi = pm.LogNormal("roi", mu=roi_mu, sigma=roi_sd, dims="channel")
                spend_total = d.media[tr].sum(axis=0)
                # contribution_c (orig. units) = beta_c * y_scale * sum_t sat_tc  ==  roi_c * spend_c
                beta = pm.Deterministic(
                    "beta",
                    roi * spend_total / (d.y_scale * saturated.sum(axis=0) + 1e-9),
                    dims="channel",
                )
            else:
                share = d.spend_share()
                beta = pm.HalfNormal(
                    "beta", sigma=p["beta_sigma"] * share * len(share) / 2, dims="channel"
                )
            params["beta"] = beta

            contrib = pm.Deterministic(
                "channel_contribution", saturated * beta[None, :], dims=("date", "channel")
            )
            mu = pm.Deterministic("mu", mu + contrib.sum(axis=1), dims="date")

            # ---------------- likelihood
            sigma = pm.HalfNormal("sigma", p["noise_sigma"])
            if self.likelihood == "studentt":
                nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
                pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y, dims="date")
            else:
                pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="date")

            # ---------------- lift-test calibration
            if self.lift_tests is not None and len(self.lift_tests):
                self._add_lift_tests(params)

        self.model = model
        return model

    def _add_lift_tests(self, params: dict):
        """Calibrate the saturation curves with experimental lift measurements.

        Each lift test says: moving steady-state spend in channel c from x0 to x1
        changed the outcome by delta_y (+/- delta_y_sd) per period. With normalised
        adstock, steady-state adstocked spend equals spend, so the model-implied lift
        is ``beta_c * (Sat_c(x1) - Sat_c(x0))``.
        """
        d, lt = self.data, self.lift_tests
        idx = np.array([d.channels.index(c) for c in lt["channel"]])
        x0 = lt["spend_before"].to_numpy() / d.media_scale[idx]
        x1 = lt["spend_after"].to_numpy() / d.media_scale[idx]
        p_idx = {k: params[k][idx] for k in self.saturation_params}
        lift = params["beta"][idx] * (
            tf.saturate(self.saturation_type, x1, p_idx, xp=pt)
            - tf.saturate(self.saturation_type, x0, p_idx, xp=pt)
        )
        pm.Deterministic("lift_model", lift)
        pm.Normal(
            "lift_obs",
            mu=lift,
            sigma=lt["delta_y_sd"].to_numpy() / d.y_scale,
            observed=lt["delta_y"].to_numpy() / d.y_scale,
        )

    # ------------------------------------------------------------------- fit
    def sample_prior_predictive(self):
        if self.model is None:
            self.build()
        s = self.cfg["sampler"]
        with self.model:
            self.prior_idata = pm.sample_prior_predictive(
                draws=int(s["prior_samples"]), random_seed=s["random_seed"]
            )
        return self.prior_idata

    def fit(self, progressbar: bool = True):
        if self.model is None:
            self.build()
        s = self.cfg["sampler"]
        cores = s.get("cores") or min(int(s["chains"]), os.cpu_count() or 1)
        kwargs = dict(
            draws=int(s["draws"]),
            tune=int(s["tune"]),
            chains=int(s["chains"]),
            cores=int(cores),
            target_accept=float(s["target_accept"]),
            random_seed=s["random_seed"],
            progressbar=progressbar,
        )
        if s.get("nuts_sampler", "pymc") != "pymc":
            kwargs["nuts_sampler"] = s["nuts_sampler"]
        with self.model:
            self.idata = pm.sample(**kwargs)
            pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=True,
                random_seed=s["random_seed"],
                progressbar=progressbar,
            )
        self._posterior_cache = None
        return self.idata

    # ---------------------------------------------------------- posterior I/O
    def posterior(self, max_draws: int | None = None) -> dict[str, np.ndarray]:
        """Posterior samples as a dict of arrays with a leading draw dimension."""
        if self.idata is None:
            raise RuntimeError("Model has not been fit yet - call .fit() first")
        if self._posterior_cache is None:
            ds = get_group(self.idata, "posterior")
            self._posterior_cache = {v: stack_draws(ds, v) for v in ds.data_vars}
        post = self._posterior_cache
        if max_draws is not None:
            n = len(next(iter(post.values())))
            if n > max_draws:
                rng = np.random.default_rng(0)
                sel = np.sort(rng.choice(n, max_draws, replace=False))
                return {k: v[sel] for k, v in post.items()}
        return post

    # --------------------------------------------------------------- forward
    def forward(
        self,
        params: dict[str, np.ndarray],
        media: np.ndarray | None = None,
        controls: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Recompute every model component in numpy for the full period.

        Used for decomposition, holdout prediction and scenario analysis. Media and
        controls default to the observed data; pass a modified ``media`` (T, C) to
        run "what-if" scenarios. All returned arrays are in ORIGINAL target units.

        Returns a dict with keys ``intercept (D, T)``, ``trend (D, T)``,
        ``seasonality (D, T)``, ``controls (D, T, K)``, ``media (D, T, C)``, ``mu (D, T)``.
        """
        d = self.data
        media = d.media if media is None else media
        n_draws = len(params["intercept"])
        t = d.time_index()
        T = len(t)

        out = {"intercept": np.repeat(params["intercept"][:, None], T, axis=1)}

        trend = np.zeros((n_draws, T))
        if self.trend_type in ("linear", "piecewise"):
            trend = params["trend_slope"][:, None] * t[None, :]
            if "trend_delta" in params:
                trend = trend + params["trend_delta"] @ self._trend_design(t).T
        out["trend"] = trend

        season = np.zeros((n_draws, T))
        if d.fourier_order:
            season = params["gamma_fourier"] @ d.fourier_features().T
        out["seasonality"] = season

        if d.control_names:
            z = d.scaled_controls() if controls is None else (controls - d.control_mean) / d.control_std
            out["controls"] = params["gamma_control"][:, None, :] * z[None, :, :]
        else:
            out["controls"] = np.zeros((n_draws, T, 0))

        w = tf.adstock_weights(self.adstock_type, params, self.l_max, self.normalize, xp=np)
        adstocked = tf.apply_adstock(self._lagged_media(media), w, xp=np)
        sat = tf.saturate(self.saturation_type, adstocked, params, xp=np, expand=True)
        out["media"] = sat * params["beta"][:, None, :]

        out["mu"] = (
            out["intercept"]
            + out["trend"]
            + out["seasonality"]
            + out["controls"].sum(axis=2)
            + out["media"].sum(axis=2)
        )
        return {k: v * d.y_scale for k, v in out.items()}

    def predictive_samples(self, params, mu_orig: np.ndarray, seed: int = 0) -> np.ndarray:
        """Add observation noise to ``mu`` to get posterior-predictive draws (orig. units)."""
        rng = np.random.default_rng(seed)
        sigma = params["sigma"][:, None] * self.data.y_scale
        if self.likelihood == "studentt":
            eps = rng.standard_t(params["nu"][:, None], size=mu_orig.shape)
        else:
            eps = rng.standard_normal(mu_orig.shape)
        return mu_orig + sigma * eps

    # ---------------------------------------------------- steady-state curves
    def steady_state_response(self, params, spend_per_period: np.ndarray) -> np.ndarray:
        """Per-period contribution if spend were held constant forever.

        ``spend_per_period`` has shape (C,) or (G, C). Returns (D, C) or (D, G, C),
        in original target units. With normalised adstock the long-run adstocked
        level equals the spend level, so only saturation and beta matter.
        """
        s = np.atleast_2d(spend_per_period) / self.data.media_scale  # (G, C)
        w_sum = 1.0
        if not self.normalize:
            w = tf.adstock_weights(self.adstock_type, params, self.l_max, False, xp=np)
            w_sum = w.sum(axis=1)[:, None, :]  # (D, 1, C)
        x = s[None, :, :] * w_sum  # (D, G, C)
        sat = tf.saturate(self.saturation_type, x, params, xp=np, expand=True)
        resp = sat * params["beta"][:, None, :] * self.data.y_scale
        return resp[:, 0, :] if np.ndim(spend_per_period) == 1 else resp
