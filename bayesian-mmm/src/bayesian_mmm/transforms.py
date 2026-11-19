"""Media transformation functions: adstock (carry-over) and saturation (diminishing returns).

Every function is written once and works with two array "backends":

* ``numpy``            - used for simulation, post-hoc analysis and budget optimisation
* ``pytensor.tensor``  - used inside the PyMC model so the sampler can differentiate

Pass the backend module as ``xp`` (defaults to numpy). Parameters must already be
broadcast-compatible with the data they are applied to - the caller is responsible
for adding any leading "draw" dimensions.

Shapes used throughout
----------------------
T : number of time periods
C : number of media channels
L : maximum adstock lag (``l_max``)
D : number of posterior draws (numpy analysis only)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Lag matrix
# ---------------------------------------------------------------------------


def build_lagged_media(x: np.ndarray, l_max: int) -> np.ndarray:
    """Stack lagged copies of a media matrix.

    Parameters
    ----------
    x : array (T, C)
        Media spend (or impressions) per period and channel.
    l_max : int
        Number of lags to include (lag 0 .. l_max-1).

    Returns
    -------
    array (L, T, C) where ``out[l, t, c] = x[t - l, c]`` (0 when ``t - l < 0``).

    Pre-computing this tensor once means adstock inside the model is a single
    weighted sum - fast and fully differentiable.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (T, C); got shape {x.shape}")
    if l_max < 1:
        raise ValueError("l_max must be >= 1")
    t, c = x.shape
    out = np.zeros((l_max, t, c), dtype=float)
    for lag in range(l_max):
        out[lag, lag:, :] = x[: t - lag, :]
    return out


# ---------------------------------------------------------------------------
# Adstock weight functions  -> return weights with shape (..., L, C)
# ---------------------------------------------------------------------------


def _lags(l_max: int) -> np.ndarray:
    return np.arange(l_max, dtype=float)[:, None]  # (L, 1)


def _normalise(w, xp=np):
    return w / w.sum(axis=-2, keepdims=True)


def geometric_weights(alpha, l_max: int, normalize: bool = True, xp=np):
    """Geometric decay: ``w_l = alpha ** l``.

    ``alpha`` in (0, 1) is the share of the effect that carries over to the next
    period. alpha=0 -> no carry-over; alpha=0.8 -> long memory (e.g. TV).
    """
    alpha = xp.expand_dims(alpha, -2)  # (..., 1, C)
    w = alpha ** _lags(l_max)
    return _normalise(w, xp) if normalize else w


def delayed_weights(alpha, theta, l_max: int, normalize: bool = True, xp=np):
    """Delayed (peaked) adstock (Jin et al., 2017, Google):

    ``w_l = alpha ** ((l - theta) ** 2)``

    ``theta`` is the lag (in periods) at which the effect peaks - useful for media
    such as TV or out-of-home where the response builds before decaying.
    """
    alpha = xp.expand_dims(alpha, -2)
    theta = xp.expand_dims(theta, -2)
    w = alpha ** ((_lags(l_max) - theta) ** 2)
    return _normalise(w, xp) if normalize else w


def weibull_pdf_weights(shape, scale, l_max: int, normalize: bool = True, xp=np):
    """Weibull-PDF adstock (as used by Meta's Robyn).

    ``shape`` < 1 -> fast decay from lag 0; ``shape`` > 1 -> delayed peak.
    ``scale`` stretches the curve over more lags (expressed in units of l_max).
    """
    shape = xp.expand_dims(shape, -2)
    scale = xp.expand_dims(scale, -2) * l_max
    lag = _lags(l_max) + 1.0  # avoid 0 ** (k-1) blow-ups when k < 1
    w = (shape / scale) * (lag / scale) ** (shape - 1.0) * xp.exp(-((lag / scale) ** shape))
    return _normalise(w, xp) if normalize else w


# ---------------------------------------------------------------------------
# Apply adstock
# ---------------------------------------------------------------------------


def apply_adstock(x_lagged, weights, xp=np):
    """Convolve lagged media with adstock weights.

    Parameters
    ----------
    x_lagged : (L, T, C)
    weights  : (L, C) for the PyMC model, or (D, L, C) for numpy analysis.

    Returns
    -------
    (T, C) or (D, T, C)
    """
    if xp is np:
        return np.einsum("...lc,ltc->...tc", weights, x_lagged)
    # pytensor: weights (L, C) -> (L, 1, C)
    return (weights[:, None, :] * x_lagged).sum(axis=0)


# ---------------------------------------------------------------------------
# Saturation functions
# ---------------------------------------------------------------------------


def hill(x, half_sat, slope, xp=np):
    """Hill saturation curve, output in [0, 1).

    ``half_sat`` (K) is the (scaled) media level that produces half the maximum
    effect. ``slope`` (S) controls the shape: S <= 1 is concave (C-shaped),
    S > 1 is S-shaped (a threshold before media starts working).
    """
    # clip at a tiny positive value: d/dS x**S = x**S * log(x) is NaN at x == 0
    x = xp.maximum(x, 1e-12)
    xs = x**slope
    return xs / (xs + half_sat**slope)


def logistic(x, lam, xp=np):
    """Logistic saturation used by PyMC-Marketing: ``(1 - e^{-lam x}) / (1 + e^{-lam x})``.

    Always concave; ``lam`` controls how fast the curve flattens.
    """
    x = xp.maximum(x, 0.0)
    e = xp.exp(-lam * x)
    return (1.0 - e) / (1.0 + e)


def michaelis_menten(x, half_sat, xp=np):
    """Michaelis-Menten saturation ``x / (x + K)`` (Hill with slope = 1)."""
    x = xp.maximum(x, 0.0)
    return x / (x + half_sat)


# ---------------------------------------------------------------------------
# Convenience dispatchers used by the model and the analysis code
# ---------------------------------------------------------------------------

ADSTOCK_TYPES = ("geometric", "delayed", "weibull")
SATURATION_TYPES = ("hill", "logistic", "michaelis_menten")


def adstock_weights(adstock_type: str, params: dict, l_max: int, normalize: bool = True, xp=np):
    """Return adstock weights for the configured adstock type."""
    if adstock_type == "geometric":
        return geometric_weights(params["adstock_alpha"], l_max, normalize, xp)
    if adstock_type == "delayed":
        return delayed_weights(params["adstock_alpha"], params["adstock_theta"], l_max, normalize, xp)
    if adstock_type == "weibull":
        return weibull_pdf_weights(
            params["adstock_shape"], params["adstock_scale"], l_max, normalize, xp
        )
    raise ValueError(f"Unknown adstock type '{adstock_type}'. Choose from {ADSTOCK_TYPES}")


def saturate(saturation_type: str, x, params: dict, xp=np, expand: bool = False):
    """Apply the configured saturation function.

    If ``expand`` is True, parameters with shape (D, C) are expanded to (D, 1, C)
    so they broadcast against data with shape (D, T, C).
    """

    def p(name):
        v = params[name]
        return xp.expand_dims(v, -2) if expand else v

    if saturation_type == "hill":
        return hill(x, p("sat_half"), p("sat_slope"), xp)
    if saturation_type == "logistic":
        return logistic(x, p("sat_lam"), xp)
    if saturation_type == "michaelis_menten":
        return michaelis_menten(x, p("sat_half"), xp)
    raise ValueError(
        f"Unknown saturation type '{saturation_type}'. Choose from {SATURATION_TYPES}"
    )
