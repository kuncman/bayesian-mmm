"""Generate a realistic synthetic MMM dataset with a KNOWN ground truth.

Because we know the true adstock, saturation and ROI of every channel, we can
check whether the model recovers them - the single best way to build trust in
an MMM before pointing it at real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import transforms as tf

TRUE_CHANNELS = {
    #            carry-over  half-sat (raw $/wk)  Hill slope  true ROI
    "tv":       {"alpha": 0.70, "half_sat": 32000.0, "slope": 2.0, "roi": 1.6},
    "search":   {"alpha": 0.15, "half_sat": 14000.0, "slope": 1.2, "roi": 3.2},
    "social":   {"alpha": 0.40, "half_sat": 12000.0, "slope": 1.5, "roi": 2.0},
    "display":  {"alpha": 0.30, "half_sat": 9000.0,  "slope": 1.0, "roi": 0.6},
    "radio":    {"alpha": 0.50, "half_sat": 10000.0, "slope": 1.8, "roi": 1.1},
}
L_MAX = 8


def _flighted(rng, n, on_prob, level, sd, min_len=3, max_len=8):
    """Bursty "flighting" spend pattern typical of TV / radio."""
    out = np.zeros(n)
    t = 0
    while t < n:
        length = rng.integers(min_len, max_len + 1)
        if rng.random() < on_prob:
            out[t : t + length] = np.clip(rng.normal(level, sd, size=min(length, n - t)), 0, None)
        t += length
    return out


def simulate(n_weeks: int = 156, start: str = "2022-01-03", seed: int = 7, holdout: int = 12):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)
    doy = dates.dayofyear.to_numpy()
    season_wave = np.sin(2 * np.pi * (doy - 80) / 365.25)  # peaks ~ late June
    q4 = np.exp(-0.5 * ((doy - 340) / 20.0) ** 2)  # holiday ramp

    # ---------------------------------------------------------------- spend
    media = pd.DataFrame(index=dates)
    media["tv"] = _flighted(rng, n_weeks, 0.55, 42000, 8000)
    media["search"] = np.clip(15000 + 4000 * season_wave + 6000 * q4 + rng.normal(0, 1800, n_weeks), 2000, None)
    media["social"] = np.clip(11000 + 0.03 * t * 100 + rng.normal(0, 2500, n_weeks), 1000, None)
    media["display"] = np.clip(rng.gamma(4.0, 1500.0, n_weeks) * (rng.random(n_weeks) < 0.8), 0, None)
    media["radio"] = _flighted(rng, n_weeks, 0.45, 14000, 3000, 2, 6)
    media = media.round(2)

    # ---------------------------------------------------------- controls
    price = 1.0 + np.cumsum(rng.normal(0, 0.01, n_weeks))
    price = price - (price.mean() - 1.0)
    competitor_promo = (rng.random(n_weeks) < 0.15).astype(int)
    holiday = np.isin(dates.isocalendar().week.to_numpy(), [47, 48, 51, 52]).astype(int)

    # ---------------------------------------------------------- baseline
    trend = 200000 + 250 * t - 150 * np.maximum(0, t - 100)  # growth slows after week 100
    seasonality = 9000 * season_wave + 3000 * np.cos(4 * np.pi * doy / 365.25)
    ctrl_effect = -60000 * (price - 1.0) - 6000 * competitor_promo + 22000 * holiday

    # ---------------------------------------------------------- media effects
    n_train = n_weeks - holdout
    x = media.to_numpy()
    x_lag = tf.build_lagged_media(x, L_MAX)
    alpha = np.array([v["alpha"] for v in TRUE_CHANNELS.values()])
    half = np.array([v["half_sat"] for v in TRUE_CHANNELS.values()])
    slope = np.array([v["slope"] for v in TRUE_CHANNELS.values()])
    roi = np.array([v["roi"] for v in TRUE_CHANNELS.values()])

    w = tf.geometric_weights(alpha, L_MAX, normalize=True)
    sat = tf.hill(tf.apply_adstock(x_lag, w), half, slope)  # (T, C) in [0, 1)
    # choose the effect size so the TRAINING-period ROI equals the target ROI
    beta = roi * x[:n_train].sum(0) / sat[:n_train].sum(0)
    effects = sat * beta

    noise = rng.normal(0, 5000, n_weeks)
    revenue = trend + seasonality + ctrl_effect + effects.sum(1) + noise

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "revenue": revenue.round(2),
            **{c: media[c].to_numpy() for c in media},
            "price_index": price.round(4),
            "competitor_promo": competitor_promo,
            "holiday": holiday,
        }
    )
    contributions = pd.DataFrame(effects, columns=list(TRUE_CHANNELS), index=df["date"])
    contributions["baseline"] = trend + seasonality + ctrl_effect

    # ---------------------------------------------------------- lift tests
    # Two simulated geo-experiments measured at steady state.
    lift_rows = []
    for ch, before, after in [("search", 15000, 20000), ("tv", 30000, 45000)]:
        i = list(TRUE_CHANNELS).index(ch)
        true_lift = beta[i] * (
            tf.hill(np.array(after), half[i], slope[i]) - tf.hill(np.array(before), half[i], slope[i])
        )
        sd = 0.2 * abs(true_lift)
        lift_rows.append(
            {
                "channel": ch,
                "spend_before": before,
                "spend_after": after,
                "delta_y": round(float(true_lift + rng.normal(0, sd)), 2),
                "delta_y_sd": round(float(sd), 2),
            }
        )
    lift = pd.DataFrame(lift_rows)

    truth = {
        "l_max": L_MAX,
        "adstock": "geometric (normalised)",
        "saturation": "hill",
        "n_train": n_train,
        "channels": {
            ch: {**TRUE_CHANNELS[ch], "beta": float(beta[i]), "train_contribution": float(effects[:n_train, i].sum())}
            for i, ch in enumerate(TRUE_CHANNELS)
        },
        "controls": {"price_index": -60000, "competitor_promo": -6000, "holiday": 22000},
        "noise_sd": 5000,
    }
    return df, contributions, lift, truth


def write_synthetic(out_dir: str | Path = "data", **kwargs):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, contrib, lift, truth = simulate(**kwargs)
    df.to_csv(out / "mmm_synthetic.csv", index=False)
    contrib.to_csv(out / "true_contributions.csv")
    lift.to_csv(out / "lift_tests.csv", index=False)
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return df, contrib, lift, truth
