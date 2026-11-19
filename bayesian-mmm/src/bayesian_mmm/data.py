"""Data loading, validation, train/holdout split, scaling and feature engineering.

All scalers are fit on the *training* window only, so the holdout window is a
genuine out-of-sample test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------


def load_dataframe(cfg: dict) -> pd.DataFrame:
    """Read the CSV given in ``cfg['data']['path']`` and run sanity checks."""
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    return validate_dataframe(df, cfg)


def validate_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    required = [d["date_col"], d["target_col"], *d["channel_cols"], *d["control_cols"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from the data file: {missing}")

    df = df[required].copy()
    df[d["date_col"]] = pd.to_datetime(df[d["date_col"]])
    df = df.sort_values(d["date_col"]).reset_index(drop=True)

    if df[d["date_col"]].duplicated().any():
        raise ValueError("Duplicate dates found - aggregate to one row per period first")
    nulls = df.isna().sum()
    if nulls.any():
        raise ValueError(f"Missing values found:\n{nulls[nulls > 0]}")
    neg = (df[d["channel_cols"]] < 0).any()
    if neg.any():
        raise ValueError(f"Negative media values in: {list(neg[neg].index)}")
    zero_channels = [c for c in d["channel_cols"] if df[c].sum() == 0]
    if zero_channels:
        raise ValueError(f"Channels with zero spend for the whole period: {zero_channels}")
    return df


def load_lift_tests(path: str | None, channels: list[str]) -> pd.DataFrame | None:
    """Load optional lift-test / geo-experiment results used to calibrate the model.

    Expected columns: ``channel, spend_before, spend_after, delta_y, delta_y_sd``
    (all per-period values in original units).
    """
    if not path:
        return None
    lt = pd.read_csv(path)
    needed = {"channel", "spend_before", "spend_after", "delta_y", "delta_y_sd"}
    if not needed.issubset(lt.columns):
        raise ValueError(f"Lift test file must contain columns {sorted(needed)}")
    unknown = set(lt["channel"]) - set(channels)
    if unknown:
        raise ValueError(f"Lift tests reference unknown channels: {unknown}")
    if (lt["delta_y_sd"] <= 0).any():
        raise ValueError("delta_y_sd must be strictly positive")
    return lt.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prepared data container
# ---------------------------------------------------------------------------


@dataclass
class MMMData:
    """Everything the model needs, for the full period (train + holdout)."""

    dates: pd.DatetimeIndex
    y: np.ndarray  # (T,)
    media: np.ndarray  # (T, C) raw spend
    controls: np.ndarray  # (T, K) raw values
    channels: list[str]
    control_names: list[str]
    n_train: int
    period_days: float
    fourier_order: int
    n_changepoints: int
    changepoint_range: float
    # scalers (fit on train)
    y_scale: float = field(init=False)
    media_scale: np.ndarray = field(init=False)
    control_mean: np.ndarray = field(init=False)
    control_std: np.ndarray = field(init=False)

    def __post_init__(self):
        tr = slice(0, self.n_train)
        self.y_scale = float(np.abs(self.y[tr]).max())
        ms = self.media[tr].max(axis=0)
        self.media_scale = np.where(ms > 0, ms, 1.0)
        if self.controls.shape[1]:
            c = self.controls[tr]
            is_binary = np.all(np.isin(c, [0, 1]), axis=0)
            mean = np.where(is_binary, 0.0, c.mean(axis=0))
            std = np.where(is_binary, 1.0, c.std(axis=0))
            self.control_mean = mean
            self.control_std = np.where(std > 0, std, 1.0)
        else:
            self.control_mean = np.zeros(0)
            self.control_std = np.ones(0)

    # ------------------------------------------------------------------ sizes
    @property
    def n_periods(self) -> int:
        return len(self.y)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_holdout(self) -> int:
        return self.n_periods - self.n_train

    # --------------------------------------------------------------- scaling
    def scale_media(self, media: np.ndarray) -> np.ndarray:
        return media / self.media_scale

    def scaled_controls(self) -> np.ndarray:
        return (self.controls - self.control_mean) / self.control_std

    @property
    def y_scaled(self) -> np.ndarray:
        return self.y / self.y_scale

    # --------------------------------------------------------- time features
    def time_index(self) -> np.ndarray:
        """Time in [0, 1] over the training window (>1 in the holdout)."""
        days = (self.dates - self.dates[0]).days.to_numpy(dtype=float)
        span = days[self.n_train - 1] if self.n_train > 1 else 1.0
        return days / span

    def changepoints(self) -> np.ndarray:
        if self.n_changepoints <= 0:
            return np.zeros(0)
        return np.linspace(0, self.changepoint_range, self.n_changepoints + 2)[1:-1]

    def fourier_features(self) -> np.ndarray:
        """Yearly Fourier terms: [sin(2πk t/P), cos(2πk t/P)] for k = 1..order."""
        if self.fourier_order <= 0:
            return np.zeros((self.n_periods, 0))
        days = (self.dates - pd.Timestamp("2000-01-01")).days.to_numpy(dtype=float)
        feats = []
        for k in range(1, self.fourier_order + 1):
            arg = 2 * np.pi * k * days / self.period_days
            feats += [np.sin(arg), np.cos(arg)]
        return np.column_stack(feats)

    def fourier_names(self) -> list[str]:
        return [f"{f}_{k}" for k in range(1, self.fourier_order + 1) for f in ("sin", "cos")]

    # --------------------------------------------------------- convenience
    def train_slice(self) -> slice:
        return slice(0, self.n_train)

    def spend_share(self) -> np.ndarray:
        tot = self.media[: self.n_train].sum(axis=0)
        return tot / tot.sum()

    def summary(self) -> pd.DataFrame:
        tr = self.train_slice()
        return pd.DataFrame(
            {
                "total_spend_train": self.media[tr].sum(axis=0),
                "spend_share_train": self.spend_share(),
                "mean_spend_per_period": self.media[tr].mean(axis=0),
                "pct_periods_with_spend": (self.media[tr] > 0).mean(axis=0) * 100,
            },
            index=self.channels,
        )


def prepare_data(df: pd.DataFrame, cfg: dict) -> MMMData:
    """Build an :class:`MMMData` from a validated dataframe and the config."""
    d, m = cfg["data"], cfg["model"]
    n = len(df)
    holdout = int(d.get("holdout_periods") or 0)
    if holdout < 0 or holdout >= n - 10:
        raise ValueError("holdout_periods must be >= 0 and leave enough training data")
    trend = m["trend"]
    return MMMData(
        dates=pd.DatetimeIndex(df[d["date_col"]]),
        y=df[d["target_col"]].to_numpy(dtype=float),
        media=df[d["channel_cols"]].to_numpy(dtype=float),
        controls=df[d["control_cols"]].to_numpy(dtype=float).reshape(n, len(d["control_cols"])),
        channels=list(d["channel_cols"]),
        control_names=list(d["control_cols"]),
        n_train=n - holdout,
        period_days=float(m["seasonality"]["period_days"]),
        fourier_order=int(m["seasonality"]["yearly_fourier_order"]),
        n_changepoints=int(trend.get("n_changepoints", 0)) if trend["type"] == "piecewise" else 0,
        changepoint_range=float(trend.get("changepoint_range", 0.8)),
    )
