"""All charts produced by the pipeline (matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .model import BayesianMMM, get_group  # noqa: E402

# Categorical palette - assigned to channels in a FIXED order so a channel keeps its colour
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
SURFACE = "#fcfcfb"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


_style()


def color_map(channels: list[str]) -> dict[str, str]:
    return {ch: PALETTE[i % len(PALETTE)] for i, ch in enumerate(channels)}


def _save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt_thousands(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k" if abs(v) >= 1000 else f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _grid(n, ncols=3, w=4.2, h=3.0):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------


def plot_data_overview(mmm: BayesianMMM, path: Path, dpi=130):
    d = mmm.data
    cmap = color_map(d.channels)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    axes[0].plot(d.dates, d.y, color=INK, lw=1.6)
    axes[0].set_title("Target over time")
    _fmt_thousands(axes[0])
    for ch, col in zip(d.channels, d.media.T):
        axes[1].plot(d.dates, col, color=cmap[ch], lw=1.4, label=ch)
    axes[1].set_title("Media spend per period")
    axes[1].legend(ncol=len(d.channels), loc="upper left", bbox_to_anchor=(0, -0.08))
    _fmt_thousands(axes[1])
    for ax in axes:
        if d.n_holdout:
            ax.axvspan(d.dates[d.n_train], d.dates[-1], color=GRID, alpha=0.5, lw=0)
    return _save(fig, path, dpi)


def plot_spend_correlation(mmm: BayesianMMM, path: Path, dpi=130):
    """Correlation between media channels - high correlation = hard to separate effects."""
    d = mmm.data
    corr = np.corrcoef(d.media[: d.n_train].T)
    n = len(d.channels)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 1.0 * n + 1.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), d.channels, rotation=45, ha="right")
    ax.set_yticks(range(n), d.channels)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spend correlation between channels")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# 2. Prior / posterior predictive
# ---------------------------------------------------------------------------


def _band(ax, x, draws, color, label=None):
    lo, hi = np.quantile(draws, [0.03, 0.97], axis=0)
    lo50, hi50 = np.quantile(draws, [0.25, 0.75], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, label=f"94% interval" if label else None)
    ax.fill_between(x, lo50, hi50, color=color, alpha=0.35, lw=0, label=f"50% interval" if label else None)


def plot_prior_predictive(mmm: BayesianMMM, path: Path, dpi=130):
    d = mmm.data
    pp = get_group(mmm.prior_idata, "prior_predictive")["y"].values
    pp = pp.reshape(-1, pp.shape[-1]) * d.y_scale
    x = d.dates[: d.n_train]
    fig, ax = plt.subplots(figsize=(12, 4))
    _band(ax, x, pp, PALETTE[0], label=True)
    ax.plot(x, d.y[: d.n_train], color=INK, lw=1.5, label="observed")
    ax.set_title("Prior predictive check - do the priors generate plausible data?")
    ax.legend(loc="upper left")
    _fmt_thousands(ax)
    return _save(fig, path, dpi)


def plot_posterior_predictive(mmm: BayesianMMM, path: Path, dpi=130, max_draws=1000):
    d = mmm.data
    post = mmm.posterior(max_draws)
    comps = mmm.forward(post)
    pred = mmm.predictive_samples(post, comps["mu"])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    _band(ax, d.dates, pred, PALETTE[0], label=True)
    ax.plot(d.dates, comps["mu"].mean(0), color=PALETTE[0], lw=1.6, label="posterior mean")
    ax.plot(d.dates, d.y, color=INK, lw=1.2, marker="o", ms=2.5, label="observed")
    if d.n_holdout:
        ax.axvline(d.dates[d.n_train], color=NEUTRAL, ls="--", lw=1)
        ax.text(d.dates[d.n_train], ax.get_ylim()[1], "  holdout", va="top", color=INK_2)
    ax.set_title("Posterior predictive fit")
    ax.legend(loc="upper left", ncol=4)
    _fmt_thousands(ax)
    return _save(fig, path, dpi)


def plot_residuals(mmm: BayesianMMM, path: Path, dpi=130, max_draws=1000):
    d = mmm.data
    comps = mmm.forward(mmm.posterior(max_draws))
    resid = d.y - comps["mu"].mean(0)
    tr = d.train_slice()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [2.4, 1, 1]})
    axes[0].plot(d.dates, resid, color=PALETTE[0], lw=1.2)
    axes[0].axhline(0, color=NEUTRAL, lw=1)
    if d.n_holdout:
        axes[0].axvline(d.dates[d.n_train], color=NEUTRAL, ls="--", lw=1)
    axes[0].set_title("Residuals over time")
    _fmt_thousands(axes[0])
    axes[1].hist(resid[tr], bins=25, color=PALETTE[0], edgecolor=SURFACE)
    axes[1].set_title("Residual distribution")
    _fmt_thousands(axes[1], "x")
    r = resid[tr] - resid[tr].mean()
    lags = range(1, 14)
    acf = [np.corrcoef(r[:-k], r[k:])[0, 1] for k in lags]
    axes[2].bar(lags, acf, color=PALETTE[0], width=0.6)
    bound = 1.96 / np.sqrt(len(r))
    axes[2].axhspan(-bound, bound, color=GRID, alpha=0.7, lw=0)
    axes[2].set_title("Residual autocorrelation")
    axes[2].set_xlabel("lag")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# 3. MCMC
# ---------------------------------------------------------------------------


def plot_trace(mmm: BayesianMMM, path: Path, dpi=130, var_names=None):
    """Compact trace plot: posterior density (left) and chain traces (right)."""
    ds = get_group(mmm.idata, "posterior")
    var_names = var_names or [v for v in mmm.parameter_names if v in ds and v != "trend_delta"]
    cmap = color_map(mmm.data.channels)
    fig, axes = plt.subplots(len(var_names), 2, figsize=(13, 1.9 * len(var_names)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 2.2]})
    for r, var in enumerate(var_names):
        arr = ds[var].values  # (chain, draw, ...)
        n_chain, n_draw = arr.shape[:2]
        flat = arr.reshape(n_chain, n_draw, -1)
        dim_names = list(ds[var].dims[2:])
        labels = list(ds[var][dim_names[0]].values) if dim_names else [var]
        for k in range(flat.shape[2]):
            col = cmap.get(str(labels[k]), PALETTE[k % len(PALETTE)])
            axes[r, 0].hist(flat[:, :, k].ravel(), bins=40, histtype="step", color=col, lw=1.3, density=True)
            for ch in range(n_chain):
                axes[r, 1].plot(np.arange(n_draw) + ch * n_draw, flat[ch, :, k], color=col, lw=0.4, alpha=0.8)
        axes[r, 0].set_ylabel(var, rotation=0, ha="right", va="center", color=INK)
        axes[r, 0].set_yticks([])
        for ch in range(1, n_chain):
            axes[r, 1].axvline(ch * n_draw, color=NEUTRAL, lw=0.8, ls=":")
    axes[0, 0].set_title("Posterior")
    axes[0, 1].set_title("Traces (chains concatenated)")
    handles = [matplotlib.lines.Line2D([], [], color=c, lw=2, label=ch) for ch, c in cmap.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0 + 0.25 / len(var_names)),
               title="channel-level parameters")
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# 4. Decomposition & channel KPIs
# ---------------------------------------------------------------------------


def plot_decomposition(mmm: BayesianMMM, decomp: pd.DataFrame, path: Path, dpi=130):
    d = mmm.data
    cmap = color_map(d.channels)
    base_cols = ["intercept", "trend", "seasonality"] + [c for c in decomp if c.startswith("control:")]
    base = decomp[base_cols].sum(1).to_numpy()
    media = [decomp[f"media:{ch}"].to_numpy() for ch in d.channels]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        d.dates, base, *media, labels=["baseline (intercept + trend + season + controls)"] + d.channels,
        colors=["#d9d8d3"] + [cmap[ch] for ch in d.channels], edgecolor=SURFACE, linewidth=0.5,
    )
    ax.plot(d.dates, decomp["actual"], color=INK, lw=1.1, label="actual")
    ax.set_title("Contribution decomposition (posterior mean)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(bottom=min(0, base.min()))
    _fmt_thousands(ax)
    return _save(fig, path, dpi)


def plot_baseline_components(mmm: BayesianMMM, decomp: pd.DataFrame, path: Path, dpi=130):
    cols = ["trend", "seasonality"] + [c for c in decomp if c.startswith("control:")]
    fig, axes = _grid(len(cols), ncols=3, w=4.3, h=2.8)
    for ax, col in zip(axes, cols):
        ax.plot(decomp.index, decomp[col], color=PALETTE[0], lw=1.4)
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ax.set_title(col)
        _fmt_thousands(ax)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_contribution_waterfall(shares: pd.DataFrame, path: Path, dpi=130):
    s = shares["total_contribution"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cum = 0.0
    for i, (name, val) in enumerate(s.items()):
        col = PALETTE[0] if name.startswith("media:") else NEUTRAL
        ax.bar(i, val, bottom=cum if val >= 0 else cum + val, color=col, width=0.65)
        cum += val
    ax.bar(len(s), cum, color=INK, width=0.65)
    ax.set_xticks(range(len(s) + 1), list(s.index) + ["total fitted"], rotation=40, ha="right")
    ax.set_title("Waterfall of total contribution (training window) - blue = media")
    _fmt_thousands(ax)
    return _save(fig, path, dpi)


def plot_roi(summary: pd.DataFrame, path: Path, dpi=130):
    ch = list(summary.index)
    cmap = color_map(ch)
    y = np.arange(len(ch))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(ch) + 1.5))
    for yi, c in zip(y, ch):
        r = summary.loc[c]
        ax.plot([r.roi_lo, r.roi_hi], [yi + 0.12] * 2, color=cmap[c], lw=2.5, solid_capstyle="round")
        ax.plot(r.roi, yi + 0.12, "o", color=cmap[c], ms=8, mec=SURFACE, mew=2)
        ax.plot([r.marginal_roi_lo, r.marginal_roi_hi], [yi - 0.12] * 2, color=cmap[c], lw=1.5, alpha=0.6)
        ax.plot(r.marginal_roi, yi - 0.12, "D", color=SURFACE, mec=cmap[c], ms=7, mew=1.8)
    ax.axvline(1, color=NEUTRAL, ls="--", lw=1)
    ax.set_yticks(y, ch)
    ax.set_xlabel("return per unit of spend")
    ax.set_title("ROI (circle) and marginal ROI at current spend (diamond), 94% intervals")
    return _save(fig, path, dpi)


def plot_share_comparison(summary: pd.DataFrame, path: Path, dpi=130):
    ch = list(summary.index)
    x = np.arange(len(ch))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.18, summary["spend_share_pct"], 0.34, color=NEUTRAL, label="share of spend")
    ax.bar(x + 0.18, summary["contribution_share_pct"], 0.34, color=PALETTE[0], label="share of media contribution")
    ax.set_xticks(x, ch)
    ax.set_ylabel("%")
    ax.legend()
    ax.set_title("Spend share vs. effect share")
    return _save(fig, path, dpi)


def plot_adstock(curves: pd.DataFrame, path: Path, dpi=130):
    ch = list(dict.fromkeys(curves["channel"]))
    cmap = color_map(ch)
    fig, axes = _grid(len(ch), ncols=3, w=4.0, h=2.8)
    for ax, c in zip(axes, ch):
        sub = curves[curves.channel == c]
        ax.bar(sub.lag, sub.weight, color=cmap[c], width=0.6)
        ax.errorbar(sub.lag, sub.weight, yerr=[sub.weight - sub.lo, sub.hi - sub.weight], fmt="none",
                    ecolor=INK_2, lw=1, capsize=2)
        ax.set_title(f"{c} - adstock weights")
        ax.set_xlabel("lag (periods)")
        ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_response_curves(curves: pd.DataFrame, summary: pd.DataFrame, path: Path, dpi=130):
    ch = list(dict.fromkeys(curves["channel"]))
    cmap = color_map(ch)
    fig, axes = _grid(len(ch), ncols=3, w=4.3, h=3.1)
    for ax, c in zip(axes, ch):
        sub = curves[curves.channel == c]
        ax.fill_between(sub.spend, sub.lo, sub.hi, color=cmap[c], alpha=0.2, lw=0)
        ax.plot(sub.spend, sub.response, color=cmap[c])
        cur = summary.loc[c, "reference_spend_per_period"]
        ycur = np.interp(cur, sub.spend, sub.response)
        ax.plot(cur, ycur, "o", color=cmap[c], ms=8, mec=SURFACE, mew=2)
        ax.annotate("current", (cur, ycur), textcoords="offset points", xytext=(6, -12), color=INK_2, fontsize=8)
        ax.set_title(f"{c} - response curve")
        ax.set_xlabel("spend per period")
        _fmt_thousands(ax)
        _fmt_thousands(ax, "x")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_budget(alloc: pd.DataFrame, path: Path, dpi=130):
    ch = list(alloc.index)
    x = np.arange(len(ch))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.18, alloc["current_spend"], 0.34, color=NEUTRAL, label="current")
    ax.bar(x + 0.18, alloc["optimal_spend"], 0.34, color=PALETTE[0], label="optimised")
    for xi, (_, r) in zip(x, alloc.iterrows()):
        if np.isfinite(r.change_pct):
            ax.text(xi + 0.18, r.optimal_spend, f"{r.change_pct:+.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, ch)
    ax.legend()
    ax.set_title("Budget allocation per period: current vs. optimised")
    _fmt_thousands(ax)
    return _save(fig, path, dpi)


def plot_prior_posterior(mmm: BayesianMMM, path: Path, var="roi", dpi=130):
    prior = get_group(mmm.prior_idata, "prior")
    if var not in prior:
        return None
    ch = mmm.data.channels
    cmap = color_map(ch)
    pr = prior[var].values.reshape(-1, len(ch))
    po = mmm.posterior()[var]
    fig, axes = _grid(len(ch), ncols=3, w=4.0, h=2.6)
    for c, (ax, name) in enumerate(zip(axes, ch)):
        hi = np.quantile(np.concatenate([pr[:, c], po[:, c]]), 0.99)
        bins = np.linspace(0, hi, 50)
        ax.hist(pr[:, c], bins=bins, density=True, color=NEUTRAL, alpha=0.5, label="prior")
        ax.hist(po[:, c], bins=bins, density=True, color=cmap[name], alpha=0.8, label="posterior")
        ax.set_title(f"{name} - {var}")
        ax.set_yticks([])
        ax.legend()
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_recovery(rec: pd.DataFrame, path: Path, dpi=130):
    params = list(dict.fromkeys(rec["parameter"]))
    fig, axes = _grid(len(params), ncols=2, w=5.5, h=0.55 * rec.channel.nunique() + 1.3)
    chs = list(dict.fromkeys(rec["channel"]))
    cmap = color_map(chs)
    for ax, p in zip(axes, params):
        sub = rec[rec.parameter == p].set_index("channel").loc[chs]
        y = np.arange(len(sub))[::-1]
        for yi, (c, r) in zip(y, sub.iterrows()):
            ax.plot([r.lo, r.hi], [yi, yi], color=cmap[c], lw=2.5)
            ax.plot(r.estimate, yi, "o", color=cmap[c], ms=8, mec=SURFACE, mew=2)
            ax.plot(r.true, yi, "x", color=INK, ms=9, mew=2)
        ax.set_yticks(y, sub.index)
        ax.set_title(f"{p}: estimate (o, 94% CI) vs truth (x)")
        if "spend" in p:
            _fmt_thousands(ax, "x")
    fig.tight_layout()
    return _save(fig, path, dpi)
