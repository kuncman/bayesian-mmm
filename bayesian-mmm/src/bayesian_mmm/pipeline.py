"""End-to-end pipeline: load -> validate -> build -> prior check -> fit -> diagnose
-> decompose -> ROI -> response curves -> optimise budget -> report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import yaml

from . import analysis, diagnostics, plotting
from .config import load_config
from .data import load_dataframe, load_lift_tests, prepare_data
from .model import BayesianMMM
from .optimizer import optimize_budget
from .report import write_report

log = logging.getLogger("bayesian_mmm")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def build_model(cfg: dict) -> BayesianMMM:
    df = load_dataframe(cfg)
    data = prepare_data(df, cfg)
    lift = load_lift_tests(cfg["data"].get("lift_tests_path"), data.channels)
    mmm = BayesianMMM(cfg, data, lift)
    mmm.build()
    return mmm


def run_pipeline(config_path: str | None = None, overrides: dict | None = None) -> dict:
    t0 = time.time()
    cfg = load_config(config_path, overrides)
    out = Path(cfg["output"]["dir"])
    plots, tables = out / "plots", out / "tables"
    for p in (plots, tables):
        p.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["output"]["dpi"])
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # ---------------------------------------------------------------- 1. data
    log.info("Step 1/9  Loading and validating data from %s", cfg["data"]["path"])
    mmm = build_model(cfg)
    d = mmm.data
    log.info("          %d periods (%d train, %d holdout), %d channels, %d controls",
             d.n_periods, d.n_train, d.n_holdout, d.n_channels, len(d.control_names))
    d.summary().to_csv(tables / "data_summary.csv")
    plotting.plot_data_overview(mmm, plots / "01_data_overview.png", dpi)
    plotting.plot_spend_correlation(mmm, plots / "02_spend_correlation.png", dpi)

    # ------------------------------------------------------- 2. prior check
    log.info("Step 2/9  Prior predictive check")
    mmm.sample_prior_predictive()
    prior_check = diagnostics.prior_predictive_check(mmm)
    plotting.plot_prior_predictive(mmm, plots / "03_prior_predictive.png", dpi)

    # -------------------------------------------------------------- 3. fit
    s = cfg["sampler"]
    log.info("Step 3/9  Sampling posterior with NUTS (%d chains x %d draws, %d tune)",
             s["chains"], s["draws"], s["tune"])
    mmm.fit()

    # -------------------------------------------------------- 4. diagnose
    log.info("Step 4/9  Convergence diagnostics")
    conv = diagnostics.convergence_report(mmm)
    diagnostics.parameter_summary(mmm).to_csv(tables / "parameter_summary.csv")
    fit = diagnostics.fit_metrics(mmm)
    if not conv["converged"]:
        log.warning("          Sampler did NOT pass all convergence checks: %s", conv["checks"])
    log.info("          R2 train=%.3f  MAPE train=%.2f%%", fit["train"]["r2"], fit["train"]["mape_pct"])
    if "holdout" in fit:
        log.info("          R2 holdout=%.3f  MAPE holdout=%.2f%%", fit["holdout"]["r2"], fit["holdout"]["mape_pct"])
    plotting.plot_trace(mmm, plots / "05_trace.png", dpi)
    plotting.plot_posterior_predictive(mmm, plots / "04_posterior_predictive.png", dpi)
    plotting.plot_residuals(mmm, plots / "05b_residuals.png", dpi)

    # ------------------------------------------------------ 5. decompose
    log.info("Step 5/9  Decomposing the outcome into components")
    decomp = analysis.decomposition(mmm)
    decomp.to_csv(tables / "decomposition.csv")
    shares = analysis.contribution_shares(decomp)
    shares.to_csv(tables / "contribution_shares.csv")
    plotting.plot_decomposition(mmm, decomp, plots / "06_decomposition.png", dpi)
    plotting.plot_baseline_components(mmm, decomp, plots / "07_baseline_components.png", dpi)
    plotting.plot_contribution_waterfall(shares, plots / "08_waterfall.png", dpi)

    # -------------------------------------------------------- 6. channel KPIs
    log.info("Step 6/9  ROI, marginal ROI and carry-over per channel")
    ref_periods = int(cfg["optimizer"]["reference_periods"])
    summ = analysis.channel_summary(mmm, ref_periods)
    summ.to_csv(tables / "channel_summary.csv")
    plotting.plot_roi(summ, plots / "09_roi.png", dpi)
    plotting.plot_share_comparison(summ, plots / "10_share_comparison.png", dpi)
    pvp = analysis.prior_vs_posterior(mmm, "roi" if mmm.media_prior == "roi" else "beta")
    pvp.to_csv(tables / "prior_vs_posterior.csv")
    plotting.plot_prior_posterior(mmm, plots / "14_prior_vs_posterior.png",
                                  "roi" if mmm.media_prior == "roi" else "beta", dpi)

    # ------------------------------------------------ 7. curves
    log.info("Step 7/9  Adstock and response curves")
    ad = analysis.adstock_curves(mmm)
    ad.to_csv(tables / "adstock_curves.csv", index=False)
    plotting.plot_adstock(ad, plots / "11_adstock.png", dpi)
    rc = analysis.response_curves(mmm)
    rc.to_csv(tables / "response_curves.csv", index=False)
    plotting.plot_response_curves(rc, summ, plots / "12_response_curves.png", dpi)

    # ------------------------------------------------- 8. optimise
    log.info("Step 8/9  Optimising budget allocation")
    o = cfg["optimizer"]
    alloc, opt_sum = optimize_budget(
        mmm,
        budget=o.get("budget"),
        reference_periods=ref_periods,
        lower_bound_multiplier=float(o["lower_bound_multiplier"]),
        upper_bound_multiplier=float(o["upper_bound_multiplier"]),
        horizon_periods=int(o["horizon_periods"]),
    )
    alloc.to_csv(tables / "budget_allocation.csv")
    plotting.plot_budget(alloc, plots / "13_budget_allocation.png", dpi)
    log.info("          Expected uplift %+.1f%% (P>0 = %.2f)", opt_sum["expected_uplift_pct"], opt_sum["prob_uplift_positive"])

    # ---------------------------------------------- 9. recovery + report
    log.info("Step 9/9  Writing report")
    rec = None
    truth = cfg["data"].get("ground_truth_path")
    if truth and Path(truth).exists():
        rec = analysis.parameter_recovery(mmm, truth, summ)
        rec.to_csv(tables / "parameter_recovery.csv", index=False)
        plotting.plot_recovery(rec, plots / "15_parameter_recovery.png", dpi)

    results = {
        "convergence": conv,
        "fit_metrics": fit,
        "prior_check": prior_check,
        "channel_summary": summ,
        "budget": (alloc, opt_sum),
        "recovery": rec,
    }
    (out / "diagnostics.json").write_text(
        json.dumps(_jsonable({"convergence": conv, "fit_metrics": fit, "prior_check": prior_check,
                              "budget_optimisation": opt_sum}), indent=2)
    )
    if cfg["output"].get("save_trace", True):
        try:
            mmm.idata.to_netcdf(str(out / "trace.nc"))
        except Exception as exc:  # pragma: no cover - depends on netCDF backend
            log.warning("Could not save trace (%s). Install h5netcdf or netCDF4.", exc)
    report = write_report(out, results)
    log.info("Done in %.1f min. Report: %s", (time.time() - t0) / 60, report)
    results["mmm"] = mmm
    return results
