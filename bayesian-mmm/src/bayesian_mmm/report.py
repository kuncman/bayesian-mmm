"""Write a human-readable Markdown report summarising a fitted model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame, floatfmt: str = ",.2f") -> str:
    def fmt(v):
        if isinstance(v, (float, int)) and not isinstance(v, bool):
            return format(v, floatfmt)
        return str(v)

    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, results: dict) -> Path:
    conv = results["convergence"]
    fit = results["fit_metrics"]
    summ = results["channel_summary"]
    opt_alloc, opt_sum = results.get("budget", (None, None))

    status = "PASSED" if conv["converged"] else "NEEDS ATTENTION"
    md = [
        "# Bayesian Marketing Mix Model - Results Report",
        "",
        f"*Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}*",
        "",
        "## 1. Sampler convergence: " + status,
        "",
        "| check | value | threshold | ok |",
        "|---|---|---|---|",
        f"| divergences | {conv['divergences']} | 0 | {conv['checks']['no_divergences']} |",
        f"| max R-hat | {conv['max_r_hat']:.3f} ({conv['worst_r_hat_param']}) | <= 1.01 | {conv['checks']['r_hat_ok']} |",
        f"| min ESS (bulk / tail) | {conv['min_ess_bulk']:.0f} / {conv['min_ess_tail']:.0f} | >= 400 | {conv['checks']['ess_ok']} |",
        f"| min E-BFMI | {conv['min_bfmi']:.2f} | >= 0.3 | {conv['checks']['bfmi_ok']} |",
        "",
        "If any check fails, see *Troubleshooting* in the README before trusting the numbers below.",
        "",
        "## 2. Model fit",
        "",
        _table(pd.DataFrame(fit).T, ",.3f"),
        "",
        "* `coverage_94_pct` should be close to 94 - lower means the model is over-confident.",
        "* Holdout metrics are true out-of-sample performance (the model never saw these periods).",
        "",
        "![fit](plots/04_posterior_predictive.png)",
        "",
        "## 3. Channel performance (training window)",
        "",
        _table(
            summ[["spend", "contribution", "roi", "roi_lo", "roi_hi", "prob_roi_gt_1",
                  "marginal_roi", "mean_carryover_lag", "spend_share_pct", "contribution_share_pct"]]
        ),
        "",
        "* **roi** = incremental outcome per unit spend over the training window (94% interval in `roi_lo`/`roi_hi`).",
        "* **marginal_roi** = return on the *next* unit of spend at the current spend level - the number that matters for reallocating budget.",
        "* **prob_roi_gt_1** = posterior probability the channel pays back.",
        "",
        "![roi](plots/09_roi.png)",
        "",
        "![decomposition](plots/06_decomposition.png)",
        "",
        "![response curves](plots/12_response_curves.png)",
        "",
    ]
    if opt_alloc is not None:
        lo, hi = opt_sum["uplift_ci"]
        md += [
            "## 4. Budget optimisation",
            "",
            f"Budget per period: **{opt_sum['budget_per_period']:,.0f}**, horizon {opt_sum['horizon_periods']} periods.",
            "",
            f"Expected uplift from reallocating: **{opt_sum['expected_uplift']:,.0f}** "
            f"({opt_sum['expected_uplift_pct']:+.1f}%), 94% interval [{lo:,.0f}, {hi:,.0f}], "
            f"P(uplift > 0) = {opt_sum['prob_uplift_positive']:.2f}.",
            "",
            _table(opt_alloc[["current_spend", "optimal_spend", "change_pct", "current_contribution", "optimal_contribution"]]),
            "",
            "![budget](plots/13_budget_allocation.png)",
            "",
        ]
    if "recovery" in results and results["recovery"] is not None and len(results["recovery"]):
        rec = results["recovery"]
        md += [
            "## 5. Ground-truth recovery (synthetic data)",
            "",
            f"{rec['covered_by_94_ci'].mean() * 100:.0f}% of true parameters fall inside the 94% credible interval.",
            "",
            _table(rec.set_index("channel"), ",.3f"),
            "",
            "![recovery](plots/15_parameter_recovery.png)",
            "",
        ]
    md += [
        "## Files",
        "",
        "* `tables/` - every number in this report as CSV",
        "* `plots/` - all charts",
        "* `diagnostics.json` - convergence + fit metrics",
        "* `trace.nc` - full posterior (open with `arviz.from_netcdf`)",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
