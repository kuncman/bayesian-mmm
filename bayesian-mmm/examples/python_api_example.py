"""Using the package from Python instead of the CLI.

Shows: building + fitting, reading results, a custom what-if scenario, and
re-running the optimiser with a different budget / locked channels.

    python examples/python_api_example.py
"""

import numpy as np

from bayesian_mmm import analysis, diagnostics, optimize_budget
from bayesian_mmm.config import load_config
from bayesian_mmm.pipeline import build_model

# 1. Load the config and override anything you like in code
cfg = load_config("config/config.yaml", {"sampler": {"draws": 500, "tune": 800, "chains": 2}})

# 2. Build and fit
mmm = build_model(cfg)
mmm.sample_prior_predictive()
mmm.fit()

# 3. Check convergence before looking at anything else
conv = diagnostics.convergence_report(mmm)
print("Converged:", conv["converged"], conv["checks"])

# 4. Channel KPIs
summary = analysis.channel_summary(mmm, reference_periods=52)
print(summary[["roi", "roi_lo", "roi_hi", "marginal_roi", "prob_roi_gt_1"]].round(2))

# 5. What-if: what would the last 13 weeks have looked like with 20% more TV?
post = mmm.posterior(max_draws=1000)
base = mmm.forward(post)
media = mmm.data.media.copy()
tv = mmm.data.channels.index("tv")
media[-13:, tv] *= 1.2
scenario = mmm.forward(post, media=media)
extra = (scenario["mu"] - base["mu"])[:, -13:].sum(axis=1)  # carry-over past the last date is not counted
extra_spend = mmm.data.media[-13:, tv].sum() * 0.2
lo, hi = np.quantile(extra, [0.03, 0.97])
print(f"+20% TV: extra outcome {extra.mean():,.0f} [{lo:,.0f}, {hi:,.0f}] for {extra_spend:,.0f} extra spend")

# 6. Optimise a 10% bigger budget while keeping radio fixed at its current level
current = analysis.reference_spend(mmm, 52)
alloc, s = optimize_budget(
    mmm,
    budget=current.sum() * 1.1,
    reference_periods=52,
    fixed_channels={"radio": float(current[mmm.data.channels.index("radio")])},
)
print(alloc.round(0))
print(f"Expected uplift: {s['expected_uplift_pct']:+.1f}% (P>0 = {s['prob_uplift_positive']:.2f})")
