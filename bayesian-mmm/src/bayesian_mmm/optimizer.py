"""Bayesian budget optimisation.

We choose per-period spend ``s_c`` for each channel to maximise the *posterior
expected* total media contribution, subject to

* a fixed total budget  ``sum_c s_c = B``
* per-channel bounds     ``lb_c <= s_c <= ub_c`` (by default 50%-150% of current spend,
  which keeps the recommendation inside the range the data has actually seen).

Because we optimise the expectation over posterior draws (rather than a single
point estimate), uncertainty in the response curves is propagated into the
recommendation and its predicted uplift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .analysis import CI, reference_spend
from .model import BayesianMMM


def optimize_budget(
    mmm: BayesianMMM,
    budget: float | None = None,
    reference_periods: int = 13,
    lower_bound_multiplier: float = 0.5,
    upper_bound_multiplier: float = 1.5,
    horizon_periods: int = 13,
    max_draws: int = 1000,
    fixed_channels: dict[str, float] | None = None,
):
    """Find the per-period budget split that maximises expected media contribution.

    Parameters
    ----------
    budget : total spend per period. Defaults to the current (reference) total.
    fixed_channels : optional ``{channel: spend}`` to lock some channels in place.

    Returns
    -------
    (allocation DataFrame, summary dict)
    """
    d = mmm.data
    post = mmm.posterior(max_draws)
    current = reference_spend(mmm, reference_periods)
    budget = float(current.sum() if budget is None else budget)

    lb = current * lower_bound_multiplier
    ub = current * upper_bound_multiplier
    # channels with no recent spend: allow up to their historical average
    hist_mean = d.media[: d.n_train].mean(0)
    ub = np.where(current == 0, hist_mean * upper_bound_multiplier, ub)
    for ch, val in (fixed_channels or {}).items():
        i = d.channels.index(ch)
        lb[i] = ub[i] = float(val)

    if lb.sum() > budget or ub.sum() < budget:
        raise ValueError(
            f"Budget {budget:,.0f} is infeasible with bounds [{lb.sum():,.0f}, {ub.sum():,.0f}]. "
            "Relax the bound multipliers or change the budget."
        )

    scale = budget  # optimise on a unit scale for numerical stability

    def neg_expected(s_unit):
        resp = mmm.steady_state_response(post, s_unit * scale)  # (D, C)
        return -resp.sum(1).mean() / scale

    x0 = np.clip(current * budget / max(current.sum(), 1e-9), lb, ub) / scale
    res = minimize(
        neg_expected,
        x0,
        method="SLSQP",
        bounds=list(zip(lb / scale, ub / scale)),
        constraints=[{"type": "eq", "fun": lambda s: s.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    opt = res.x * scale

    cur_scaled = current * budget / current.sum() if current.sum() > 0 else current
    resp_cur = mmm.steady_state_response(post, cur_scaled)  # same total budget, current mix
    resp_opt = mmm.steady_state_response(post, opt)

    alloc = pd.DataFrame(
        {
            "current_spend": cur_scaled,
            "optimal_spend": opt,
            "change_pct": (opt / np.where(cur_scaled > 0, cur_scaled, np.nan) - 1) * 100,
            "lower_bound": lb,
            "upper_bound": ub,
            "current_contribution": resp_cur.mean(0),
            "optimal_contribution": resp_opt.mean(0),
        },
        index=d.channels,
    )
    alloc.index.name = "channel"

    tot_cur = resp_cur.sum(1) * horizon_periods
    tot_opt = resp_opt.sum(1) * horizon_periods
    uplift = tot_opt - tot_cur
    summary = {
        "success": bool(res.success),
        "message": str(res.message),
        "budget_per_period": budget,
        "horizon_periods": horizon_periods,
        "expected_contribution_current": float(tot_cur.mean()),
        "expected_contribution_optimal": float(tot_opt.mean()),
        "expected_uplift": float(uplift.mean()),
        "expected_uplift_pct": float(uplift.mean() / tot_cur.mean() * 100),
        "uplift_ci": [float(x) for x in np.quantile(uplift, CI)],
        "prob_uplift_positive": float((uplift > 0).mean()),
    }
    return alloc, summary
