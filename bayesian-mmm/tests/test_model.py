import numpy as np
import pymc as pm
import pytest

from bayesian_mmm import analysis, diagnostics
from bayesian_mmm.config import load_config
from bayesian_mmm.data import prepare_data, validate_dataframe
from bayesian_mmm.model import BayesianMMM
from bayesian_mmm.optimizer import optimize_budget


@pytest.mark.parametrize("adstock", ["geometric", "delayed", "weibull"])
@pytest.mark.parametrize("saturation", ["hill", "logistic", "michaelis_menten"])
@pytest.mark.parametrize("media_prior", ["roi", "spend_share"])
def test_every_variant_builds_with_finite_logp(synthetic, cfg, adstock, saturation, media_prior):
    df, _, _ = synthetic
    c = load_config(None, {**cfg, "model": {**cfg["model"], "adstock": {"type": adstock, "l_max": 6},
                                              "saturation": {"type": saturation}, "media_prior": media_prior}})
    d = prepare_data(validate_dataframe(df, c), c)
    m = BayesianMMM(c, d)
    model = m.build()
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)
    assert set(m.adstock_params + m.saturation_params) <= set(model.named_vars)


@pytest.mark.parametrize("trend", ["none", "linear", "piecewise"])
@pytest.mark.parametrize("likelihood", ["normal", "studentt"])
def test_trend_and_likelihood_variants(synthetic, cfg, trend, likelihood):
    df, _, _ = synthetic
    c = load_config(None, {**cfg, "model": {**cfg["model"], "trend": {"type": trend}, "likelihood": likelihood}})
    d = prepare_data(validate_dataframe(df, c), c)
    model = BayesianMMM(c, d).build()
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_forward_matches_pymc_graph_at_prior_draws(cfg, mmm_data):
    """The numpy re-implementation must reproduce the PyMC model exactly."""
    m = BayesianMMM(cfg, mmm_data)
    m.build()
    with m.model:
        prior = pm.sample_prior_predictive(draws=20, random_seed=1)
    from bayesian_mmm.model import get_group, stack_draws

    ds = get_group(prior, "prior")
    params = {v: stack_draws(ds, v) for v in ds.data_vars}
    f = m.forward(params)
    np.testing.assert_allclose(f["mu"][:, : mmm_data.n_train], params["mu"] * mmm_data.y_scale, rtol=1e-6)
    np.testing.assert_allclose(
        f["media"][:, : mmm_data.n_train], params["channel_contribution"] * mmm_data.y_scale, rtol=1e-6
    )


def test_roi_parameterisation_is_consistent(cfg, mmm_data):
    """With media_prior='roi', contribution / spend over the training window must equal roi."""
    m = BayesianMMM(cfg, mmm_data)
    m.build()
    with m.model:
        prior = pm.sample_prior_predictive(draws=10, random_seed=2)
    from bayesian_mmm.model import get_group, stack_draws

    ds = get_group(prior, "prior")
    params = {v: stack_draws(ds, v) for v in ds.data_vars}
    f = m.forward(params)
    roi_implied = f["media"][:, : mmm_data.n_train].sum(1) / mmm_data.media[: mmm_data.n_train].sum(0)
    np.testing.assert_allclose(roi_implied, params["roi"], rtol=1e-5)


# ----------------------------------------------------------------- slow tests


@pytest.mark.slow
def test_fit_and_full_analysis(fitted, tmp_path):
    m = fitted
    rep = diagnostics.convergence_report(m)
    assert {"max_r_hat", "divergences", "min_ess_bulk"} <= set(rep)
    fit = diagnostics.fit_metrics(m)
    assert fit["train"]["r2"] > 0.8
    assert "holdout" in fit

    dec = analysis.decomposition(m)
    comp_cols = [c for c in dec.columns if c not in ("actual", "fitted", "is_holdout")]
    np.testing.assert_allclose(dec[comp_cols].sum(1), dec["fitted"], rtol=1e-8)

    summ = analysis.channel_summary(m, 26)
    assert (summ["roi_lo"] <= summ["roi"]).all() and (summ["roi"] <= summ["roi_hi"]).all()

    rc = analysis.response_curves(m, n_points=10)
    for _, g in rc.groupby("channel"):
        assert np.all(np.diff(g["response"].to_numpy()) >= -1e-9)

    alloc, s = optimize_budget(m, reference_periods=26)
    assert alloc["optimal_spend"].sum() == pytest.approx(alloc["current_spend"].sum(), rel=1e-6)
    assert (alloc["optimal_spend"] >= alloc["lower_bound"] - 1e-6).all()
    assert (alloc["optimal_spend"] <= alloc["upper_bound"] + 1e-6).all()
    assert s["expected_uplift"] >= -1e-6  # optimum is never worse than current mix


@pytest.mark.slow
def test_infeasible_budget_raises(fitted):
    with pytest.raises(ValueError, match="infeasible"):
        optimize_budget(fitted, budget=1.0)
