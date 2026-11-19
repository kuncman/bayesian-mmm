import numpy as np
import pytest

from bayesian_mmm.config import load_config
from bayesian_mmm.data import load_lift_tests, prepare_data, validate_dataframe
from bayesian_mmm.model import BayesianMMM
from bayesian_mmm.simulate import simulate

CHANNELS = ["tv", "search", "social", "display", "radio"]
CONTROLS = ["price_index", "competitor_promo", "holiday"]


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    df, contrib, lift, truth = simulate(n_weeks=104, seed=3, holdout=8)
    d = tmp_path_factory.mktemp("data")
    df.to_csv(d / "data.csv", index=False)
    lift.to_csv(d / "lift.csv", index=False)
    return df, lift, d


@pytest.fixture(scope="session")
def cfg(synthetic):
    _, _, d = synthetic
    return load_config(
        None,
        {
            "data": {
                "path": str(d / "data.csv"),
                "channel_cols": CHANNELS,
                "control_cols": CONTROLS,
                "holdout_periods": 8,
                "lift_tests_path": str(d / "lift.csv"),
            },
            "sampler": {"draws": 100, "tune": 150, "chains": 2, "cores": 1, "prior_samples": 50},
        },
    )


@pytest.fixture(scope="session")
def mmm_data(synthetic, cfg):
    df, _, _ = synthetic
    return prepare_data(validate_dataframe(df, cfg), cfg)


@pytest.fixture(scope="session")
def fitted(cfg, mmm_data):
    """A (very short) fitted model shared by the slow tests."""
    lift = load_lift_tests(cfg["data"]["lift_tests_path"], mmm_data.channels)
    m = BayesianMMM(cfg, mmm_data, lift)
    m.build()
    m.sample_prior_predictive()
    m.fit(progressbar=False)
    return m


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run MCMC tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow (runs MCMC)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
