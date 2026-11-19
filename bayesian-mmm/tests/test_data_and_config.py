import numpy as np
import pandas as pd
import pytest

from bayesian_mmm.config import load_config
from bayesian_mmm.data import load_lift_tests, validate_dataframe


def test_defaults_fill_missing_keys(cfg):
    assert cfg["model"]["adstock"]["type"] == "geometric"
    assert cfg["sampler"]["draws"] == 100  # override applied


@pytest.mark.parametrize(
    "override",
    [
        {"data": {"channel_cols": []}},
        {"data": {"channel_cols": ["tv"]}, "model": {"adstock": {"type": "bogus"}}},
        {"data": {"channel_cols": ["tv"]}, "model": {"saturation": {"type": "bogus"}}},
        {"data": {"channel_cols": ["tv"], "control_cols": ["tv"]}},
        {"data": {"channel_cols": ["tv"]}, "model": {"likelihood": "poisson"}},
    ],
)
def test_invalid_config_raises(override):
    with pytest.raises(ValueError):
        load_config(None, override)


def test_validation_catches_bad_data(synthetic, cfg):
    df, _, _ = synthetic
    bad = df.copy()
    bad.loc[3, "tv"] = -1
    with pytest.raises(ValueError, match="Negative"):
        validate_dataframe(bad, cfg)
    bad = df.copy()
    bad.loc[3, "search"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        validate_dataframe(bad, cfg)
    with pytest.raises(ValueError, match="missing"):
        validate_dataframe(df.drop(columns=["tv"]), cfg)


def test_scalers_fit_on_train_only(mmm_data):
    d = mmm_data
    assert d.n_train == d.n_periods - 8
    assert d.y_scale == pytest.approx(d.y[: d.n_train].max())
    np.testing.assert_allclose(d.media_scale, d.media[: d.n_train].max(0))
    assert d.scale_media(d.media)[: d.n_train].max() == pytest.approx(1.0)


def test_binary_controls_not_standardised(mmm_data):
    z = mmm_data.scaled_controls()
    idx = mmm_data.control_names.index("holiday")
    assert set(np.unique(z[:, idx])) <= {0.0, 1.0}
    idx = mmm_data.control_names.index("price_index")
    assert abs(z[: mmm_data.n_train, idx].mean()) < 1e-9


def test_time_features(mmm_data):
    t = mmm_data.time_index()
    assert t[0] == 0 and t[mmm_data.n_train - 1] == pytest.approx(1.0) and t[-1] > 1
    f = mmm_data.fourier_features()
    assert f.shape == (mmm_data.n_periods, 2 * mmm_data.fourier_order)
    assert np.all(np.abs(f) <= 1)


def test_lift_test_loader(synthetic, mmm_data, tmp_path):
    _, lift, d = synthetic
    lt = load_lift_tests(str(d / "lift.csv"), mmm_data.channels)
    assert len(lt) == len(lift)
    bad = lift.copy()
    bad.loc[0, "channel"] = "unknown"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        load_lift_tests(str(tmp_path / "bad.csv"), mmm_data.channels)
    assert load_lift_tests(None, mmm_data.channels) is None
