import numpy as np
import pytensor.tensor as pt
import pytest

from bayesian_mmm import transforms as tf


def test_lagged_media_shifts_correctly():
    x = np.arange(1, 7, dtype=float).reshape(3, 2)  # T=3, C=2
    lag = tf.build_lagged_media(x, 3)
    assert lag.shape == (3, 3, 2)
    np.testing.assert_array_equal(lag[0], x)
    np.testing.assert_array_equal(lag[1], [[0, 0], [1, 2], [3, 4]])
    np.testing.assert_array_equal(lag[2], [[0, 0], [0, 0], [1, 2]])


def test_lagged_media_rejects_bad_input():
    with pytest.raises(ValueError):
        tf.build_lagged_media(np.ones(5), 2)
    with pytest.raises(ValueError):
        tf.build_lagged_media(np.ones((5, 2)), 0)


@pytest.mark.parametrize(
    "fn,args",
    [
        (tf.geometric_weights, (np.array([0.3, 0.8]),)),
        (tf.delayed_weights, (np.array([0.3, 0.8]), np.array([1.0, 2.0]))),
        (tf.weibull_pdf_weights, (np.array([0.8, 2.0]), np.array([0.3, 0.5]))),
    ],
)
def test_normalised_weights_sum_to_one(fn, args):
    w = fn(*args, l_max=6)
    assert w.shape == (6, 2)
    np.testing.assert_allclose(w.sum(0), 1.0)
    assert (w >= 0).all()


def test_geometric_weights_values():
    w = tf.geometric_weights(np.array([0.5]), 3, normalize=False)
    np.testing.assert_allclose(w[:, 0], [1, 0.5, 0.25])


def test_delayed_peak_location():
    w = tf.delayed_weights(np.array([0.5]), np.array([2.0]), 6)
    assert np.argmax(w[:, 0]) == 2


def test_weights_with_draw_dimension():
    alpha = np.array([[0.2, 0.5], [0.4, 0.9], [0.1, 0.1]])  # D=3, C=2
    w = tf.geometric_weights(alpha, 4)
    assert w.shape == (3, 4, 2)
    np.testing.assert_allclose(w.sum(1), 1.0)


def test_adstock_preserves_total_with_normalised_weights():
    rng = np.random.default_rng(0)
    x = np.zeros((40, 1))
    x[5] = 100.0  # single pulse well inside the window
    w = tf.geometric_weights(np.array([0.6]), 8)
    out = tf.apply_adstock(tf.build_lagged_media(x, 8), w)
    np.testing.assert_allclose(out.sum(), 100.0)
    assert out[5, 0] > out[6, 0] > out[7, 0]  # decays after the pulse
    assert out[:5].sum() == 0  # no effect before the spend


def test_adstock_numpy_matches_pytensor():
    rng = np.random.default_rng(1)
    x = rng.gamma(2, 1, size=(30, 3))
    alpha = np.array([0.2, 0.5, 0.8])
    lag = tf.build_lagged_media(x, 5)
    np_out = tf.apply_adstock(lag, tf.geometric_weights(alpha, 5))
    pt_out = tf.apply_adstock(pt.as_tensor(lag), tf.geometric_weights(pt.as_tensor(alpha), 5, xp=pt), xp=pt).eval()
    np.testing.assert_allclose(np_out, pt_out, rtol=1e-10)


def test_hill_properties():
    x = np.linspace(0, 10, 101)
    y = tf.hill(x, 2.0, 1.5)
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(y) > 0)  # monotone
    assert y.max() < 1
    assert tf.hill(np.array(2.0), 2.0, 1.5) == pytest.approx(0.5)  # half saturation


def test_hill_gradient_is_finite_at_zero():
    s = pt.dscalar("s")
    out = tf.hill(pt.as_tensor(np.array([0.0, 0.5])), 0.5, s, xp=pt).sum()
    g = pt.grad(out, s).eval({s: 1.5})
    assert np.isfinite(g)


def test_logistic_and_mm():
    x = np.linspace(0, 5, 50)
    assert np.all(np.diff(tf.logistic(x, 2.0)) > 0)
    assert tf.michaelis_menten(np.array(1.0), 1.0) == pytest.approx(0.5)


def test_dispatchers_raise_on_unknown():
    with pytest.raises(ValueError):
        tf.adstock_weights("nope", {}, 3)
    with pytest.raises(ValueError):
        tf.saturate("nope", np.ones(2), {})
