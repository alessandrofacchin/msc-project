from lorenz import LorenzGenerator
import numpy as np


def test_lorenz_generate_latent():
    g = LorenzGenerator()
    t, z = g.generate_latent()

    np.testing.assert_equal(t.shape, (np.ceil(1/0.006),))
    np.testing.assert_equal(z.shape, (np.ceil(1/0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)

def test_lorenz_generate_rates():
    g = LorenzGenerator()
    t, f, weights, z = g.generate_rates(n=5, trials=2)

    np.testing.assert_equal(t.shape, (np.ceil(1/0.006),))
    np.testing.assert_equal(f.shape, (2, np.ceil(1/0.006), 5))
    np.testing.assert_equal(weights.shape, (3, 5))
    np.testing.assert_equal(z.shape, (2, np.ceil(1/0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)

def test_lorenz_generate_spikes():
    g = LorenzGenerator()
    t, s, f, weights, z = g.generate_spikes(n=2, trials=5, conditions=10)

    np.testing.assert_equal(t.shape, (np.ceil(1/0.006),))
    np.testing.assert_equal(s.shape, (10, 5, np.ceil(1/0.006), 2))
    np.testing.assert_equal(f.shape, (10, 5, np.ceil(1/0.006), 2))
    np.testing.assert_equal(weights.shape, (10, 3, 2))
    np.testing.assert_equal(z.shape, (10, 5, np.ceil(1/0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)