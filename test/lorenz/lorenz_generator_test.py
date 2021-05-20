from lorenz import LorenzGenerator
import numpy as np


def test_lorenz_generator():
    g = LorenzGenerator()
    t, Z = g.generate_latent()

    np.testing.assert_equal(t.shape, (np.ceil(1/0.006),))
    np.testing.assert_equal(Z.shape, (np.ceil(1/0.006), 3))
    np.testing.assert_almost_equal(t[-1], 0.996)