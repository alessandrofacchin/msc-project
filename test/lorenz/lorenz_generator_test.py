from lorenz import LorenzGenerator
import numpy as np


def test_lorenz_generator():
    g = LorenzGenerator()
    df = g.generate_data()

    np.testing.assert_almost_equal(df.index[-1], 99.99)
    np.testing.assert_equal(list(df.columns), ['x', 'y', 'z'])