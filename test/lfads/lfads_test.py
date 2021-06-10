import tensorflow as tf
import numpy as np
import pytest

from latentneural import LFADS


@pytest.mark.unit
def test_dimensionality():
    input_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    model = LFADS(neural_space=50)
    model.build(input_shape=[None] + list(input_data.shape[1:]))
    
    log_f, (g0_r, r_mean, r_logvar) = model.call(input_data, training=True)

    tf.debugging.assert_equal(log_f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))

    log_f, (g0_r, r_mean, r_logvar) = model.call(input_data, training=False)

    tf.debugging.assert_equal(log_f.shape, tf.TensorShape([10, 100, 50]))
    tf.debugging.assert_equal(r_mean.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(r_logvar.shape, tf.TensorShape([10, 64]))
    tf.debugging.assert_equal(g0_r.shape, tf.TensorShape([10, 64]))
