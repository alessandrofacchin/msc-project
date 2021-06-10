import tensorflow as tf
from typing import List
import numpy as np

@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(model, neural, coefficients: List[float]=[1,1]):
    f, (g0_r, r_mean, r_logvar) = model.call(neural)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=f, labels=tf.cast(neural, tf.float32))

    reconstruction_error_neural = tf.reduce_mean(cross_ent, axis=[1, 2])

    kl_neural = log_normal_pdf(g0_r, 0., 0.) - log_normal_pdf(g0_r, r_mean, r_logvar)

    tf.print(
        'R_N: ', tf.reduce_mean(reconstruction_error_neural, axis=0),
        ' \tKL_N: ', tf.reduce_mean(kl_neural, axis=0),
    )
    
    return tf.reduce_mean(
        coefficients[0] * reconstruction_error_neural - 
        coefficients[1] * kl_neural)
