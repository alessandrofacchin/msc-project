import tensorflow as tf
from typing import Dict, Any, List
import numpy as np
import tensorflow_probability as tfp

from latentneural.utils import ArgsParser
from .layers import GaussianSampling
from .layers import GeneratorGRU


class LFADS(tf.keras.Model):

  def __init__(self, **kwargs: Dict[str, Any]):
    super(LFADS, self).__init__()

    self.encoded_space: int = ArgsParser.get_or_default(kwargs, 'encoded_space', 64)
    self.factor_space: int = ArgsParser.get_or_default(kwargs, 'factor_space', 3)
    self.neural_space: int = ArgsParser.get_or_default(kwargs, 'neural_space', 50)
    self.max_grad_norm: float = ArgsParser.get_or_default(kwargs, 'max_grad_norm', 200)
    self.prior_variance: float = ArgsParser.get_or_default(kwargs, 'prior_variance', 0.1)

    # METRICS
    self.tracker_loss = tf.keras.metrics.Mean(name="loss")
    self.tracker_loss_loglike = tf.keras.metrics.Mean(name="loss_loglike")
    self.tracker_loss_kldiv = tf.keras.metrics.Mean(name="loss_kldiv")
    self.tracker_loss_reg = tf.keras.metrics.Mean(name="loss_reg")
    self.tracker_loss_w_loglike = tf.keras.metrics.Mean(name="loss_w_loglike")
    self.tracker_loss_w_kldiv = tf.keras.metrics.Mean(name="loss_w_kldiv")
    self.tracker_loss_w_reg = tf.keras.metrics.Mean(name="loss_w_reg")
    self.tracker_grad_global_norm = tf.keras.metrics.Mean(name="grad_global_norm")

    # ENCODER
    encoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'encoder', {})
    forward_layer = tf.keras.layers.GRU(self.encoded_space, time_major=False, name="EncoderRNNForward", return_sequences=True, **encoder_args)
    backward_layer = tf.keras.layers.GRU(self.encoded_space, time_major=False, name="EncoderRNNBackward", return_sequences=True, go_backwards=True, **encoder_args)
    self.encoder = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer, name='EncoderRNN')
    self.flatten_post_encoder = tf.keras.layers.Flatten()
    encoder_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'encoder_dense', {})
    self.encoder_dense = tf.keras.layers.Dense(self.encoded_space, name="EncoderDense", **encoder_dense_args)
    
    # DISTRIBUTION
    dense_mean_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_mean', {})
    self.dense_mean = tf.keras.layers.Dense(self.encoded_space, name="DenseMean", **dense_mean_args)
    dense_logvar_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_logvar', {})
    self.dense_logvar = tf.keras.layers.Dense(self.encoded_space, name="DenseLogVar", **dense_logvar_args)

    # SAMPLING
    self.sampling = GaussianSampling(name="GaussianSampling")
    
    # DECODERS
    self.pre_decoder_activation = tf.keras.layers.Activation('tanh')
    decoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'decoder', {})
    cell = GeneratorGRU(self.encoded_space, clip_value=5.0, **decoder_args)
    self.decoder = tf.keras.layers.RNN(cell, return_sequences=True, time_major=False)
    
    # DIMENSIONALITY REDUCTION
    dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense', {})
    self.dense = tf.keras.layers.Dense(self.factor_space, name="Dense", activation='tanh', **dense_args)
    
    # NEURAL
    neural_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'neural_dense', {})
    self.neural_dense = tf.keras.layers.Dense(self.neural_space, name="NeuralDense", **neural_dense_args)

  @tf.function
  def call(self, inputs, training: bool=True):
   g0, mean, logvar = self.encode(inputs, training=training)
   log_f, z = self.decode(g0, inputs, training=training)
   return log_f, (g0, mean, logvar), z

  @tf.function
  def decode(self, g0, inputs, training: bool=True):
    # Assuming inputs are zero and everything comes from the GRU
    u = tf.stack([tf.zeros_like(inputs)[:,:,-1] for i in range(self.decoder.cell.units)], axis=-1)
    
    g0_activated = self.pre_decoder_activation(g0)
    g = self.decoder(u, initial_state=g0_activated, training=training)

    z = self.dense(g, training=training)

    log_f = self.neural_dense(z, training=training)

    # In order to be able to auto-encode, the dimensions should be the same
    if not self.built:
      assert all([f_i == i_i for f_i, i_i in zip(list(log_f.shape), list(inputs.shape))])

    return log_f, z

  @tf.function
  def encode(self, inputs, training: bool=True):
    encoded = self.encoder(inputs, training=training)
    encoded_flattened = self.flatten_post_encoder(encoded, training=training)
    encoded_reduced = self.encoder_dense(encoded_flattened, training=training)

    mean = self.dense_mean(encoded_reduced, training=training)
    logvar = self.dense_logvar(encoded_reduced, training=training)

    g0 = self.sampling(tf.stack([mean, logvar], axis=-1), training=training)
    return g0, mean, logvar

  def compile(self, optimizer, loss_weights, *args, **kwargs):
    super(LFADS, self).compile()
    self.optimizer = optimizer
    self.loss_weights = loss_weights

  @tf.function
  def train_step(self, data):
    """The logic for one training step.
    This method can be overridden to support custom training logic.
    For concrete examples of how to override this method see
    [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
    This method is called by `Model.make_train_function`.
    This method should contain the mathematical logic for one step of training.
    This typically includes the forward pass, loss calculation, backpropagation,
    and metric updates.
    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_train_function`, which can also be overridden.
    Args:
      data: A nested structure of `Tensor`s.
    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # These are the only transformations `Model.fit` applies to user-input
    # data when a `tf.data.Dataset` is provided.
    x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
    # Run forward pass.
    with tf.GradientTape() as tape:
      log_f, (g0, mean, logvar), z = self(x, training=True)
      
      # LOG-LIKELIHOOD
      reconstruction_loglike = tf.nn.log_poisson_loss(
        targets=tf.cast(x, tf.float32), log_input=log_f, compute_full_loss=True
      )

      # KL DIVERGENCE
      dist_prior = tfp.distributions.Normal(0., tf.sqrt(self.prior_variance), name='PriorNormal')
      dist_posterior = tfp.distributions.Normal(mean, tf.exp(0.5 * logvar), name='PosteriorNormal')
      kl_divergence = tfp.distributions.kl_divergence(
        dist_prior, dist_posterior, allow_nan_stats=True, name=None
      )

      loss_loglike = tf.reduce_sum(reconstruction_loglike)
      loss_kldiv = tf.reduce_sum(kl_divergence)
      loss_reg = tf.reduce_sum(self.losses)
      loss = self.loss_weights[0] * loss_loglike + self.loss_weights[1] * loss_kldiv + self.loss_weights[2] * loss_reg

    gradients = tape.gradient(loss, self.trainable_variables)
    gradients, grad_global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
    # Run backwards pass.
    self.optimizer.apply_gradients(
      (grad, var) 
      for (grad, var) in zip(gradients, self.trainable_variables) 
      if grad is not None
    )

    # Compute our own metrics
    self.tracker_loss.update_state(loss)
    self.tracker_loss_loglike.update_state(loss_loglike)
    self.tracker_loss_kldiv.update_state(loss_kldiv)
    self.tracker_loss_reg.update_state(loss_reg)
    self.tracker_loss_w_loglike.update_state(self.loss_weights[0])
    self.tracker_loss_w_kldiv.update_state(self.loss_weights[1])
    self.tracker_loss_w_reg.update_state(self.loss_weights[2])
    self.tracker_grad_global_norm.update_state(grad_global_norm)

    return {
      'loss': self.tracker_loss.result(),
      'loss_loglike': self.tracker_loss_loglike.result(),
      'loss_kldiv': self.tracker_loss_kldiv.result(),
      'loss_reg': self.tracker_loss_reg.result(),
      'loss_w_loglike': self.tracker_loss_w_loglike.result(),
      'loss_w_kldiv': self.tracker_loss_w_kldiv.result(),
      'loss_w_reg': self.tracker_loss_w_reg.result(),
      'grad_global_norm': self.tracker_grad_global_norm.result()
    }