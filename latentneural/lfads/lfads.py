import tensorflow as tf
from typing import Dict, Any

from latentneural.utils import ArgsParser
from .sampling import Sampling


class LFADS(tf.keras.Model):

  def __init__(self, **kwargs: Dict[str, Any]):
    super(LFADS, self).__init__()

    dynamics: int = ArgsParser.get_or_default(kwargs, 'dynamics', 64)
    factors: int = ArgsParser.get_or_default(kwargs, 'factors', 2)
    neural_space: int = ArgsParser.get_or_default(kwargs, 'neural_space', 50)

    # ENCODER
    encoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'encoder', {})
    self.encoder = tf.keras.layers.GRU(dynamics, time_major=False, name="EncoderRNN", **encoder_args)
    dense_mean_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_mean', {})
    self.dense_mean = tf.keras.layers.Dense(dynamics, name="DenseMean", **dense_mean_args)
    dense_logvar_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_logvar', {})
    self.dense_logvar = tf.keras.layers.Dense(dynamics, name="DenseLogVar", **dense_logvar_args)

    # SAMPLING
    self.sampling = Sampling(name="Sampling")
    
    # DECODERS
    decoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'decoder', {})
    self.decoder = tf.keras.layers.GRU(dynamics, return_sequences=True, time_major=False, name="DecoderRNN", **decoder_args)
    
    # DIMENSIONALITY REDUCTION
    dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense', {})
    self.dense = tf.keras.layers.Dense(factors, name="Dense", **dense_args)
    
    # NEURAL
    neural_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'neural_dense', {})
    self.neural_dense = tf.keras.layers.Dense(neural_space, name="NeuralDense", **neural_dense_args)
    

  def call(self, inputs, training: bool=True):
    encoded = self.encoder(inputs, training=training)

    mean = self.dense_mean(encoded)
    logvar = self.dense_logvar(encoded)

    g0 = self.sampling(tf.stack([mean, logvar], axis=-1), training=training)
    
    # Assuming inputs are zero and everything comes from the GRU
    u = tf.stack([tf.zeros_like(inputs)[:,:,-1] for i in range(self.decoder.units)], axis=-1)
    
    g = self.decoder(u, initial_state=g0, training=training)

    z = self.dense(g, training=training)

    f = self.neural_dense(z)

    # In order to be able to auto-encode, the dimensions should be the same
    if not self.built:
      assert all([f_i == i_i for f_i, i_i in zip(list(f.shape), list(inputs.shape))])

    return f, (g0, mean, logvar)