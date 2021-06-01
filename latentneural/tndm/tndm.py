import tensorflow as tf
from typing import Dict, Any

from latentneural.utils import ArgsParser
from .sampling_layer import SamplingLayer
from .crop_layer import CropLayer




class TNDM(tf.keras.Model):

  def __init__(self, kwargs: Dict[str, Any]):
    super(TNDM, self).__init__()

    relevant_dynamics: int = ArgsParser.get_or_default(kwargs, 'relevant_dynamics', 64)
    irrelevant_dynamics: int = ArgsParser.get_or_default(kwargs, 'irrelevant_dynamics', 64)
    relevant_factors: int = ArgsParser.get_or_default(kwargs, 'relevant_factors', 2)
    irrelevant_factors: int = ArgsParser.get_or_default(kwargs, 'irrelevant_factors', 1)
    behavioural_space: int = ArgsParser.get_or_default(kwargs, 'behaviour_space', 1)
    neural_space: int = ArgsParser.get_or_default(kwargs, 'neural_space', 1)

    # ENCODER
    encoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'encoder', {})
    self.encoder = tf.keras.layers.GRU((2 * (relevant_dynamics + irrelevant_dynamics)), **encoder_args)
    self.split_relevant_mean = CropLayer(-1, 0, relevant_dynamics)
    self.split_relevant_logvar = CropLayer(-1, relevant_dynamics, 2 * relevant_dynamics)
    self.split_irrelevant_mean = CropLayer(-1, 2 * relevant_dynamics, 2 * relevant_dynamics + irrelevant_dynamics)
    self.split_irrelevant_logvar = CropLayer(-1, 2 * relevant_dynamics + irrelevant_dynamics, 2 * (relevant_dynamics + irrelevant_dynamics))

    # SAMPLING
    self.relevant_sampling = SamplingLayer()
    self.irrelevant_sampling = SamplingLayer()
    
    # DECODERS
    relevant_decoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'relevant_decoder', {})
    self.relevant_decoder = tf.keras.layers.GRU(relevant_dynamics, **relevant_decoder_args)
    irrelevant_decoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'irrelevant_decoder', {})
    self.irrelevant_decoder = tf.keras.layers.GRU(irrelevant_dynamics, **irrelevant_decoder_args)

    # DIMENSIONALITY REDUCTION
    relevant_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'relevant_dense', {})
    self.relevant_dense = tf.keras.layers.Dense(relevant_factors, **relevant_dense_args)
    irrelevant_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'irrelevant_dense', {})
    self.irrelevant_dense = tf.keras.layers.Dense(irrelevant_factors, **irrelevant_dense_args)

    # BEHAVIOUR
    behavioural_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'behavioural_dense', {})
    self.behavioural_dense = tf.keras.layers.Dense(behavioural_space, **behavioural_dense_args)

    # NEURAL
    self.neural_concatenation = tf.keras.layers.Concatenate()
    neural_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'neural_dense', {})
    self.neural_dense = tf.keras.layers.Dense(neural_space, **neural_dense_args)
    

  def call(self, inputs, training: bool=False):
    inputs = tf.keras.Input(inputs.shape)

    encoded = self.encoder(inputs, training=training)

    r_mean = self.split_relevant_mean(encoded)
    r_logvar = self.split_relevant_logvar(encoded)
    i_mean = self.split_irrelevant_mean(encoded)
    i_logvar = self.split_irrelevant_logvar(encoded)

    if training:
      g0_r = self.relevant_sampling([r_mean, r_logvar])
      g0_i = self.irrelevant_sampling([i_mean, i_logvar])
    else:
      g0_r = r_mean
      g0_i = i_mean

    g_r = self.relevant_decoder([], g0_r, training=training)
    g_i = self.relevant_decoder([], g0_i, training=training)

    z_r = self.relevant_dense(g_r, training=training)
    z_i = self.relevant_dense(g_i, training=training)

    b = self.behavioural_dense(z_r)

    z = self.neural_concatenation([z_r, z_i])
    f = self.neural_dense(z)

    return b, f, (r_mean, r_logvar), (i_mean, i_logvar)