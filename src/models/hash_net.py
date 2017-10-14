# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K
from keras import models, layers
from keras.engine.topology import Layer
  

# --------------------------------------------------------------------------- #
# Encoder utils.                                                              #
# --------------------------------------------------------------------------- #

class BinaryEncoderLayer(Layer):

  def __init__(self, **kwargs):
    super(BinaryEncoderLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask=None):
    return K.batch_flatten(K.cast(K.greater(x, 0), "float32"))

  def get_output_shape_for(self, input_shape):
    return input_shape[:-1]


# --------------------------------------------------------------------------- #
# Encoder networks, to be shared across each element of a triplet.            #
# --------------------------------------------------------------------------- #

def shareable_dilated_conv_layers(n_channels, dilation_rate):
  extent = 1 + (2 * dilation_rate)
  shared_cnv = layers.AtrousConv1D(
      n_channels, extent, atrous_rate=dilation_rate, border_mode="same")
  shared_bnm = layers.BatchNormalization(mode=2)
  return shared_cnv, shared_bnm


def dilated_encoder(n_chars, width):
  shared_emb = layers.Embedding(n_chars, 24, input_length=width)
  shared_grams = [
      layers.Conv1D(64, 3, border_mode="same"),
      layers.Conv1D(32, 3, border_mode="same")]
  shared_bn1 = layers.BatchNormalization(mode=2)
  conv_block_layers = [
      shareable_dilated_conv_layers(96, rate)
      for rate in (2, 4, 8)]
  shared_dense = layers.Dense(32, activation="relu")

  def instantiate_shared_encoder(x):
    emb = shared_emb(x)
    next_input = layers.LeakyReLU(0.2)(shared_bn1(layers.merge(
        [gr(emb) for gr in shared_grams], mode="concat")))
    skip_inputs = []
    for i, (sh_cnv, sh_bnm) in enumerate(conv_block_layers):
      act = layers.LeakyReLU(0.2)(sh_bnm(sh_cnv(next_input)))
      if i == len(conv_block_layers) - 1:
        next_input = layers.merge(skip_inputs + [act], mode="sum")
      else:
        skip_inputs.append(act)
        next_input = layers.merge([next_input, act], mode="sum")
    chmax = layers.GlobalMaxPooling1D()(next_input)
    dense = shared_dense(chmax)
    return layers.Reshape((32, 1))(dense)

  return instantiate_shared_encoder


def dense_encoder(n_chars, width):
  shared_emb = layers.Embedding(n_chars, 32, input_length=width)
  shared_grams = [
      layers.Conv1D(64, 3, border_mode="same"),
      layers.Conv1D(64, 5, border_mode="same")]
  shared_dense1 = layers.Dense(128)
  shared_dense2 = layers.Dense(32, activation="relu")

  def instantiate_shared_encoder(x):
    emb = shared_emb(x)
    grams = layers.LeakyReLU(0.2)(layers.merge(
        [gr(emb) for gr in shared_grams], mode="concat"))
    chmax = layers.GlobalMaxPooling1D()(grams)
    dense1 = layers.LeakyReLU(0.2)(shared_dense1(chmax))
    dense2 = shared_dense2(dense1)
    return layers.Reshape((32, 1))(dense2)

  return instantiate_shared_encoder


def pooling_encoder(n_chars, width):
  shared_emb = layers.Embedding(n_chars, 32, input_length=width)
  shared_grams = [
      layers.Conv1D(48, 2, border_mode="same", activation="relu"),
      layers.Conv1D(64, 3, border_mode="same", activation="relu"),
      layers.Conv1D(96, 5, border_mode="same", activation="relu"),
      layers.Conv1D(128, 7, border_mode="same", activation="relu")]
  shared_conv = [
      layers.Conv1D(128, 1, border_mode="valid", activation="relu"),
      layers.Conv1D(128, 3, border_mode="valid",  activation="relu")]
  shared_dense1 = layers.Dense(64, activation="relu")
  shared_dense2 = layers.Dense(32, activation="relu")

  def instantiate_shared_encoder(x):
    emb = shared_emb(x)
    grams = layers.merge(
        [gr(emb) for gr in shared_grams], mode="concat")
    next_input = layers.MaxPooling1D(pool_length=5, stride=5)(grams)
    for layer in shared_conv:
      next_input = layer(next_input)
    chmax = layers.GlobalMaxPooling1D()(next_input)
    dense1 = shared_dense1(chmax)
    dense2 = shared_dense2(dense1)
    return layers.Reshape((32, 1))(dense2)

  return instantiate_shared_encoder


# --------------------------------------------------------------------------- #
# Loss function. For each triplet of encodings x, y, and z, we want x         #
# and y to be closer in encoding space than x and z.                          #
# --------------------------------------------------------------------------- #

class TripletHingeLossLayer(Layer):

  def __init__(
      self, dist_fn,
      hinge_bias=0.5, reg_fn=None, reg_weight=1.0, **kwargs):
    self.dist_fn = dist_fn
    self.reg_fn = reg_fn
    self.reg_weight = reg_weight
    self.hinge_bias = hinge_bias
    super(TripletHingeLossLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    if len(input_shape) != 3:
      raise ValueError("Exepcted input shape of length 3; saw {}".format(
          input_shape))
    if input_shape[2] != 3:
      raise ValueError(
          "Expected third index of input_shape to be 3; saw {}".format(
              input_shape))

  def call(self, merged, mask=None):
    x, y, z = [merged[:, :, i] for i in np.arange(3)]
    h_pos = self.dist_fn(x, y)
    h_neg = self.dist_fn(x, z)
    err = K.maximum(h_pos - h_neg + self.hinge_bias, 0)
    if self.reg_fn is not None:
      err += self.reg_fn(x) * self.reg_weight
    return err

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], 1)


# --------------------------------------------------------------------------- #
# Differentiable Hamming-Like Distance Functions. Good candidates are upper   #
# or lower bounds to Hamming distance that converge to Hamming distance under #
# some useful set of conditions, eg when distance is 0. Must normalize such   #
# that maximum distance is 1 regardless of hash code length.                  #
# --------------------------------------------------------------------------- #

def hamming(x, y):
  # Not actually useable, because it's not differentiable! Provided for
  # illustrative purposes only. If you use it, the model won't learn. :(
  d = K.sum(K.cast(K.not_equal(K.greater(x, 0), K.greater(y, 0)), "float32"))
  return d / int(x.get_shape()[0])


def positive_hyperbolic_hammingoid(x, y):
  mask = K.cast(K.not_equal(K.greater(x, 0), K.greater(y, 0)), "float32")
  # Masked elementwise maximum
  err = K.maximum(mask * x, mask * y)
  # Make it smooth and asymptotically approach 1
  err = err / (err + 1)
  # Sum across rows, normalize by length
  length = int(x.get_shape()[1])
  return K.sum(err, axis=-1) / length


def tanh_hammingoid(x, y):
  return K.mean(K.abs(K.tanh(x) - K.tanh(y)))


def masked_tanh_hammingoid(x, y):
  mask = K.cast(K.not_equal(K.greater(x, 0), K.greater(y, 0)), "float32")
  return K.mean(mask * K.abs(K.tanh(x) - K.tanh(y)))


# --------------------------------------------------------------------------- #
# Build it.                                                                   #
# --------------------------------------------------------------------------- #

def build_triplet_encoder(
    n_chars, width, encoder_fn,
    dist_fn=positive_hyperbolic_hammingoid, reg_fn=None,
    reg_weight=1.0, hinge_bias=0.5):
  input_xyz = [layers.Input(shape=(width,)) for _ in range(3)]
  shared_encoder = encoder_fn(n_chars, width)
  encoded_xyz = [shared_encoder(inp) for inp in input_xyz]

  # Triplet training
  merged = layers.merge(encoded_xyz, mode="concat", concat_axis=-1)
  loss_layer = TripletHingeLossLayer(
      dist_fn,
      hinge_bias=hinge_bias, reg_fn=reg_fn, reg_weight=reg_weight)
  loss = loss_layer(merged)
  train_model = models.Model(input=input_xyz, output=loss)

  # Binary encoding
  binary_encoder = BinaryEncoderLayer()
  binary_encoded = binary_encoder(encoded_xyz[0])
  encoder_model = models.Model(input=input_xyz[0], output=binary_encoded)

  return train_model, encoder_model
