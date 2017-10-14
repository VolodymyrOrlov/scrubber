# -*- coding: utf-8 -*-
import os
import re
import sys

import gflags
import numpy as np
import pandas as pd

from keras import models, layers
from sklearn.externals import joblib

from neural_merchant_scrubbing.src.features import build_features


FLAGS = gflags.FLAGS
gflags.DEFINE_integer(
    "embed_size", 8,
    "Size of the model's initial character embedding space.")
gflags.DEFINE_integer(
    "base_channels", 32,
    "Number of channels in short-range convolutions. Higher level atrous "
    "convolutions will have 3x this many channels.")
gflags.DEFINE_integer(
    "batch_size", 32,
    "Number of training examples per batched model update.")
gflags.DEFINE_integer(
    "epochs", 1,
    "Number of cycles through an arbitrarily sized epoch.")
gflags.DEFINE_integer(
    "batches_per_epoch", 1000,
    "Number of training batches per epoch.")
gflags.DEFINE_integer(
    "validation_batches", None,
    "Number of batches used in validation at end of each epoch; defaults to "
    "number of training batches over 20.")
gflags.DEFINE_integer(
    "string_width", 125,
    "Length of string representations; longer strings will be truncated.")
gflags.DEFINE_string(
    "model_load_path", None,
    "Path from which to load pre-trained model for further training.")
gflags.DEFINE_boolean(
    "permute_merchants", True,
    "If True, randomly permute target merchant strings.")
gflags.DEFINE_boolean(
    "permute_schemas", True,
    "If True, randomly permute context schema strings.")
gflags.DEFINE_list(
    "dilation_rates", [2, 4, 8, 2, 4, 8],
    "Dilation rates of dilated convolutional filters.")
gflags.DEFINE_float(
    "mutation_rate", 0.1,
    "Rate at which each merchant and schema permutation is applied.")
gflags.DEFINE_list(
    "grams", [2, 3, 5, 7],
    "Extract ngrams of these lengths. The first two in the list will have "
    "channels equal to --base_channels; the rest half as many.")


# --------------------------------------------------------------------------- #
# Utils.                                                                      #
# --------------------------------------------------------------------------- #

def get_data(project_dir):
  schemas_path = os.path.join(project_dir, "data", "raw", "schemas.dat")
  schemas = joblib.load(schemas_path)
  n_blanks_to_add = schemas.shape[0] / 4
  schemas = np.concatenate([schemas, np.repeat("_____", n_blanks_to_add)])
  ixs = np.arange(schemas.shape[0])
  np.random.shuffle(ixs)
  n_test = int(schemas.shape[0] / 10.0)
  train_schemas = schemas[ixs[n_test:]]
  test_schemas = schemas[ixs[:n_test]]

  merchants_path = os.path.join(project_dir, "references", "merchants.txt")
  with open(merchants_path, "r") as f:
    merchants = np.array(list(set(
        [line.strip() for line in f.readlines() if line])))

  return train_schemas, test_schemas, merchants


def now():
  dt_str = str(np.datetime64("now"))
  return dt_str.replace("-", "").replace("T", "").replace(":", "")


# --------------------------------------------------------------------------- #
# Data generator.                                                             #
# --------------------------------------------------------------------------- #

PUNCT_RE = re.compile(r"[:/.@*`'-]")


def permute_merchant(m, mut_rate=0.1):
    if len(m.split()) > 2 and np.random.random() < mut_rate:
      m = " ".join(m.split()[:-1])
    # strip up to 5 characters off the end
    if np.random.random() < mut_rate:
      m = m[:-min(len(m) - 2, np.random.randint(1, 6))].strip()
    # replace all punctuation
    if np.random.random() < mut_rate:
      replacer = np.random.choice(["'", ".", " ", "", "*", "-"])
      m = PUNCT_RE.sub(replacer, m)
    # delete all spaces
    if np.random.random() < mut_rate:
      m = m.replace(" ", "")
    return m


def permute_schema(s, mut_rate=0.1):
  before, blank, after = s.partition("_____")
  # strip up to 5 characters from the end
  if np.random.random() < mut_rate:
    after = after[:-np.random.randint(1, 6)].strip()
  # strip the first word for a multi-word start
  if np.random.random() < mut_rate and len(before.split()) >= 2:
    before = " ".join(before.split()[1:]) + " "
  # strip the last word for a multi-word end
  if np.random.random() < mut_rate and len(after.split()) >= 2:
    after = " " + " ".join(after.split()[:1])
  # remove all spaces
  if np.random.random() < mut_rate:
    before = before.replace(" ", "")
    after = after.replace(" ", "")
  # remove spaces between the start and the merchant
  if np.random.random() < mut_rate:
    before = before.strip()
  # remove spaces between the end and the merchant
  if np.random.random() < mut_rate:
    after = after.strip()
  return "".join([before, blank, after])


def one_hot(ints, width):
  oh = np.zeros((ints.shape[0], width))
  hot_ixs = np.minimum(ints, width - 1)
  oh[np.arange(ints.shape[0]), hot_ixs] = 1
  return oh


def batch_generator(
    batch_size, schemas, merchants, cv, width,
    permute_merchants=True, permute_schemas=True, mutation_rate=0.1,
    vectorize=True):
  schm_cursor = 0
  schm_size = schemas.shape[0]
  schm_ixs = np.arange(schm_size)
  mrch_cursor = 0
  mrch_size = merchants.shape[0]
  mrch_ixs = np.arange(mrch_size)
  while True:

    if schm_cursor + batch_size >= schm_size:
      schm_cursor = 0
    if schm_cursor == 0:
      np.random.shuffle(schm_ixs)

    if mrch_cursor + batch_size >= mrch_size:
      mrch_cursor = 0
    if mrch_cursor == 0:
      np.random.shuffle(mrch_ixs)

    batch_merchants = [
        permute_merchant(m, mut_rate=mutation_rate) for m
        in merchants[mrch_ixs[mrch_cursor:mrch_cursor + batch_size]]]
    batch_schemas = [
        permute_schema(s, mut_rate=mutation_rate) for s
        in schemas[schm_ixs[schm_cursor:schm_cursor + batch_size]]]
    complete = [
        s.replace("_____", m)
        for m, s in zip(batch_merchants, batch_schemas)]
    merchant_starts = np.array([s.index("_____") for s in batch_schemas])
    merchant_ends = np.array(
        [st + len(m) for m, st in zip(batch_merchants, merchant_starts)])

    schm_cursor += batch_size
    mrch_cursor += batch_size
    if vectorize:
      yield (
          cv.transform(complete).toarray(),
          [one_hot(merchant_starts, width),
           one_hot(merchant_ends, width)])
    else:
      yield [complete, (merchant_starts, merchant_ends)]


# --------------------------------------------------------------------------- #
# Model.                                                                      #
# --------------------------------------------------------------------------- #

def dilated_conv_block(x, dilation_rate, skip_inputs=None):
  extent = 1 + (2 * dilation_rate)
  n_channels = int(x.get_shape()[-1])
  cnv = layers.AtrousConv1D(
      n_channels, extent,
      atrous_rate=dilation_rate,
      border_mode="same")(x)
  bnm = layers.BatchNormalization()(cnv)
  act = layers.LeakyReLU(0.2)(bnm)
  if skip_inputs is None:
    return act
  else:
    return act, layers.merge(skip_inputs + [act], mode="sum")


def dilated_conv_tower(x, dilation_rates):
  next_input = x
  final_layer_skip_inputs = []
  last_block = len(dilation_rates) - 1
  for i, rate in enumerate(dilation_rates):
    skip_inputs = final_layer_skip_inputs if i == last_block else [next_input]
    activation, merged = dilated_conv_block(
        next_input, rate, skip_inputs=skip_inputs)
    next_input = merged
    final_layer_skip_inputs.append(activation)
  return merged


def ngrams(x, grams, n_ch):
  l = len(grams)
  channels_list = (([n_ch] * 2) + ([n_ch / 2] * (l - 2)))[:l]
  gram_layers = [
      layers.Conv1D(ch, n, border_mode="same")(x)
      for ch, n in zip(channels_list, grams)]
  return layers.merge(gram_layers, mode="concat")


def make_model(
    width, n_chars, embed_size=8, n_ch=64,
    grams=[2, 3, 5, 7], dilation_rates=[2, 4, 8, 2, 4, 8]):
  # Embed
  inp = layers.Input((width,))
  emb = layers.Embedding(n_chars, embed_size, input_length=width)(inp)

  # Extract 2-gram, 3-gram, 5-gram, and 7-gram features
  grm = ngrams(emb, grams, n_ch)
  bn1 = layers.BatchNormalization()(grm)
  rl1 = layers.LeakyReLU(0.2)(bn1)

  # Atrous Convolutions
  dilated_output = dilated_conv_tower(rl1, dilation_rates)

  # Output
  out_start = layers.Conv1D(1, 1, border_mode="same")(dilated_output)
  flt_start = layers.Flatten()(out_start)
  sft_start = layers.Activation("softmax", name="start_index")(flt_start)
  out_end = layers.Conv1D(1, 1, border_mode="same")(dilated_output)
  flt_end = layers.Flatten()(out_end)
  sft_end = layers.Activation("softmax", name="end_index")(flt_end)
  model = models.Model(input=inp, output=[sft_start, sft_end])

  return model


# --------------------------------------------------------------------------- #
# Driver.                                                                     #
# --------------------------------------------------------------------------- #

def main(project_dir):
  # Load data
  print "Loading in data..."
  train_schemas, test_schemas, merchants = get_data(project_dir)
  cv = build_features.CharacterVectorizer(width=FLAGS.string_width)
  train_generator = batch_generator(
      FLAGS.batch_size, train_schemas, merchants, cv, FLAGS.string_width,
      permute_merchants=FLAGS.permute_merchants,
      permute_schemas=FLAGS.permute_schemas,
      mutation_rate=FLAGS.mutation_rate)
  test_generator = batch_generator(
      FLAGS.batch_size, test_schemas, merchants, cv, FLAGS.string_width,
      permute_merchants=FLAGS.permute_merchants,
      permute_schemas=FLAGS.permute_schemas,
      mutation_rate=FLAGS.mutation_rate)

  # Compile a model
  if FLAGS.model_load_path is not None:
    print "Loading model from {}...".format(FLAGS.model_load_path)
    model = models.load_model(FLAGS.model_load_path)
  else:
    print "Building model..."
    dilation_rates = [int(x) for x in FLAGS.dilation_rates]
    grams = [int(x) for x in FLAGS.grams]
    model = make_model(
        FLAGS.string_width, cv.MAX_VALUE + 1,
        embed_size=FLAGS.embed_size,
        n_ch=FLAGS.base_channels,
        grams=grams,
        dilation_rates=dilation_rates)
    print "Compiling model..."
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

  # Training
  print "Starting training..."
  try:
    val_batches = FLAGS.validation_batches or (FLAGS.batches_per_epoch / 20)
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=FLAGS.batch_size * FLAGS.batches_per_epoch,
        nb_epoch=FLAGS.epochs,
        verbose=1,
        validation_data=test_generator,
        nb_val_samples=FLAGS.batch_size * val_batches)
    interrupted = False
  except KeyboardInterrupt:
    print "Training interrupted!"
    history = None
    interrupted = True

  # Save
  hdn_fmt = (
      "em{em}_ch{ch}_ba{ba}_ep{ep}_bpe{bpe}_sw{sw}_gr{gr}"
      "{pm}{ps}{mr}_{now}{ki}")
  uses_permutaitons = FLAGS.permute_merchants or FLAGS.permute_schemas
  handle = hdn_fmt.format(
      em=FLAGS.embed_size, ch=FLAGS.base_channels, ba=FLAGS.batch_size,
      ep=FLAGS.epochs, bpe=FLAGS.batches_per_epoch, sw=FLAGS.string_width,
      gr="".join(FLAGS.grams),
      pm="_pm" if FLAGS.permute_merchants else "",
      ps="_ps" if FLAGS.permute_schemas else "",
      mr="_mr{}".format(FLAGS.mutation_rate) * uses_permutaitons,
      now=now(), ki="_interrupted" if interrupted else "")
  if FLAGS.model_load_path is not None:
    orig_handle = os.path.splitext(os.path.basename(FLAGS.model_load_path))[0]
    handle = orig_handle + "_update_" + handle
  outpath = os.path.join(project_dir, "models", handle + ".model")
  print "Saving model to {}...".format(outpath)
  model.save(outpath)
  if history is not None:
    histpath = os.path.join(
        project_dir, "reports", "history_" + handle + ".csv")
    print "Saving history to {}...".format(histpath)
    pd.DataFrame(history.history).to_csv(histpath, index=False)


if __name__ == '__main__':
  FLAGS(sys.argv)
  project_dir = os.path.realpath(
      os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
  main(project_dir)
