# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.base import TransformerMixin
from sklearn.externals import joblib


# --------------------------------------------------------------------------- #
# Feature extractor class.                                                    #
# --------------------------------------------------------------------------- #

class CharacterVectorizer(TransformerMixin):

  MAX_VALUE = 51
  UNK_CHAR = "_"
  CHAR_DICT = dict(zip([chr(x) for x in range(65, 91)], range(1, 27)))
  CHAR_DICT.update(dict(zip([chr(x) for x in range(48, 58)], range(27, 37))))
  CHAR_DICT.update({
    " ": 37, "-": 38, "'": 39, "#": 40, "/": 41, "=": 42, "?": 43,
    "*": 44, ",": 45, ".": 46, ":": 47, "&": 48, "@": 49, "%": 50,
    UNK_CHAR: MAX_VALUE})
  INVERSE_DICT = dict([(v, k) for k, v in CHAR_DICT.items()])

  def __init__(self, width=125):
    self.width = None
    self.set_params(width=width)

  def _fix_array(self, X):
    if isinstance(X, np.ndarray):
      return X
    else:
      try:
        return X.toarray()
      except AttributeError:
        return np.array(X)

  def _string_to_ints(self, s):
    ords = [self.CHAR_DICT.get(c, 51) for c in s.upper()]
    if len(ords) >= self.width:
      return np.array(ords[:self.width])
    else:
      return np.array(ords + ([0] * (self.width - len(ords))))

  def set_params(self, **params):
    self.width = params.get("width")

  def get_params(self):
    return {"width": self.width}

  def transform(self, X):
    X = self._fix_array(X)
    shape = X.shape
    flat = X.flatten()
    intified = [self._string_to_ints(s) for s in flat]
    return sparse.csr_matrix(np.array(
        intified).reshape(shape + (self.width,)))

  def inverse_transform(self, X):
    X = self._fix_array(X)
    strings = []
    for ints in X:
      strings.append("".join(self.INVERSE_DICT.get(i, "") for i in ints))
    return np.array(strings)


# --------------------------------------------------------------------------- #
# Utils.                                                                      #
# --------------------------------------------------------------------------- #

def start_and_stop(merchant, schema):
  start = schema.index("_____")
  return [start, start + len(merchant)]


def save_verbosely(obj, dir, human_readable, fname):
  outpath = os.path.join(dir, fname + ".dat")
  print "Saving {} to {}...".format(human_readable, outpath)
  joblib.dump(obj, outpath)


# --------------------------------------------------------------------------- #
# Driver.                                                                     #
# --------------------------------------------------------------------------- #

def main(project_dir):
  # Schemas
  schemas_path = os.path.realpath(os.path.join(
      project_dir, "data", "raw", "schemas.dat"))
  print "Reading in schema data from {}...".format(schemas_path)
  schemas = joblib.load(schemas_path)
  print "Schemas shape:", schemas.shape

  # Merchants
  merchants_path = os.path.realpath(os.path.join(
      project_dir, "references", "rare_merchants.txt"))
  print "Reading in merchants from {}...".format(merchants_path)
  with open(merchants_path, "r") as f:
    merchants = np.array([line.strip() for line in f.readlines() if line])
  print "Merchants shape:", merchants.shape

  # Data Augmentation
  print "Augmenting data..."
  apostrophe = np.array([s for s in merchants if s.endswith("'S")])
  apostrophe_with_space = np.array([s.replace("'", " ") for s in apostrophe])
  apostrophe_without = np.array([s.replace("'", "") for s in apostrophe])

  long_merchants = np.array([s for s in merchants if len(s) >= 9])
  drop1 = np.array([s[:-1] for s in long_merchants])
  drop2 = np.array([s[:-2] for s in long_merchants])
  drop3 = np.array([s[:-3] for s in long_merchants])
  [np.random.shuffle(ar) for ar in (drop1, drop2, drop3)]
  n = long_merchants.shape[0]
  drop1 = drop1[:n / 2]
  drop2 = drop1[:n / 3]
  drop3 = drop1[:n / 4]

  all_merchant_strings = np.concatenate([
      merchants,
      apostrophe_with_space, apostrophe_without,
      drop1, drop2, drop3])
  np.random.shuffle(all_merchant_strings)

  # Combine merchants and schemas
  repeats = (1 + (schemas.shape[0] / all_merchant_strings.shape[0]))
  repeated_merchants = np.concatenate([all_merchant_strings] * repeats)
  np.random.shuffle(repeated_merchants)
  repeated_merchants = repeated_merchants[:schemas.shape[0]]
  paired = zip(repeated_merchants, schemas)
  synthetic_merchant_strings = np.array(
      [s.replace("_____", m) for m, s in paired] +
      all_merchant_strings.tolist())
  starts_and_stops = np.array(
      [start_and_stop(m, s) for m, s in paired] +
      [(0, len(m)) for m in all_merchant_strings])
  repeated_merchants = np.concatenate(
      [repeated_merchants, all_merchant_strings])

  # Vectorize
  print "Vectorizing..."
  input_cv = CharacterVectorizer(width=125)
  vectorized_inputs = input_cv.transform(synthetic_merchant_strings)
  target_cv = CharacterVectorizer(width=40)
  vectorized_targets = target_cv.transform(repeated_merchants)

  # Cleanup
  interim_dir = os.path.realpath(os.path.join(project_dir, "data", "interim"))
  interim_files = os.listdir(interim_dir)
  print "Deleting old data..."
  for handle in interim_files:
    if not handle.startswith("."):
      os.remove(os.path.join(interim_dir, handle))
      print "-- REMOVED:", handle

  # Save
  save_verbosely(
      synthetic_merchant_strings, interim_dir, "string inputs", "raw_input")
  save_verbosely(
      vectorized_inputs, interim_dir, "vectorized inputs", "inputs")
  save_verbosely(
      repeated_merchants, interim_dir, "string targets", "raw_targets")
  save_verbosely(
      vectorized_targets, interim_dir, "vectorized targets", "targets")
  save_verbosely(
      starts_and_stops, interim_dir, "target indices", "target_indices")

  print "\nDone!\n"


if __name__ == '__main__':
  project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
  main(project_dir)
