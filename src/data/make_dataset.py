# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sharknado.scipy.lib.data_manipulation import shark


# --------------------------------------------------------------------------- #
# Queries.                                                                    #
# --------------------------------------------------------------------------- #

CREATE_TABLE = r"""
CREATE TABLE IF NOT EXISTS scrub_schemas AS
SELECT
  REGEXP_REPLACE(
      REGEXP_REPLACE(UPPER(merchant), UPPER(seller), "_____"),
      "\\d+", "0") AS scrubbed_schema,
  REGEXP_REPLACE(
      UPPER(FIRST(merchant)), UPPER(FIRST(seller)),
      "_____") AS schema,
  COUNT(*) AS txns,
  COUNT(DISTINCT uid) AS users,
  COUNT(DISTINCT merchant) AS strings,
  COLLECT_SET(seller) AS sellers,
  COUNT(DISTINCT(seller)) AS num_sellers,
  CAST(RAND() * 10 AS INT) AS shard
FROM transactions
WHERE seller != ""
  AND UPPER(merchant) REGEXP UPPER(seller)
GROUP BY REGEXP_REPLACE(
    REGEXP_REPLACE(UPPER(merchant), UPPER(seller), "_____"),
    "\\d+", "0")
HAVING num_sellers >= 2
ORDER BY users DESC
"""


PULL_SHARD = r"""
SELECT schema
FROM scrub_schemas
WHERE shard = {}
"""


# --------------------------------------------------------------------------- #
# Driver.                                                                     #
# --------------------------------------------------------------------------- #

def main(project_dir):
  print "Creating table on the cluster..."
  shark.Query(CREATE_TABLE)
  print "Pulling data..."
  dfs = [shark.Query(PULL_SHARD.format(x)) for x in range(10)]
  results_df = pd.concat(dfs, ignore_index=True)
  print "Pulled {} rows!".format(results_df.shape[0])
  raw_data_dir = os.path.realpath(os.path.join(project_dir, "data", "raw"))
  print "Cleaning..."
  old_files = os.listdir(raw_data_dir)
  for handle in old_files:
    if not handle.startswith("."):
      os.remove(os.path.join(raw_data_dir, handle))
      print "-- REMOVED:", handle
  ar_outpath = os.path.join(raw_data_dir, "schemas.dat")
  print "Writing schemas to {} ...".format(ar_outpath)
  joblib.dump(np.array(results_df.schema.tolist()), ar_outpath)
  print "Done!"


if __name__ == '__main__':
  project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
  main(project_dir)
