import lmdb
import numpy as np
# import desired deep-learning library:
# numpy, torch, tensorflow, paddle, flax, mlx
from safetensors.numpy import load
from pathlib import Path

encoded_path = "/faststorage/joanna/lmdb/Encoded-Hydro"

# Make sure to only open the environment once
# and not everytime an item is accessed.
env = lmdb.open(str(encoded_path), readonly=True)

with env.begin() as txn:
  # string encoding is required to map the string to an LMDB key
  safetensor_dict = load(txn.get("patch_0".encode()))

tensor = np.stack(
  [
    safetensor_dict[key]
    for key in [
      "B01",
      "B02",
      "B03",
      "B04",
      "B05",
      "B06",
      "B07",
      "B08",
      "B8A",
      "B09",
      "B11",
      "B12",
    ]
  ]
)
assert tensor.shape == (12, 256, 256)