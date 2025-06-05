import lmdb
import pickle  # If values are serialized Python objects
import numpy as np  # If values might be arrays
from PIL import Image
import io

# Path to your LMDB database
lmdb_path = "/faststorage/joanna/lmdb/Encoded-Hydro"

# Open the LMDB environment
env = lmdb.open(lmdb_path, readonly=True)

with env.begin() as txn:
    with txn.cursor() as cursor:
        for idx, (key, value) in enumerate(cursor):
            print(f"Entry {idx + 1}:")
            print(f"  Key: {key.decode('utf-8') if isinstance(key, bytes) else key}")
            print(f"  Value size: {len(value)} bytes")
            
            # Attempt to parse the value as an image
            try:
                image = Image.open(io.BytesIO(value))
                print(f"  Parsed Value Type: Image")
                print(f"  Image Size: {image.size}")
                print(f"  Image Mode: {image.mode}")
            except Exception:
                # If not an image, attempt to parse the value as serialized data
                try:
                    data = pickle.loads(value)
                    print(f"  Parsed Value Type: {type(data)}")
                    if isinstance(data, np.ndarray):
                        print(f"  Array Shape: {data.shape}")
                    elif isinstance(data, (list, dict)):
                        print(f"  Preview: {str(data)[:100]}")  # Preview list/dict
                except Exception as e:
                    print(f"  Could not parse value: {e}")

            print("-" * 50)
