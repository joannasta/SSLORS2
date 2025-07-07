import rasterio
from pathlib import Path
import numpy as np
import argparse
from typing import Tuple # <--- ADD THIS LINE

def get_bbox_from_tifs(tif_dir: Path) -> Tuple[float, float, float, float]:
    """
    Calculates the overall bounding box (min_lat, max_lat, min_lon, max_lon)
    for all .tif files in the given directory (and its subdirectories).
    """
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')

    tif_files = sorted(list(tif_dir.rglob("*.tif"))) # Use rglob to find all TIFs
    if not tif_files:
        raise ValueError(f"No .tif files found in {tif_dir}")

    print(f"Scanning {len(tif_files)} TIF files to determine overall bounding box...")

    for i, file_path in enumerate(tif_files):
        if i % 1000 == 0:
            print(f"  Processing TIF {i}/{len(tif_files)}: {file_path.name}")
        try:
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                # Rasterio bounds are (left, bottom, right, top)
                # left=min_lon, bottom=min_lat, right=max_lon, top=max_lat
                min_lat = min(min_lat, bounds.bottom)
                max_lat = max(max_lat, bounds.top)
                min_lon = min(min_lon, bounds.left)
                max_lon = max(max_lon, bounds.right)
        except rasterio.errors.RasterioIOError as e:
            print(f"WARNING: Could not open {file_path} for bounding box calculation: {e}. Skipping.")
        except Exception as e:
            print(f"WARNING: An unexpected error occurred with {file_path}: {e}. Skipping.")

    print("Finished scanning TIF files.")
    return min_lat, max_lat, min_lon, max_lon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate bounding box of TIF files.")
    parser.add_argument("--tif_dir", type=str, required=True,
                        help="Path to the directory containing TIF files (e.g., /faststorage/joanna/Hydro/raw_data).")
    args = parser.parse_args()

    try:
        min_lat, max_lat, min_lon, max_lon = get_bbox_from_tifs(Path(args.tif_dir))
        print(f"\nOverall Bounding Box for TIFs in {args.tif_dir}:")
        print(f"  Min Latitude: {min_lat}")
        print(f"  Max Latitude: {max_lat}")
        print(f"  Min Longitude: {min_lon}")
        print(f"  Max Longitude: {max_lon}")
        print("\nUse these values in the next step to filter your large CSV.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")