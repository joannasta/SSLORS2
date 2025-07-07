import rasterio
from pathlib import Path

tif_file_path = "/data/joanna/Hydro/patch_300958.tif" # <--- IMPORTANT: REPLACE WITH AN ACTUAL PATH TO ONE OF YOUR .TIF FILES

try:
    with rasterio.open(tif_file_path) as src:
        print(f"CRS of {tif_file_path}: {src.crs}")
        print(f"CRS as WKT: {src.crs.to_wkt()}") # More detailed info
        print(f"Units: {src.crs.axis_info[0].unit_name}") # Should be 'metre' if projected
except rasterio.errors.RasterioIOError as e:
    print(f"Error: Could not open {tif_file_path}. Please ensure the path is correct and it's a valid TIFF.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")