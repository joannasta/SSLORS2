import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
import pandas as pd
import numpy as np
from rasterio.transform import xy
from rasterio.merge import merge
from tqdm import tqdm
import os
import glob

base = os.getcwd()
bathy_path = os.path.join(base, "bathy", "sub_ice")
existing_csv_path = "/mnt/storagecube/joanna/ocean_features_projected.csv"
output_updated_csv_path = os.path.join("/mnt/storagecube/joanna/", "ocean_features_combined_updated_bathy.csv")

TARGET_CRS = "EPSG:4326"

print("--- Step 1: Generating New Bathymetry Data ---")

bathy_tif_files = sorted(glob.glob(os.path.join(bathy_path, "*.tif")))
print("Bathymetry TIFF files found:", bathy_tif_files)

if not bathy_tif_files:
    print(f"\nERROR: No bathymetry TIFF files found at '{bathy_path}'. Please check the path. Exiting.")
    exit()

src_files_to_mosaic = [rasterio.open(fp) for fp in bathy_tif_files]

mosaic_full, out_transform_full = merge(src_files_to_mosaic)
mosaic_full = mosaic_full[0]

height_full, width_full = mosaic_full.shape

for src in src_files_to_mosaic:
    src.close()

with rasterio.open(bathy_tif_files[0]) as first_src:
    if first_src.crs != TARGET_CRS:
        print(f"  Reprojecting full bathymetry mosaic from {first_src.crs.to_string()} to {TARGET_CRS}...")
        reprojected_mosaic = np.empty_like(mosaic_full, dtype=mosaic_full.dtype)
        reproject(
            source=mosaic_full,
            destination=reprojected_mosaic,
            src_transform=out_transform_full,
            src_crs=first_src.crs,
            dst_transform=out_transform_full,
            dst_crs=TARGET_CRS,
            resampling=Resampling.nearest,
            num_threads=os.cpu_count()
        )
        mosaic_full = reprojected_mosaic
    
    mosaic_final_crs = TARGET_CRS

print("Creating new bathymetry DataFrame...")
new_bathy_data = []

tile_size_rows = 500
tile_size_cols = 500

for r_start in tqdm(range(0, height_full, tile_size_rows), desc="Extracting Bathy Rows"):
    r_end = min(r_start + tile_size_rows, height_full)

    for c_start in tqdm(range(0, width_full, tile_size_cols), desc=f"  Extracting Bathy Cols (Row {r_start}-{r_end})", leave=False):
        c_end = min(c_start + tile_size_cols, width_full)

        mosaic_tile = mosaic_full[r_start:r_end, c_start:c_end]

        for r_offset in range(r_end - r_start):
            for c_offset in range(c_end - c_start):
                global_row = r_start + r_offset
                global_col = c_start + c_offset
                lon, lat = xy(out_transform_full, global_row, global_col)
                bathy_val = mosaic_tile[r_offset, c_offset]
                new_bathy_data.append({"lat": lat, "lon": lon, "new_bathy": bathy_val})

new_bathy_df = pd.DataFrame(new_bathy_data)

print(f"Generated new bathymetry data for {len(new_bathy_df)} points.")
print("Sample of new bathymetry data:")
print(new_bathy_df.head())

print("\n--- Step 2: Loading Existing CSV and Merging ---")

print(f"Loading existing CSV file from: {existing_csv_path}...")
try:
    df_existing = pd.read_csv(existing_csv_path)
    print(f"Loaded {len(df_existing)} rows from existing CSV.")
    print("Sample of existing DataFrame:")
    print(df_existing.head())
except FileNotFoundError:
    print(f"ERROR: Existing CSV file not found at '{existing_csv_path}'. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load existing CSV: {e}")
    exit()

print("\nMerging existing CSV with new bathymetry data...")
df_merged = pd.merge(
    df_existing,
    new_bathy_df,
    on=['lat', 'lon'],
    how='left'
)

df_merged['bathy'] = df_merged['new_bathy'].fillna(df_merged['bathy'])

df_merged = df_merged.drop(columns=['new_bathy'])

print("Merge complete.")
print("Sample of merged DataFrame with updated bathy:")
print(df_merged.head())

if df_merged['bathy'].isnull().any():
    print(f"WARNING: There are still {df_merged['bathy'].isnull().sum()} NaN values in the 'bathy' column after merge and fillna. This indicates points in your CSV that did not have a corresponding new bathymetry value.")

print("\n--- Step 3: Saving Updated CSV ---")

print(f"Saving updated CSV to: {output_updated_csv_path}...")
df_merged.to_csv(output_updated_csv_path, index=False)
print(f"Updated CSV saved successfully to: {output_updated_csv_path}")

print("\nProcessing finished.")