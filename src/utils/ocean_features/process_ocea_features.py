import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
import rioxarray # Important: This enables the .rio accessor for xarray datasets
import pandas as pd
import numpy as np
from rasterio.transform import xy
from rasterio.merge import merge
from tqdm import tqdm
import os
import glob
import warnings

# Suppress specific warnings from rasterio/xarray that might be verbose but not critical
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message="The input raster has a CRS, but it was not set on the 'crs' attribute.")
warnings.filterwarnings("ignore", message="Could not find 'spatial_ref' or 'crs' in attrs. Defaulting to EPSG:4326.")


# --- Directories ---
script_dir = os.path.dirname(os.path.abspath(__file__)) # This correctly gets the directory of the current script

# Corrected paths relative to the script's location
bathy_path = os.path.join(script_dir, "bathy","sub_ice")
secchi_path = os.path.join(script_dir, "secchi", "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D_1747905614912.nc")
geotiff_dir = os.path.join(script_dir, "chlorophyll_level", "geotiffs")

# Output CSV will be placed in the same directory as the script
output_path = os.path.join(script_dir, "ocean_features_combined.csv")


# Chlorophyll TIFFs May 2024 to April 2025
tiff_files = [
    os.path.join(geotiff_dir, f"MY1DMM_CHLORA_2024-{month:02d}-01_rgb_3600x1800.TIFF") for month in range(5, 13)
] + [
    os.path.join(geotiff_dir, f"MY1DMM_CHLORA_2025-{month:02d}-01_rgb_3600x1800.TIFF") for month in range(1, 5)
]

# --- Define a target CRS for all data (WGS 84 Geographic) ---
TARGET_CRS = "EPSG:4326"

print("--- Step 1: Checking Coordinate Reference Systems (CRSs) of input files ---")

# --- Check Bathymetry (GEBCO) CRS ---
bathy_tif_files = sorted(glob.glob(os.path.join(bathy_path, "*.tif")))
if bathy_tif_files:
    with rasterio.open(bathy_tif_files[0]) as src:
        bathy_crs = src.crs
        print(f"\nBathymetry (GEBCO) CRS: {bathy_crs.to_string()} (EPSG:{src.crs.to_epsg()})")
        if bathy_crs != TARGET_CRS:
            print(f"  WARNING: Bathymetry CRS is not {TARGET_CRS}. It will be handled during merge/processing.")
else:
    print(f"\nERROR: No bathymetry TIFF files found at '{bathy_path}'. Please check the path and content. Exiting.")
    exit()

# --- Check Secchi (CMEMS NetCDF) CRS ---
secchi_ds_global = None # Declare globally to close later
if os.path.exists(secchi_path):
    try:
        # Load Secchi dataset once, it's typically smaller and will be interpolated in chunks
        secchi_ds_global = xr.open_dataset(secchi_path)

        # Attempt to set CRS using rioxarray's write_crs if not explicitly found, assuming EPSG:4326
        if not secchi_ds_global.rio.crs:
            print("  Secchi (CMEMS) dataset has no explicit CRS via .rio.crs. Assuming EPSG:4326.")
            secchi_ds_global = secchi_ds_global.rio.write_crs(TARGET_CRS, inplace=False) # Return new dataset
        
        print(f"\nSecchi (CMEMS) CRS: {secchi_ds_global.rio.crs.to_string()} (EPSG:{secchi_ds_global.rio.crs.to_epsg()})")
        if secchi_ds_global.rio.crs != TARGET_CRS:
            print(f"  WARNING: Secchi CRS is not {TARGET_CRS}. It will be reprojected during interpolation.")
            # Reproject the entire Secchi dataset if its CRS is explicitly different.
            secchi_ds_global = secchi_ds_global.rio.reproject(TARGET_CRS, resampling=Resampling.linear)

        secchi_var = list(secchi_ds_global.data_vars)[0]

    except Exception as e:
        print(f"\nERROR: Could not load or determine CRS for Secchi NetCDF at '{secchi_path}': {e}. Exiting.")
        if secchi_ds_global:
            secchi_ds_global.close()
        exit()
else:
    print(f"\nERROR: Secchi NetCDF file not found at '{secchi_path}'. Please check the path. Exiting.")
    exit()

# --- Check Chlorophyll CRS ---
if tiff_files:
    with rasterio.open(tiff_files[0]) as src:
        chl_crs = src.crs
        print(f"\nChlorophyll CRS: {chl_crs.to_string()} (EPSG:{src.crs.to_epsg()})")
        if chl_crs != TARGET_CRS:
            print(f"  WARNING: Chlorophyll CRS is not {TARGET_CRS}. It will be reprojected per tile.")
else:
    print(f"\nERROR: No chlorophyll TIFF files found at '{geotiff_dir}'. Please check directory and file names. Exiting.")
    exit()

print("\n--- Step 1 Complete: CRS Verification ---")
print("All data will be processed and reprojected to EPSG:4326 as the common standard.")


print("\n--- Step 2: Processing and Combining Data ---")

# --- 1. Load and mosaic GEBCO Bathymetry Tiles ---
src_files_to_mosaic = [rasterio.open(fp) for fp in bathy_tif_files]

mosaic_full, out_transform_full = merge(src_files_to_mosaic)
mosaic_full = mosaic_full[0] # Take the first band (H, W)

height_full, width_full = mosaic_full.shape

# Close the source files to free memory
for src in src_files_to_mosaic:
    src.close()

# If the merged bathy mosaic is not in TARGET_CRS, reproject it now.
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


# Define tiling parameters
tile_size_rows = 500
tile_size_cols = 500

# Open the output CSV file in append mode. Write header only once.
file_exists = os.path.exists(output_path)
output_file = open(output_path, 'a')
if not file_exists or os.stat(output_path).st_size == 0: # Check if file is empty or doesn't exist
    pd.DataFrame(columns=["lat", "lon", "bathy", "chlorophyll", "secchi"]).to_csv(output_file, index=False, header=True)
else:
    pass # If file exists and is not empty, ensure no header is written again

# Initialize global metrics for dropped rows
total_rows_before_dropna = 0
total_rows_after_dropna = 0
total_dropped_rows = 0
dropped_bathy_values = [] # Store bathy values of dropped rows for analysis

# Iterate over tiles
for r_start in tqdm(range(0, height_full, tile_size_rows), desc="Processing Rows"):
    r_end = min(r_start + tile_size_rows, height_full)

    for c_start in tqdm(range(0, width_full, tile_size_cols), desc=f"  Processing Cols (Row {r_start}-{r_end})", leave=False):
        c_end = min(c_start + tile_size_cols, width_full)

        # --- Extract bathy for current tile ---
        mosaic_tile = mosaic_full[r_start:r_end, c_start:c_end]

        # --- Create coordinates for current tile (these are in TARGET_CRS implicitly now) ---
        lats_tile, lons_tile = [], []
        for r_offset in range(r_end - r_start):
            for c_offset in range(c_end - c_start):
                global_row = r_start + r_offset
                global_col = c_start + c_offset
                lon, lat = xy(out_transform_full, global_row, global_col)
                lats_tile.append(lat)
                lons_tile.append(lon)

        # --- Load and average Chlorophyll for current tile ---
        chl_arrays_tile = []
        expected_tile_height = r_end - r_start
        expected_tile_width = c_end - c_start

        # Determine the target geographic bounds for this tile (in TARGET_CRS)
        min_lon_tile, max_lat_tile = xy(out_transform_full, r_start, c_start)
        max_lon_tile, min_lat_tile = xy(out_transform_full, r_end, c_end)

        # Create a new transform for the current tile in TARGET_CRS.
        dst_transform_tile = rasterio.transform.from_bounds(
            west=min_lon_tile, south=min_lat_tile, east=max_lon_tile, north=max_lat_tile,
            width=expected_tile_width, height=expected_tile_height
        )

        for tiff_path in tiff_files:
            with rasterio.open(tiff_path) as src_chl:
                reprojected_chl_data = np.full(
                    (expected_tile_height, expected_tile_width), np.nan, dtype=np.float32
                )

                try:
                    reproject(
                        source=rasterio.band(src_chl, 1),
                        destination=reprojected_chl_data,
                        src_transform=src_chl.transform,
                        src_crs=src_chl.crs,
                        dst_transform=dst_transform_tile,
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.nearest,
                        num_threads=os.cpu_count(),
                    )

                    if src_chl.nodata is not None:
                        reprojected_chl_data[reprojected_chl_data == src_chl.nodata] = np.nan
                    chl_arrays_tile.append(reprojected_chl_data)

                except Exception as e:
                    print(f"Warning: Error reprojecting {os.path.basename(tiff_path)} for tile "
                          f"R{r_start}-{r_end}, C{c_start}-{c_end}: {e}")
                    chl_arrays_tile.append(np.full((expected_tile_height, expected_tile_width), np.nan, dtype="float32"))

        if chl_arrays_tile:
            chl_stack_tile = np.stack(chl_arrays_tile, axis=0)
            chl_mean_tile = np.nanmean(chl_stack_tile, axis=0)
            chl_interp_tile = chl_mean_tile.flatten()
        else:
            chl_interp_tile = np.full(expected_tile_height * expected_tile_width, np.nan, dtype="float32")


        # --- Interpolate Secchi for current tile's coordinates ---
        secchi_interp_tile = secchi_ds_global[secchi_var].interp(
            latitude=("points", lats_tile), longitude=("points", lons_tile), method="linear",
            kwargs={"fill_value": np.nan}
        ).values.flatten()

        # Add assertions to catch size mismatches early
        assert len(lats_tile) == len(lons_tile) == len(mosaic_tile.flatten()) == len(chl_interp_tile) == len(secchi_interp_tile), \
            f"Array lengths do not match! Lats: {len(lats_tile)}, Lons: {len(lons_tile)}, " \
            f"Bathy: {len(mosaic_tile.flatten())}, Chl: {len(chl_interp_tile)}, Secchi: {len(secchi_interp_tile)}"

        df_tile = pd.DataFrame({
            "lat": lats_tile,
            "lon": lons_tile,
            "bathy": mosaic_tile.flatten(),
            "chlorophyll": chl_interp_tile,
            "secchi": secchi_interp_tile,
        })

        # --- Logging before dropna() ---
        rows_before = len(df_tile)
        nan_counts = df_tile.isnull().sum()
        total_rows_before_dropna += rows_before

        # Identify rows that will be dropped and capture their bathymetry
        rows_to_drop = df_tile[df_tile.isnull().any(axis=1)]
        dropped_bathy_values.extend(rows_to_drop['bathy'].tolist())
        
        # --- Remove NaN rows for the tile ---
        df_tile = df_tile.dropna()

        # --- Logging after dropna() ---
        rows_after = len(df_tile)
        dropped_count_tile = rows_before - rows_after
        total_rows_after_dropna += rows_after
        total_dropped_rows += dropped_count_tile

        if dropped_count_tile > 0:
            print(f"\n  Tile R{r_start}-{r_end}, C{c_start}-{c_end}:")
            print(f"    Rows before dropna: {rows_before}")
            print(f"    NaNs per column before dropna:\n{nan_counts}")
            print(f"    Rows dropped: {dropped_count_tile}")
            if len(rows_to_drop) > 0:
                print(f"    Min/Max/Mean Bathy of dropped rows: {np.nanmin(rows_to_drop['bathy']):.2f}m / {np.nanmax(rows_to_drop['bathy']):.2f}m / {np.nanmean(rows_to_drop['bathy']):.2f}m")
            print("-" * 50)


        # --- Append to CSV ---
        df_tile.to_csv(output_file, index=False, header=False) # No header for subsequent appends

# Close the output file after all tiles are processed
output_file.close()

# Close the global Secchi dataset
if secchi_ds_global:
    secchi_ds_global.close()

print(f"\n--- Step 2 Complete: DataFrame saved to: {output_path} ---")
print("\n--- Dropna() Impact Summary ---")
print(f"Total rows considered before dropna(): {total_rows_before_dropna}")
print(f"Total rows retained after dropna(): {total_rows_after_dropna}")
print(f"Total rows dropped by dropna(): {total_dropped_rows}")

if total_dropped_rows > 0:
    print(f"\nAnalysis of Bathymetry in Dropped Rows:")
    print(f"  Min Bathy of all dropped rows: {np.nanmin(dropped_bathy_values):.2f}m")
    print(f"  Max Bathy of all dropped rows: {np.nanmax(dropped_bathy_values):.2f}m")
    print(f"  Mean Bathy of all dropped rows: {np.nanmean(dropped_bathy_values):.2f}m")
    print(f"  Median Bathy of all dropped rows: {np.nanmedian(dropped_bathy_values):.2f}m")
    print(f"  Standard Deviation of Bathy in dropped rows: {np.nanstd(dropped_bathy_values):.2f}m")
else:
    print("No rows were dropped by dropna().")

print("Processing finished.")