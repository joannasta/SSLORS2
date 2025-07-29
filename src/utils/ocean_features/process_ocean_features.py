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
base = os.getcwd()
bathy_path = os.path.join(base,  "bathy","sub_ice")
secchi_path = os.path.join(base,  "secchi", "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D_1747905614912.nc")

script_dir = os.path.dirname(os.path.abspath(__file__))
geotiff_dir = os.path.join(script_dir, "chlorophyll_level", "geotiffs")

# Chlorophyll TIFFs May 2024 to April 2025
tiff_files = [
    os.path.join(geotiff_dir, f"MY1DMM_CHLORA_2024-{month:02d}-01_rgb_3600x1800.TIFF") for month in range(5, 13)
] + [
    os.path.join(geotiff_dir, f"MY1DMM_CHLORA_2025-{month:02d}-01_rgb_3600x1800.TIFF") for month in range(1, 5)
]

# --- Define a target CRS for all data (WGS 84 Geographic) ---
# This is a common standard for global oceanographic data.
TARGET_CRS = "EPSG:4326"

print("--- Step 1: Checking Coordinate Reference Systems (CRSs) of input files ---")

# --- Check Bathymetry (GEBCO) CRS ---
bathy_tif_files = sorted(glob.glob(os.path.join(bathy_path, "*.tif")))
print("bathy_path",bathy_path)
print("bathy_tif_files",bathy_tif_files)
if bathy_tif_files:
    with rasterio.open(bathy_tif_files[0]) as src:
        bathy_crs = src.crs
        print(f"\nBathymetry (GEBCO) CRS: {bathy_crs.to_string()} (EPSG:{src.crs.to_epsg()})")
        if bathy_crs != TARGET_CRS:
            print(f"  WARNING: Bathymetry CRS is not {TARGET_CRS}. It will be handled during merge/processing.")
else:
    print("\nERROR: No bathymetry TIFF files found. Please check 'bathy_path'. Exiting.")
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
            # For most CMEMS data, this won't be necessary as it's already EPSG:4326.
            secchi_ds_global = secchi_ds_global.rio.reproject(TARGET_CRS, resampling=Resampling.linear)

        secchi_var = list(secchi_ds_global.data_vars)[0]

    except Exception as e:
        print(f"\nERROR: Could not load or determine CRS for Secchi NetCDF: {e}. Exiting.")
        if secchi_ds_global:
            secchi_ds_global.close()
        exit()
else:
    print("\nERROR: Secchi NetCDF file not found. Please check 'secchi_path'. Exiting.")
    exit()

# --- Check Chlorophyll CRS ---
if tiff_files:
    with rasterio.open(tiff_files[0]) as src:
        chl_crs = src.crs
        print(f"\nChlorophyll CRS: {chl_crs.to_string()} (EPSG:{src.crs.to_epsg()})")
        if chl_crs != TARGET_CRS:
            print(f"  WARNING: Chlorophyll CRS is not {TARGET_CRS}. It will be reprojected per tile.")
else:
    print("\nERROR: No chlorophyll TIFF files found. Please check 'geotiff_dir' and file names. Exiting.")
    exit()

print("\n--- Step 1 Complete: CRS Verification ---")
print("All data will be processed and reprojected to EPSG:4326 as the common standard.")


print("\n--- Step 2: Processing and Combining Data ---")

# --- 1. Load and mosaic GEBCO Bathymetry Tiles ---
# Merge will attempt to use the CRS of the first dataset if CRSs differ among source files.
# It's safest if all bathy tiles already have the same CRS.
src_files_to_mosaic = [rasterio.open(fp) for fp in bathy_tif_files]

mosaic_full, out_transform_full = merge(src_files_to_mosaic)
mosaic_full = mosaic_full[0] # Take the first band (H, W)

height_full, width_full = mosaic_full.shape

# Close the source files to free memory
for src in src_files_to_mosaic:
    src.close()

# If the merged bathy mosaic is not in TARGET_CRS, reproject it now.
# This ensures the base grid (lats/lons) generated from out_transform_full is in TARGET_CRS.
with rasterio.open(bathy_tif_files[0]) as first_src: # Re-open to get metadata for reprojection
    if first_src.crs != TARGET_CRS:
        print(f"  Reprojecting full bathymetry mosaic from {first_src.crs.to_string()} to {TARGET_CRS}...")
        # Create a new destination array for the reprojected mosaic
        reprojected_mosaic = np.empty_like(mosaic_full, dtype=mosaic_full.dtype)
        reproject(
            source=mosaic_full,
            destination=reprojected_mosaic,
            src_transform=out_transform_full,
            src_crs=first_src.crs, # Use the actual CRS of the merged mosaic
            dst_transform=out_transform_full, # Keep the same transform but with new CRS
            dst_crs=TARGET_CRS,
            resampling=Resampling.nearest, # Or bilinear
            num_threads=os.cpu_count()
        )
        mosaic_full = reprojected_mosaic
        # Update transform's CRS as well, though `xy` will implicitly use the destination CRS
        # out_transform_full is now conceptually linked to TARGET_CRS
    
    # Store the final CRS of the mosaic for coordinate generation
    mosaic_final_crs = TARGET_CRS


# Define tiling parameters
# You might need to adjust these based on your available RAM and data size
tile_size_rows = 500
tile_size_cols = 500

output_path = os.path.join(base, "ocean_features_combined.csv")

# Open the output CSV file in append mode. Write header only once.
file_exists = os.path.exists(output_path)
output_file = open(output_path, 'a')
if not file_exists or os.stat(output_path).st_size == 0: # Check if file is empty or doesn't exist
    pd.DataFrame(columns=["lat", "lon", "bathy", "chlorophyll", "secchi"]).to_csv(output_file, index=False, header=True)
else:
    # If file exists and is not empty, ensure no header is written again
    pass

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
                # xy function converts pixel to (lon, lat) based on the mosaic's transform
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
        # This will be the destination transform for reprojecting chlorophyll data.
        dst_transform_tile = rasterio.transform.from_bounds(
            west=min_lon_tile, south=min_lat_tile, east=max_lon_tile, north=max_lat_tile,
            width=expected_tile_width, height=expected_tile_height
        )

        for tiff_path in tiff_files:
            with rasterio.open(tiff_path) as src_chl:
                reprojected_chl_data = np.full(
                    (expected_tile_height, expected_tile_width), np.nan, dtype=np.float32
                )

                # Perform the reprojection of the chlorophyll data to the TARGET_CRS grid
                try:
                    reproject(
                        source=rasterio.band(src_chl, 1),
                        destination=reprojected_chl_data,
                        src_transform=src_chl.transform,
                        src_crs=src_chl.crs,
                        dst_transform=dst_transform_tile, # Use the tile-specific destination transform
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.nearest, # Or Resampling.bilinear, Resampling.cubic for smoother results
                        num_threads=os.cpu_count(),
                        # Define bounds that cover the chlorophyll data. Use boundless for safety.
                        # It is better to rely on `reproject` to handle source windowing implicitly.
                        # window=src_chl.window(min_lon_tile, min_lat_tile, max_lon_tile, max_lat_tile)
                    )

                    if src_chl.nodata is not None:
                        reprojected_chl_data[reprojected_chl_data == src_chl.nodata] = np.nan
                    chl_arrays_tile.append(reprojected_chl_data)

                except Exception as e:
                    print(f"Warning: Error reprojecting {os.path.basename(tiff_path)} for tile "
                          f"R{r_start}-{r_end}, C{c_start}-{c_end}: {e}")
                    # Append NaNs if reprojection fails for this specific tile/file
                    chl_arrays_tile.append(np.full((expected_tile_height, expected_tile_width), np.nan, dtype="float32"))

        if chl_arrays_tile:
            chl_stack_tile = np.stack(chl_arrays_tile, axis=0)
            chl_mean_tile = np.nanmean(chl_stack_tile, axis=0)
            chl_interp_tile = chl_mean_tile.flatten()
        else:
            chl_interp_tile = np.full(expected_tile_height * expected_tile_width, np.nan, dtype="float32")


        # --- Interpolate Secchi for current tile's coordinates ---
        # The secchi_ds_global is already reprojected to TARGET_CRS if needed.
        secchi_interp_tile = secchi_ds_global[secchi_var].interp(
            latitude=("points", lats_tile), longitude=("points", lons_tile), method="linear",
            kwargs={"fill_value": np.nan} # Ensure NaNs for out-of-bounds
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

        # --- Remove NaN rows for the tile ---
        df_tile = df_tile.dropna()

        # --- Append to CSV ---
        df_tile.to_csv(output_file, index=False, header=False) # No header for subsequent appends

# Close the output file after all tiles are processed
output_file.close()

# Close the global Secchi dataset
if secchi_ds_global:
    secchi_ds_global.close()

print(f"\n--- Step 2 Complete: DataFrame saved to: {output_path} ---")
print("Processing finished.")