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
TARGET_CRS = "EPSG:4326"

# --- Check Bathymetry CRS ---
bathy_tif_files = sorted(glob.glob(os.path.join(bathy_path, "*.tif")))
if bathy_tif_files:
    with rasterio.open(bathy_tif_files[0]) as src:
        bathy_crs = src.crs

# --- Check Secchi CRS ---
secchi_ds_global = None 
if os.path.exists(secchi_path):
    try:
        secchi_ds_global = xr.open_dataset(secchi_path)     
        if not secchi_ds_global.rio.crs:
            secchi_ds_global = secchi_ds_global.rio.write_crs(TARGET_CRS, inplace=False) 
    
        if secchi_ds_global.rio.crs != TARGET_CRS:
            secchi_ds_global = secchi_ds_global.rio.reproject(TARGET_CRS, resampling=Resampling.linear)

        secchi_var = list(secchi_ds_global.data_vars)[0]

    except Exception as e:
        print(f"ERROR: Could not load or determine CRS for Secchi NetCDF: {e}. Exiting.")
else:
    print("ERROR: Secchi file not found.")


# --- Check Chlorophyll CRS ---
if tiff_files:
    with rasterio.open(tiff_files[0]) as src:
        chl_crs = src.crs
else:
    print("\nERROR: No chlorophyll TIFF files found. Please check 'geotiff_dir' and file names. Exiting.")

# --- 1. Load and mosaic Bathymetry Tiles ---

src_files_to_mosaic = [rasterio.open(fp) for fp in bathy_tif_files]

mosaic_full, out_transform_full = merge(src_files_to_mosaic)
mosaic_full = mosaic_full[0] 

height_full, width_full = mosaic_full.shape

for src in src_files_to_mosaic:
    src.close()

with rasterio.open(bathy_tif_files[0]) as first_src: 
    if first_src.crs != TARGET_CRS:
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

tile_size_rows = 500
tile_size_cols = 500

output_path = os.path.join("/mnt/storagecube/joanna/", "ocean_features.csv")

file_exists = os.path.exists(output_path)
output_file = open(output_path, 'a')
if not file_exists or os.stat(output_path).st_size == 0: 
    pd.DataFrame(columns=["lat", "lon", "bathy", "chlorophyll", "secchi"]).to_csv(output_file, index=False, header=True)
else:
    pass

for r_start in tqdm(range(0, height_full, tile_size_rows), desc="Processing Rows"):
    r_end = min(r_start + tile_size_rows, height_full)

    for c_start in tqdm(range(0, width_full, tile_size_cols), desc=f"  Processing Cols (Row {r_start}-{r_end})", leave=False):
        c_end = min(c_start + tile_size_cols, width_full)
        mosaic_tile = mosaic_full[r_start:r_end, c_start:c_end]
        lats_tile, lons_tile = [], []
        for r_offset in range(r_end - r_start):
            for c_offset in range(c_end - c_start):
                global_row = r_start + r_offset
                global_col = c_start + c_offset
                lon, lat = xy(out_transform_full, global_row, global_col)
                lats_tile.append(lat)
                lons_tile.append(lon)

        chl_arrays_tile = []
        expected_tile_height = r_end - r_start
        expected_tile_width = c_end - c_start
        
        min_lon_tile, max_lat_tile = xy(out_transform_full, r_start, c_start)
        max_lon_tile, min_lat_tile = xy(out_transform_full, r_end, c_end)

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
                        num_threads=os.cpu_count()
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

        df_tile = df_tile.dropna()
        df_tile.to_csv(output_file, index=False, header=False) # No header for subsequent appends

output_file.close()

if secchi_ds_global:
    secchi_ds_global.close()

print(f"\n Complete: DataFrame saved to: {output_path}")