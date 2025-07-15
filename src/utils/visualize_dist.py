import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box, Point
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature
import os
import pandas as pd
from tqdm import tqdm
# --- Configuration ---
OCEAN_FEATURES_PATH = '/mnt/storagecube/joanna/ocean_features_projected.csv'
TIF_DIRECTORY = '/mnt/storagecube/joanna/Hydro'
OUTPUT_PLOT_PATH = 'world_distribution_hydro_ocean_plot_proj.png'

# --- 1. Load Ocean Features from CSV ---
try:
    # Read the CSV using pandas first
    df = pd.read_csv(OCEAN_FEATURES_PATH)
    
    # --- IMPORTANT: Inspect CSV Head ---
    print(f"\n--- Head of '{OCEAN_FEATURES_PATH}' ---")
    print(df.head())
    print(f"--- Columns in CSV: {df.columns.tolist()} ---\n")


    lat_cols = ['latitude', 'lat', 'Latitude', 'Lat']
    lon_cols = ['longitude', 'lon', 'Longitude', 'Lon']

    found_lat_col = None
    found_lon_col = None

    for col in lat_cols:
        if col in df.columns:
            found_lat_col = col
            break
    for col in lon_cols:
        if col in df.columns:
            found_lon_col = col
            break

    if found_lat_col is None or found_lon_col is None:
        raise ValueError(
            f"Could not find suitable latitude/longitude columns in CSV. "
            f"Looked for: {lat_cols} (lat) and {lon_cols} (lon). "
            f"Found columns: {df.columns.tolist()}. "
            f"Please inspect the CSV head above and update 'lat_cols'/'lon_cols' if necessary."
        )

    # Create a GeoDataFrame from the pandas DataFrame, specifying geometry
    ocean_features = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[found_lon_col], df[found_lat_col]),
        crs="EPSG:4326" # Set the CRS directly, assuming your CSV lat/lon are in WGS84
    )
    print(f"Ocean Features loaded successfully from CSV using '{found_lat_col}' and '{found_lon_col}'.")
    print(f"Ocean Features CRS: {ocean_features.crs}")

except Exception as e:
    print(f"Error loading ocean features: {e}")
    print("Please ensure your CSV exists, is readable, and contains coordinate columns.")
    exit()

# The rest of your code remains the same:
# --- 2. Reproject Ocean Features to WGS84 (EPSG:4326) if necessary ---
if ocean_features.crs != 'EPSG:4326':
    print("Reprojecting ocean_features to EPSG:4326...")
    ocean_features = ocean_features.to_crs(epsg=4326)
    print(f"Ocean Features (reprojected) CRS: {ocean_features.crs}")

# --- 3. Extract Lat/Lon for TIF files ---
tif_data = []
tif_files_list = [os.path.join(TIF_DIRECTORY, f) for f in os.listdir(TIF_DIRECTORY) if f.endswith('.tif')]

if not tif_files_list:
    print(f"No TIF files found in {TIF_DIRECTORY}. Please check the path.")
    exit()

print(f"Processing {len(tif_files_list)} TIF files...")

for tif_path in tqdm(tif_files_list, desc="Processing TIFs"):
    try:
        with rasterio.open(tif_path) as src:
            src_crs = src.crs
            src_bounds = src.bounds

            if src_crs != 'EPSG:4326':
                lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, 'EPSG:4326', *src_bounds)
            else:
                lon_min, lat_min, lon_max, lat_max = src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top

            center_lon = (lon_min + lon_max) / 2
            center_lat = (lat_min + lat_max) / 2
            
            tif_bbox_geom = box(lon_min, lat_min, lon_max, lat_max)

            tif_data.append({
                'filepath': tif_path,
                'geometry': Point(center_lon, center_lat),
                'bbox_geometry': tif_bbox_geom
            })
    except rasterio.errors.RasterioIOError as e:
        pass # Suppress repeated warnings if this is expected for some files
    except Exception as e:
        print(f"An unexpected error occurred with {tif_path}: {e}")

if not tif_data:
    print("No valid TIF data could be processed. Exiting.")
    exit()

tif_gdf = gpd.GeoDataFrame(tif_data, crs="EPSG:4326")

# --- 4. Determine which TIFs matched (for visualization) ---
print("Performing spatial intersection for visualization purposes...")

ocean_features_sindex = ocean_features.sindex

tif_gdf['matched'] = False
for i, tif_row in tqdm(tif_gdf.iterrows(), total=len(tif_gdf), desc="Checking matches"):
    possible_matches_index = list(ocean_features_sindex.intersection(tif_row['bbox_geometry'].bounds))
    possible_matches = ocean_features.iloc[possible_matches_index]

    if not possible_matches.empty and possible_matches.intersects(tif_row['bbox_geometry']).any():
        tif_gdf.at[i, 'matched'] = True

print(f"Matched TIFs for plotting: {tif_gdf['matched'].sum()} out of {len(tif_gdf)}")

try:
    fig, ax = plt.subplots(1, 1, figsize=(15, 10),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cartopy.feature.LAND, edgecolor='black', facecolor='lightgray', zorder=1)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue', zorder=0)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':', zorder=2)

    ax.set_global()

    ocean_features.plot(ax=ax, marker='.', markersize=2, color='blue', alpha=0.5, label='Ocean Features', zorder=3)

    tif_gdf[tif_gdf['matched']].plot(ax=ax, marker='o', color='green', markersize=10, label='Matched TIFs', zorder=4)
    tif_gdf[~tif_gdf['matched']].plot(ax=ax, marker='x', color='red', markersize=20, label='Unmatched TIFs', zorder=5)

    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

    ax.set_title('Distribution of TIF Files and Ocean Features (WGS84)')
    ax.legend(loc='lower left')

    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {OUTPUT_PLOT_PATH}") # Moved this print BEFORE plt.show()

    # plt.show() 
except Exception as e:
    print(f"\n--- ERROR during plotting ---")
    print(f"An error occurred while trying to generate or save the plot: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for more details
    print(f"--- End of plotting error ---")

print("\n--- Interpretation Guide ---") # This will now print even if there's a plotting error
print("1. Are red 'X' markers (unmatched TIFs) mostly over land or areas without any blue 'Ocean Features'?")
