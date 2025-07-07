import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from tqdm.notebook import tqdm # For progress bar

# --- Configuration ---
OCEAN_FEATURES_PATH = '/mnt/storagecube/joanna/ocean_features_filtered.csv' # Or .shp, etc.
TIF_DIRECTORY = '/data/joanna/Hydro' # Directory containing your TIF files
OUTPUT_PLOT_PATH = 'world_distribution_hydro_ocean_plot.png'

# --- 1. Load Ocean Features ---
try:
    ocean_features = gpd.read_file(OCEAN_FEATURES_PATH)
    print(f"Ocean Features CRS: {ocean_features.crs}")
except Exception as e:
    print(f"Error loading ocean features: {e}")
    print("Please check the path and file format for OCEAN_FEATURES_PATH.")
    exit()

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

            # If TIF is not in WGS84, reproject its bounds to WGS84 for consistency
            if src_crs != 'EPSG:4326':
                # transform_bounds returns (west, south, east, north) in target CRS
                lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, 'EPSG:4326', *src_bounds)
            else:
                lon_min, lat_min, lon_max, lat_max = src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top

            # Calculate centroid for plotting
            center_lon = (lon_min + lon_max) / 2
            center_lat = (lat_min + lat_max) / 2
            
            # Create a shapely Polygon for the TIF's bounding box
            tif_bbox_geom = box(lon_min, lat_min, lon_max, lat_max)

            tif_data.append({
                'filepath': tif_path,
                'geometry': Point(center_lon, center_lat),
                'bbox_geometry': tif_bbox_geom # Store the bbox for intersection checking
            })
    except rasterio.errors.RasterioIOError as e:
        print(f"Warning: Could not open or read CRS/bounds for {tif_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {tif_path}: {e}")

if not tif_data:
    print("No valid TIF data could be processed. Exiting.")
    exit()

tif_gdf = gpd.GeoDataFrame(tif_data, crs="EPSG:4326")

# --- 4. Determine which TIFs matched (for visualization) ---
# This simulates your original matching process.
# We'll use a simple spatial intersection for demonstration.
# For a real scenario, you'd use the results of your actual matching process.

print("Performing spatial intersection for visualization purposes...")
tif_gdf['matched'] = False
for i, tif_row in tqdm(tif_gdf.iterrows(), total=len(tif_gdf), desc="Checking matches"):
    if ocean_features.intersects(tif_row['bbox_geometry']).any():
        tif_gdf.at[i, 'matched'] = True

print(f"Matched TIFs for plotting: {tif_gdf['matched'].sum()} out of {len(tif_gdf)}")

# --- 5. Plotting ---

fig, ax = plt.subplots(1, 1, figsize=(15, 10),
                       subplot_kw={'projection': ccrs.PlateCarree()}) # PlateCarree is good for global maps

# Add natural earth features for context
ax.add_feature(cartopy.feature.LAND, edgecolor='black', facecolor='lightgray', zorder=1)
ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue', zorder=0)
ax.add_feature(cartopy.feature.COASTLINE, zorder=2)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':', zorder=2)

# Set extent to show the whole world
ax.set_global()

# Plot Ocean Features (e.g., in blue with some transparency)
ocean_features.plot(ax=ax, color='blue', alpha=0.5, label='Ocean Features', zorder=3)

# Plot TIF origins: Matched in one color, Unmatched in another
tif_gdf[tif_gdf['matched']].plot(ax=ax, marker='o', color='green', markersize=10, label='Matched TIFs', zorder=4)
tif_gdf[~tif_gdf['matched']].plot(ax=ax, marker='x', color='red', markersize=20, label='Unmatched TIFs', zorder=5) # Larger 'x' to stand out

# Add gridlines and labels
ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

# Add title and legend
ax.set_title('Distribution of TIF Files and Ocean Features (WGS84)')
ax.legend(loc='lower left')

# Save the plot
plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {OUTPUT_PLOT_PATH}")
print("\n--- Interpretation Guide ---")
print("1. Are red 'X' markers (unmatched TIFs) mostly over land or areas without any blue 'Ocean Features'?")
print("   -> This would indicate TIFs are in irrelevant areas or your ocean feature data is incomplete.")
print("2. Are red 'X' markers very close to blue 'Ocean Features' but not overlapping?")
print("   -> This suggests misalignment or strict matching criteria. Consider buffering or adjusting tolerance.")
print("3. Are blue 'Ocean Features' densely clustered in some areas, while TIFs are elsewhere?")
print("   -> You might be trying to match TIFs from regions where ocean features are sparse or non-existent in your current dataset.")
print("4. Do the green 'O' markers (matched TIFs) appear mostly within or directly on the blue 'Ocean Features'?")
print("   -> This validates your current matching logic for successful cases.")