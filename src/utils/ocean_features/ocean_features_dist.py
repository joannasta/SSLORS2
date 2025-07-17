import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Cartopy imports for mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the path to your ocean features CSV file
#'/mnt/storagecube/joanna/ocean_features_projected.csv'
OCEAN_FEATURES_PATH = Path('/home/joanna/SSLORS2/src/utils/ocean_features/ocean_features_combined.csv')

# Define output paths for the generated maps
output_bathy_map_path = Path("geographic_bathymetry_distribution_proj.png")
output_chlorophyll_map_path = Path("geographic_chlorophyll_distribution_proj.png")
output_secchi_map_path = Path("geographic_secchi_distribution_proj.png")

# --- Load Ocean Features Data ---
try:
    ocean_df = pd.read_csv(OCEAN_FEATURES_PATH)
    # Check if all required columns exist in the DataFrame
    required_cols = ['lat', 'lon', 'bathy', 'chlorophyll', 'secchi']
    if not all(col in ocean_df.columns for col in required_cols):
        raise ValueError(f"Ocean features CSV missing one or more required columns: {required_cols}")
    
    print(f"Successfully loaded {len(ocean_df)} data points from {OCEAN_FEATURES_PATH}")

except FileNotFoundError:
    print(f"Error: Ocean features CSV not found at {OCEAN_FEATURES_PATH}.")
    exit()
except Exception as e:
    print(f"Error loading or processing ocean features CSV: {e}")
    exit()

# --- Plot Geographic Distribution of Bathymetry ---
print(f"Generating map for Bathymetry and saving to {output_bathy_map_path}...")
fig_bathy = plt.figure(figsize=(15, 10))
ax_bathy = fig_bathy.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # Use add_subplot for cleaner figure management

# Add standard map features for context
ax_bathy.add_feature(cfeature.LAND, facecolor='lightgray')
ax_bathy.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_bathy.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax_bathy.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax_bathy.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray')
ax_bathy.add_feature(cfeature.RIVERS, edgecolor='blue')

# Plot the bathymetry data points
scatter_bathy = ax_bathy.scatter(
    ocean_df['lon'], # Longitude
    ocean_df['lat'],  # Latitude
    c=ocean_df['bathy'], # Color based on bathymetry values
    cmap='viridis',     # Viridis colormap for continuous data
    s=5,                # Small marker size for individual points
    alpha=0.7,          # Transparency to see overlapping points
    transform=ccrs.PlateCarree(), # Important: tell Cartopy these are lat/lon coordinates
    edgecolor='none'    # No edge around markers
)

# Add a colorbar to explain the bathymetry values
fig_bathy.colorbar(scatter_bathy, label="Bathymetry (m)", orientation='vertical', pad=0.05)
ax_bathy.set_title("Geographic Distribution of Bathymetry")

# Add gridlines with labels
gl_bathy = ax_bathy.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl_bathy.top_labels = False
gl_bathy.right_labels = False

# Save the figure and close it to free memory
fig_bathy.savefig(output_bathy_map_path, dpi=300, bbox_inches='tight')
plt.close(fig_bathy)
print("Bathymetry map saved.")

# --- Plot Geographic Distribution of Chlorophyll ---
print(f"Generating map for Chlorophyll and saving to {output_chlorophyll_map_path}...")
fig_chlorophyll = plt.figure(figsize=(15, 10))
ax_chlorophyll = fig_chlorophyll.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax_chlorophyll.add_feature(cfeature.LAND, facecolor='lightgray')
ax_chlorophyll.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_chlorophyll.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax_chlorophyll.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax_chlorophyll.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray')
ax_chlorophyll.add_feature(cfeature.RIVERS, edgecolor='blue')

scatter_chlorophyll = ax_chlorophyll.scatter(
    ocean_df['lon'],
    ocean_df['lat'],
    c=ocean_df['chlorophyll'],
    cmap='viridis', # Often 'viridis', 'plasma', or 'YlGn' (Yellow-Green) are good for chlorophyll
    s=5,
    alpha=0.7,
    transform=ccrs.PlateCarree(),
    edgecolor='none'
)

fig_chlorophyll.colorbar(scatter_chlorophyll, label="Chlorophyll Level", orientation='vertical', pad=0.05)
ax_chlorophyll.set_title("Geographic Distribution of Chlorophyll Level")

gl_chlorophyll = ax_chlorophyll.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl_chlorophyll.top_labels = False
gl_chlorophyll.right_labels = False

fig_chlorophyll.savefig(output_chlorophyll_map_path, dpi=300, bbox_inches='tight')
plt.close(fig_chlorophyll)
print("Chlorophyll map saved.")

# --- Plot Geographic Distribution of Secchi Depth ---
print(f"Generating map for Secchi Depth and saving to {output_secchi_map_path}...")
fig_secchi = plt.figure(figsize=(15, 10))
ax_secchi = fig_secchi.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax_secchi.add_feature(cfeature.LAND, facecolor='lightgray')
ax_secchi.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_secchi.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax_secchi.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax_secchi.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray')
ax_secchi.add_feature(cfeature.RIVERS, edgecolor='blue')

scatter_secchi = ax_secchi.scatter(
    ocean_df['lon'],
    ocean_df['lat'],
    c=ocean_df['secchi'],
    cmap='viridis', # Can also use 'plasma', 'cividis', or 'YlGnBu' (Yellow-Green-Blue)
    s=5,
    alpha=0.7,
    transform=ccrs.PlateCarree(),
    edgecolor='none'
)

fig_secchi.colorbar(scatter_secchi, label="Secchi Depth", orientation='vertical', pad=0.05)
ax_secchi.set_title("Geographic Distribution of Secchi Depth")

gl_secchi = ax_secchi.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl_secchi.top_labels = False
gl_secchi.right_labels = False

fig_secchi.savefig(output_secchi_map_path, dpi=300, bbox_inches='tight')
plt.close(fig_secchi)
print("Secchi Depth map saved.")

print("\nAll geographic distribution maps generated and saved successfully!")