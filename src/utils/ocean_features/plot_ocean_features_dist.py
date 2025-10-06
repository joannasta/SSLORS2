import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path

#'/mnt/storagecube/joanna/ocean_features_projected.csv'
OCEAN_FEATURES_PATH = Path('/mnt/storagecube/joanna/ocean_features_combined.csv')

output_bathy_map_path = Path("geographic_bathymetry_distribution_proj.png")
output_chlorophyll_map_path = Path("geographic_chlorophyll_distribution_proj.png")
output_secchi_map_path = Path("geographic_secchi_distribution_proj.png")

# Load Ocean Features Data
ocean_df = pd.read_csv(OCEAN_FEATURES_PATH)
required_cols = ['lat', 'lon', 'bathy', 'chlorophyll', 'secchi']

# Plot Geographic Distribution of Bathymetry 
fig_bathy = plt.figure(figsize=(15, 10))
ax_bathy = fig_bathy.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) 

ax_bathy.add_feature(cfeature.LAND, facecolor='lightgray')
ax_bathy.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_bathy.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax_bathy.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax_bathy.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray')
ax_bathy.add_feature(cfeature.RIVERS, edgecolor='blue')

# Plot the bathymetry data points
scatter_bathy = ax_bathy.scatter(
    ocean_df['lon'], 
    ocean_df['lat'],  
    c=ocean_df['bathy'],
    cmap='viridis',    
    s=5,              
    alpha=0.7,      
    transform=ccrs.PlateCarree(), 
    edgecolor='none' 
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

#  Plot Geographic Distribution of Chlorophyll
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
    cmap='viridis', 
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

# Plot Geographic Distribution of Secchi Depth
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
    cmap='viridis', 
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
