import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import rasterio
import csv
import pandas as pd
from scipy.spatial import KDTree
from rasterio.warp import transform as rasterio_transform

import cartopy.crs as ccrs
import cartopy.feature as cfeature

TIF_DIRECTORY = Path('/data/joanna/Hydro')
OCEAN_FEATURES_PATH = Path('/mnt/storagecube/joanna/ocean_features_filtered.csv')

n_clusters = 10

output_csv_path = Path(f"train_ocean_labels_{n_clusters}_clusters.csv")

# Define paths for saving plots
output_3d_plot_path = Path(f"3d_ocean_features_clusters_{n_clusters}.png")
output_map_plot_path = Path(f"geographic_clusters_map_{n_clusters}.png")


MAX_MATCH_DISTANCE = 0.01

try:
    ocean_df = pd.read_csv(OCEAN_FEATURES_PATH)
    required_cols = ['lat', 'lon', 'bathy', 'chlorophyll', 'secchi']
    if not all(col in ocean_df.columns for col in required_cols):
        raise ValueError(f"Ocean features CSV missing one or more required columns: {required_cols}")
    
    ocean_coords = ocean_df[['lat', 'lon']].values
    ocean_kdtree = KDTree(ocean_coords)

except FileNotFoundError:
    print(f"Error: Ocean features CSV not found at {OCEAN_FEATURES_PATH}.")
    exit()
except Exception as e:
    print(f"Error loading or processing ocean features CSV: {e}")
    exit()

tif_file_paths = sorted(list(TIF_DIRECTORY.glob("*.tif")))

matched_file_paths = []
matched_geo_coords = []
matched_ocean_features = []

for file_path in tif_file_paths:
    try:
        with rasterio.open(file_path) as src:
            row, col = src.height // 2, src.width // 2
            native_center_lon, native_center_lat = src.transform * (col, row)

            if src.crs and src.crs.to_epsg() != 4326:
                transformed_lon, transformed_lat = rasterio_transform(
                    src.crs,
                    'EPSG:4326',
                    [native_center_lon],
                    [native_center_lat]
                )
                tif_center_lon = transformed_lon[0]
                tif_center_lat = transformed_lat[0]
            else:
                tif_center_lon = native_center_lon
                tif_center_lat = native_center_lat

            query_point = np.array([tif_center_lat, tif_center_lon])

            distance, index = ocean_kdtree.query(query_point, k=1, distance_upper_bound=MAX_MATCH_DISTANCE)

            if distance <= MAX_MATCH_DISTANCE and index != ocean_kdtree.n:
                matched_row = ocean_df.iloc[index]
                
                matched_file_paths.append(str(file_path))
                matched_geo_coords.append([tif_center_lon, tif_center_lat])
                matched_ocean_features.append([
                    matched_row['bathy'],
                    matched_row['chlorophyll'],
                    matched_row['secchi']
                ])

    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening {file_path}: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred for {file_path}: {e}.")

if not matched_file_paths:
    print("No TIFF files could be matched to ocean features. Exiting.")
    exit()

ocean_features_for_clustering = np.array(matched_ocean_features)
geo_coords_matched_np = np.array(matched_geo_coords)

actual_n_clusters = min(n_clusters, len(ocean_features_for_clustering))

if actual_n_clusters == 0:
    print("Not enough data points to form clusters. Skipping clustering and CSV saving.")
else:
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(ocean_features_for_clustering)

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'file_dir', 'lon', 'lat', 'bathy', 'chlorophyll', 'secchi']) 
        for i in range(len(matched_file_paths)):
            label = cluster_assignments[i]
            file_dir = matched_file_paths[i]
            lon = geo_coords_matched_np[i, 0]
            lat = geo_coords_matched_np[i, 1]
            bathy = ocean_features_for_clustering[i, 0]
            chlorophyll = ocean_features_for_clustering[i, 1]
            secchi = ocean_features_for_clustering[i, 2]
            writer.writerow([label, file_dir, lon, lat, bathy, chlorophyll, secchi])

    # 3D plot of ocean features
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    scatter_3d = ax_3d.scatter(
        ocean_features_for_clustering[:, 0],
        ocean_features_for_clustering[:, 1],
        ocean_features_for_clustering[:, 2],
        c=cluster_assignments,
        cmap='viridis',
        s=50,
        alpha=0.7
    )

    ax_3d.set_xlabel("Bathymetry")
    ax_3d.set_ylabel("Chlorophyll Level")
    ax_3d.set_zlabel("Secchi Depth")
    ax_3d.set_title(f"K-means Clustering of Ocean Features (K={actual_n_clusters})")
    fig_3d.colorbar(scatter_3d, label="Cluster ID")
    ax_3d.grid(True)
    fig_3d.savefig(output_3d_plot_path, dpi=300, bbox_inches='tight') # Save the 3D plot
    plt.close(fig_3d) # Close the figure to free up memory

    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    print(f"Number of unique clusters: {len(unique_clusters)}")
    print("Cluster counts:", dict(zip(unique_clusters, counts)))

    for i, centroid in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i}: Bathy={centroid[0]:.2f}, Chlorophyll={centroid[1]:.4f}, Secchi={centroid[2]:.2f}")

    # Map plot of clusters
    fig_map = plt.figure(figsize=(15, 10))
    ax_map = plt.axes(projection=ccrs.PlateCarree())
    
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax_map.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray')
    ax_map.add_feature(cfeature.RIVERS, edgecolor='blue')

    scatter_map = ax_map.scatter(
        geo_coords_matched_np[:, 0],
        geo_coords_matched_np[:, 1],
        c=cluster_assignments,
        cmap='viridis',
        s=50,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        edgecolor='k',
        linewidth=0.5
    )

    fig_map.colorbar(scatter_map, label="Cluster ID", orientation='vertical', pad=0.05)
    ax_map.set_title(f"K-means Clusters on Earth Map (K={actual_n_clusters})")
    
    gl = ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    fig_map.savefig(output_map_plot_path, dpi=300, bbox_inches='tight') # Save the map plot
    plt.close(fig_map) # Close the figure to free up memory