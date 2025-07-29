import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import rasterio
import csv
import pandas as pd
from scipy.spatial import KDTree
from rasterio.warp import transform as rasterio_transform
from sklearn.metrics import silhouette_score 

import cartopy.crs as ccrs
import cartopy.feature as cfeature


TIF_DIRECTORY = Path('/mnt/storagecube/joanna/Hydro')
#'/mnt/storagecube/joanna/ocean_features_projected.csv'
OCEAN_FEATURES_PATH = Path("/home/joanna/SSLORS2/src/utils/ocean_features_capped_bathy.csv")

n_clusters = 3 

output_csv_path = Path(f"train_ocean_labels_{n_clusters}_clusters_capped.csv")
output_3d_plot_path = Path(f"3d_ocean_features_clusters_{n_clusters}_capped.png")
output_map_plot_path = Path(f"geographic_clusters_map_{n_clusters}_capped.png")
output_elbow_plot_path = Path("elbow_method_ocean_clusters_capped.png") 
output_silhouette_plot_path = Path("silhouette_score_ocean_clusters.png") 


MAX_MATCH_DISTANCE = 0.01

# --- Load Ocean Features Data ---

ocean_df = pd.read_csv(OCEAN_FEATURES_PATH)
required_cols = ['lat', 'lon', 'bathy', 'chlorophyll', 'secchi']
if not all(col in ocean_df.columns for col in required_cols):
    raise ValueError(f"Ocean features CSV missing one or more required columns: {required_cols}")
    
ocean_coords = ocean_df[['lat', 'lon']].values
ocean_kdtree = KDTree(ocean_coords)
print(f"Successfully loaded {len(ocean_df)} ocean features points.")

# --- Match TIFF Files to Ocean Features ---
tif_file_paths = sorted(list(TIF_DIRECTORY.glob("*.tif")))
print(f"Found {len(tif_file_paths)} TIFF files in {TIF_DIRECTORY}.")

matched_file_paths = []
matched_geo_coords = []
matched_ocean_features = [] 

for file_path in tif_file_paths:

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
                
            bathy_val = matched_row['bathy']
            chlorophyll_val = matched_row['chlorophyll']
            secchi_val = matched_row['secchi']

            if pd.notna(bathy_val) and pd.notna(chlorophyll_val) and pd.notna(secchi_val):
                matched_file_paths.append(str(file_path))
                matched_geo_coords.append([tif_center_lon, tif_center_lat])
                matched_ocean_features.append([
                    bathy_val,
                    chlorophyll_val,
                    secchi_val
                ])
            else:
                print(f"Skipping {file_path} due to NaN values in ocean features after masking.")


ocean_features_for_clustering = np.array(matched_ocean_features)
geo_coords_matched_np = np.array(matched_geo_coords)

# --- Determine Optimal Number of Clusters (Elbow Method & Silhouette Score) ---
print("\n--- Determining Optimal Number of Clusters ---")

# Elbow Method
wcss = []
k_range = range(1, min(len(ocean_features_for_clustering), 20)) 

if len(k_range) > 0:
    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit(ocean_features_for_clustering)
        wcss.append(kmeans_model.inertia_) 

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_range) 
    plt.grid(True)
    plt.savefig(output_elbow_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Elbow method plot saved to '{output_elbow_plot_path}'. Review this plot for an 'elbow' point.")
else:
    print("Not enough data points to perform Elbow Method analysis.")

print("\n--- Performing K-means Clustering and Plotting with n_clusters = 3---")
# Adjust actual_n_clusters based on the available data points, then use the chosen n_clusters value
actual_n_clusters = min(n_clusters, len(ocean_features_for_clustering))


kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(ocean_features_for_clustering)

# --- Save Clustered Data to CSV ---
print(f"Saving cluster assignments to {output_csv_path}...")
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
print("Cluster assignments CSV saved.")

# --- Print Cluster Information ---
unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
print(f"Number of unique clusters formed: {len(unique_clusters)}")
print("Cluster counts:", dict(zip(unique_clusters, counts)))

for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: Bathy={centroid[0]:.2f}, Chlorophyll={centroid[1]:.4f}, Secchi={centroid[2]:.2f}")

# --- 3D Plot of Ocean Features Clusters ---
print(f"Generating 3D plot and saving to {output_3d_plot_path}...")
fig_3d = plt.figure(figsize=(12, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')

scatter_3d = ax_3d.scatter(
    ocean_features_for_clustering[:, 0],
    ocean_features_for_clustering[:, 1],
    ocean_features_for_clustering[:, 2],
    c=cluster_assignments,
    cmap='tab10', 
    s=50,
    alpha=0.7
)

ax_3d.set_xlabel("Bathymetry")
ax_3d.set_ylabel("Chlorophyll Level")
ax_3d.set_zlabel("Secchi Depth")
ax_3d.set_title(f"K-means Clustering of Ocean Features (K={actual_n_clusters})")
    
unique_cluster_ids = np.unique(cluster_assignments)
bounds = np.arange(len(unique_cluster_ids) + 1) - 0.5 
cbar_3d = fig_3d.colorbar(scatter_3d, label="Cluster ID", 
                          ticks=unique_cluster_ids, boundaries=bounds)
cbar_3d.set_ticklabels([str(int(i)) for i in unique_cluster_ids])

ax_3d.grid(True)
fig_3d.savefig(output_3d_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig_3d)
print("3D plot saved.")
        
# --- Map Plot of Clusters ---
print(f"Generating geographic map of clusters and saving to {output_map_plot_path}...")
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
    cmap='tab10', 
    s=50,
    alpha=0.7,
    transform=ccrs.PlateCarree(),
    edgecolor='k',
    linewidth=0.5
)

unique_cluster_ids = np.unique(cluster_assignments)    
bounds = np.arange(len(unique_cluster_ids) + 1) - 0.5 
norm = plt.Normalize(vmin=unique_cluster_ids.min() - 0.5, vmax=unique_cluster_ids.max() + 0.5)
    
# Create the colorbar using the scatter plot and the discrete norm
cbar = fig_map.colorbar(scatter_map, label="Cluster ID", orientation='vertical', pad=0.05, 
                        ticks=unique_cluster_ids, boundaries=bounds)
cbar.set_ticklabels([str(int(i)) for i in unique_cluster_ids])

ax_map.set_title(f"K-means Clusters on Earth Map (K={actual_n_clusters})")
    
gl = ax_map.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

fig_map.savefig(output_map_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig_map)
print("Geographic cluster map saved.")