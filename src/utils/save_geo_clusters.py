import matplotlib.pyplot as plt
import numpy as np
import rasterio
import csv
import pandas as pd

from sklearn.cluster import KMeans
from pathlib import Path

base_tif_directory = Path("/data/joanna/Hydro/")
input_csv = Path("/home/joanna/SSLORS2/src/utils/ocean_features/train_ocean_labels_3_clusters.csv")
n_clusters = 10
output_csv_path = Path(f"train_goe_labels{n_clusters}_matched.csv")


df = pd.read_csv(input_csv)
file_paths_to_process = [Path(p) for p in df['file_dir'].tolist()]

file_paths = file_paths_to_process

longitudes = []
latitudes = []
original_file_paths = []

#Extracting geo-coordinates from TIFF files
for file_path in file_paths:
    with rasterio.open(file_path) as src:
        row, col = src.height // 2, src.width // 2
        lon, lat = src.transform * (col, row)
        longitudes.append(lon)
        latitudes.append(lat)
        original_file_paths.append(str(file_path))

geo_coords = np.array(list(zip(longitudes, latitudes)))
actual_n_clusters = min(n_clusters, len(geo_coords))

kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(geo_coords)

# Saving cluster assignments 
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['label', 'file_dir', 'lat', 'lon'])
    
    for i in range(len(original_file_paths)):
        label = cluster_assignments[i]
        file_dir = original_file_paths[i]
        lat = latitudes[i]
        lon = longitudes[i]
        writer.writerow([label, file_dir, lat, lon])

# Generating scatter plot of clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(geo_coords[:, 0], geo_coords[:, 1], c=cluster_assignments, cmap='viridis', s=5, alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"K-means Clustering of Geo-coordinates (K={actual_n_clusters})")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.show()

unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)

print(f"\n--- Cluster Statistics (K={actual_n_clusters}) ---")
print(f"Number of unique clusters: {len(unique_clusters)}")
print("Cluster counts:", dict(zip(unique_clusters, counts)))
print("-------------------------------------")