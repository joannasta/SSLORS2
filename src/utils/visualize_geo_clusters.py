import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import rasterio

path_dataset = Path("/mnt/storagecube/joanna/Hydro/")
file_paths = sorted(list(path_dataset.glob("*.tif")))

longitudes = []
latitudes = []

for file_path in file_paths:
    with rasterio.open(file_path) as src:
        row, col = src.height // 2, src.width // 2
        lon, lat = src.transform * (col, row)
        longitudes.append(lon)
        latitudes.append(lat)

geo_coords = np.array(list(zip(longitudes, latitudes)))
n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(geo_coords)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(geo_coords[:, 0], geo_coords[:, 1], c=cluster_assignments, cmap='viridis', s=5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"K-means Clustering of Geo-coordinates (K={n_clusters})")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.savefig(f"Geolocation_clusters_{n_clusters}")

unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)