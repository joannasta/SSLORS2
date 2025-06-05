import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import rasterio

# Define the path to your training dataset
path_dataset = Path("/faststorage/joanna/Hydro/raw_data/train")
# Get all tif file paths
file_paths = sorted(list(path_dataset.glob("*.tif")))

# Lists to store longitudes and latitudes
longitudes = []
latitudes = []

# Extract longitude and latitude from each tif file
for file_path in file_paths:
    try:
        with rasterio.open(file_path) as src:
            row, col = src.height // 2, src.width // 2
            lon, lat = src.transform * (col, row)
            longitudes.append(lon)
            latitudes.append(lat)
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening {file_path}: {e}")

# Convert lists to numpy arrays
geo_coords = np.array(list(zip(longitudes, latitudes)))

# Number of clusters (same as your num_geo_clusters)
n_clusters = 100

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(geo_coords)

# Create a scatter plot of the coordinates with cluster assignments
plt.figure(figsize=(10, 8))
scatter = plt.scatter(geo_coords[:, 0], geo_coords[:, 1], c=cluster_assignments, cmap='viridis', s=5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"K-means Clustering of Geo-coordinates (K={n_clusters})")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.show()

# You can also print some statistics about the clusters
unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
print(f"Number of unique clusters: {len(unique_clusters)}")
print("Cluster counts:", dict(zip(unique_clusters, counts)))