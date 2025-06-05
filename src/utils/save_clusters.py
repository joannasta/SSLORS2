import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import rasterio
import csv # Added explicitly for saving to CSV

# --- Configuration ---
# Define the path to your training dataset
path_dataset = Path("/faststorage/joanna/Hydro/raw_data/train")

# Number of clusters (can be made a command-line argument or variable)
n_clusters = 100

# Path where the generated CSV will be saved
output_csv_path = Path(f"train_geo_labels{n_clusters}_projected.csv") # Example output name

# --- Data Extraction ---
# Get all tif file paths within the dataset directory
file_paths = sorted(list(path_dataset.glob("*.tif")))

# Lists to store longitude, latitude, and original file paths
longitudes = []
latitudes = []
original_file_paths = [] # Store file paths to link back to labels

# Extract longitude and latitude from each tif file
print("Extracting geo-coordinates from TIFF files...")
for file_path in file_paths:
    try:
        with rasterio.open(file_path) as src:
            # Get center coordinates of the raster
            row, col = src.height // 2, src.width // 2
            lon, lat = src.transform * (col, row)
            longitudes.append(lon)
            latitudes.append(lat)
            original_file_paths.append(str(file_path)) # Store as string for CSV
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening {file_path}: {e}. Skipping this file.")

# Convert lists to numpy arrays for clustering
geo_coords = np.array(list(zip(longitudes, latitudes)))

if len(geo_coords) == 0:
    print("No geographic coordinates found. Exiting.")
else:
    # --- K-means Clustering ---
    print(f"Performing K-means clustering with K={n_clusters}...")
    # Ensure n_clusters is not greater than the number of samples
    actual_n_clusters = min(n_clusters, len(geo_coords))
    if actual_n_clusters == 0:
        print("Not enough data points to form clusters (0 data points). Skipping clustering and CSV saving.")
    else:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10) # n_init is important for robustness
        cluster_assignments = kmeans.fit_predict(geo_coords)

        # --- Save Clusters to CSV ---
        print(f"Saving cluster assignments to {output_csv_path}...")
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(['label', 'file_dir', 'lat', 'lon']) 
            # Write data rows
            for i in range(len(original_file_paths)):
                label = cluster_assignments[i]
                file_dir = original_file_paths[i]
                lat = latitudes[i]
                lon = longitudes[i]
                writer.writerow([label, file_dir, lat, lon])
        print("CSV file saved successfully.")

        # --- Visualization ---
        print("Generating scatter plot of clusters...")
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(geo_coords[:, 0], geo_coords[:, 1], c=cluster_assignments, cmap='viridis', s=5, alpha=0.7)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"K-means Clustering of Geo-coordinates (K={actual_n_clusters})")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True)
        plt.show()
        print("Scatter plot displayed.")

        # --- Statistics ---
        unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
        print(f"\n--- Cluster Statistics (K={actual_n_clusters}) ---")
        print(f"Number of unique clusters: {len(unique_clusters)}")
        print("Cluster counts:", dict(zip(unique_clusters, counts)))
        print("-------------------------------------")

