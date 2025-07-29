import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import rasterio
import csv
import pandas as pd

base_tif_directory = Path("/data/joanna/Hydro/")
n_clusters = 10
output_csv_path = Path(f"train_ocean_labels{n_clusters}_projected.csv")

input_csv = Path("/home/joanna/SSLORS2/src/utils/train_ocean_labels_3_clusters_correct.csv")

try:
    df = pd.read_csv(input_csv)
    if 'file_dir' not in df.columns:
        raise ValueError("CSV must contain a 'file_dir' column.")
    file_paths_to_process = [Path(p) for p in df['file_dir'].tolist()]
    print(f"Successfully loaded {len(file_paths_to_process)} file paths from '{input_csv}'.")
except FileNotFoundError:
    print(f"Error: Input CSV file not found at {input_csv}. Exiting.")
    exit()
except ValueError as e:
    print(f"Error with CSV format: {e}. Exiting.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the input CSV: {e}. Exiting.")
    exit()

file_paths = file_paths_to_process

longitudes = []
latitudes = []
original_file_paths = []

print("Extracting geo-coordinates from TIFF files...")
for file_path in file_paths:
    if not file_path.exists():
        print(f"Warning: File listed in CSV not found: {file_path}. Skipping.")
        continue

    try:
        with rasterio.open(file_path) as src:
            row, col = src.height // 2, src.width // 2
            lon, lat = src.transform * (col, row)
            longitudes.append(lon)
            latitudes.append(lat)
            original_file_paths.append(str(file_path))
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening {file_path}: {e}. Skipping this file.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}. Skipping this file.")

geo_coords = np.array(list(zip(longitudes, latitudes)))

if len(geo_coords) == 0:
    print("No geographic coordinates found from the specified CSV files. Exiting.")
else:
    print(f"Performing K-means clustering with K={n_clusters}...")
    actual_n_clusters = min(n_clusters, len(geo_coords))
    if actual_n_clusters == 0:
        print("Not enough data points to form clusters (0 data points). Skipping clustering and CSV saving.")
    else:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(geo_coords)

        print(f"Saving cluster assignments to {output_csv_path}...")
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['label', 'file_dir', 'lat', 'lon'])
            for i in range(len(original_file_paths)):
                label = cluster_assignments[i]
                file_dir = original_file_paths[i]
                lat = latitudes[i]
                lon = longitudes[i]
                writer.writerow([label, file_dir, lat, lon])
        print("CSV file saved successfully.")

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

        unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
        print(f"\n--- Cluster Statistics (K={actual_n_clusters}) ---")
        print(f"Number of unique clusters: {len(unique_clusters)}")
        print("Cluster counts:", dict(zip(unique_clusters, counts)))
        print("-------------------------------------")