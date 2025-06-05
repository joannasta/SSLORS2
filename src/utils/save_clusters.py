    def _create_geo_labels(self):
        geo_coords = np.array([[item[2], item[1]] for item in self._load_data()])  # [lon, lat]
        kmeans = KMeans(n_clusters=self.num_geo_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(geo_coords)
        geo_to_label_map = {}
        if self.save_csv:  # Only create and save CSV if save_csv is True
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['label', 'file_dir', 'lat', 'lon'])  # Write the header row
                for i, (path, lat, lon) in enumerate(self._load_data()):
                    label = cluster_assignments[i]
                    geo_to_label_map[path] = label
                    writer.writerow([label, path, lat, lon])  # Write data to CSV
        else:
            for i, (path, lat, lon) in enumerate(self._load_data()):
                geo_to_label_map[path] = cluster_assignments[i]
        return geo_to_label_map
