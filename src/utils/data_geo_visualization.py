import rasterio
import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import os
from pyproj import Transformer

def extract_tiff_metadata(tiff_file):
    """Extracts coordinates and CRS from a TIFF file."""
    try:
        with rasterio.open(tiff_file) as src:
            cols, rows = src.width / 2, src.height / 2
            x, y = src.xy(rows, cols)
            crs = src.crs
            return x, y, crs
    except Exception as e:
        print(f"Error processing {tiff_file}: {e}")
        return None, None, None

def create_geolocation_scatter_plot(tiff_directory, world_shapefile_path, location="hydro"):
    """Creates a scatter plot of image geolocations with a world map background."""
    coordinates = []
    tiff_files = [os.path.join(tiff_directory, f) for f in os.listdir(tiff_directory) if f.endswith('.tif')]

    for tiff_file in tiff_files:
        x, y, crs = extract_tiff_metadata(tiff_file)
        if x is not None and y is not None and crs is not None:
            coordinates.append((x, y, crs)) #append the crs into the coordinates list.

    if coordinates:
        data = []
        for x,y,crs in coordinates:
            data.append({"longitude":x,"latitude":y, "crs":crs})
        df = pd.DataFrame(data)

        reprojected_coordinates = []
        for index, row in df.iterrows():
            source_crs = row['crs']
            transformer = Transformer.from_crs(source_crs, 'EPSG:4326', always_xy=True)
            lon, lat = transformer.transform(row['longitude'], row['latitude'])
            reprojected_coordinates.append((lon, lat))

        reprojected_df = pd.DataFrame(reprojected_coordinates, columns=['longitude', 'latitude'])

        gdf = geopandas.GeoDataFrame(reprojected_df, geometry=geopandas.points_from_xy(reprojected_df.longitude, reprojected_df.latitude))

        try:
            world = geopandas.read_file(world_shapefile_path)
            print("World Shapefile CRS:", world.crs)

            fig, ax = plt.subplots(figsize=(10, 5))
            world.plot(ax=ax, alpha=0.4, color='grey')
            gdf.plot(ax=ax, markersize=5, color='black')
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Image Locations")
            plt.show()
            plt.savefig(f'geolocation_{location}_plot.png')
        except FileNotFoundError:
            print(f"Error: World shapefile not found at {world_shapefile_path}")
        except Exception as e:
            print(f"Error plotting: {e}")
    else:
        print("No valid coordinates found in TIFF files.")

# Example usage: Replace with your actual paths
tiff_directory = '/faststorage/joanna/Hydro/raw_data'
print("os.getcwd():", os.getcwd())
world_shapefile_path = '/home/joanna/SSLORS/src/utils/world_shapefile/ne_110m_admin_0_countries.shp'
location = "hydro"

create_geolocation_scatter_plot(tiff_directory, world_shapefile_path, location=location)

# Print TIFF metadata
tiff_file_1 = '/faststorage/joanna/Hydro/raw_data/patch_20571.tif'
tiff_file_2 = '/faststorage/joanna/Hydro/raw_data/patch_116999.tif'
tiff_file_3 = '/faststorage/joanna/Hydro/raw_data/patch_210078.tif'
tiff_file_4 = '/faststorage/joanna/Hydro/raw_data/patch_21010.tif'
tiff_file_5 = '/faststorage/joanna/Hydro/raw_data/patch_20.tif'
tiff_files =[tiff_file_1,tiff_file_2,tiff_file_3,tiff_file_4,tiff_file_5]
for tiff_file in tiff_files:
    with rasterio.open(tiff_file) as src:
        print("CRS:", src.crs)
        print("tags: ", src.tags())