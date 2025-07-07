import pyproj
from typing import Tuple

def reproject_bounding_box(
    min_x: float, max_x: float, min_y: float, max_y: float,
    source_crs: str, target_crs: str
) -> Tuple[float, float, float, float]:
    """
    Reprojects the corners of a bounding box from a source CRS to a target CRS.
    Assumes min/max_x are longitudes/eastings and min/max_y are latitudes/northings.

    Returns: (min_lat_reproj, max_lat_reproj, min_lon_reproj, max_lon_reproj) in target CRS.
    """
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Corners of the source bounding box (x, y)
    corners_xy = [
        (min_x, min_y), # Bottom-left
        (max_x, min_y), # Bottom-right
        (min_x, max_y), # Top-left
        (max_x, max_y)  # Top-right
    ]

    reprojected_lons = []
    reprojected_lats = []

    for x, y in corners_xy:
        lon_reproj, lat_reproj = transformer.transform(x, y)
        reprojected_lons.append(lon_reproj)
        reprojected_lats.append(lat_reproj)

    return (
        min(reprojected_lats),
        max(reprojected_lats),
        min(reprojected_lons),
        max(reprojected_lons)
    )

if __name__ == "__main__":
    # These are the values you got from get_tif_bbox.py (UTM coordinates)
    # REPLACE WITH YOUR EXACT VALUES IF THEY DIFFER SLIGHTLY
    tif_min_lat_utm = -9515.0
    tif_max_lat_utm = 10000005.0
    tif_min_lon_utm = 101215.0
    tif_max_lon_utm = 909745.0

    source_crs_tif = "EPSG:32609" # WGS 84 / UTM zone 9N (your TIFs)
    target_crs_csv = "EPSG:4326"  # WGS 84 Geographic (your CSV)

    print(f"Reprojecting TIF bounding box from {source_crs_tif} to {target_crs_csv}...")

    try:
        min_lat_deg, max_lat_deg, min_lon_deg, max_lon_deg = reproject_bounding_box(
            tif_min_lon_utm, tif_max_lon_utm, tif_min_lat_utm, tif_max_lat_utm,
            source_crs_tif, target_crs_csv
        )

        print(f"\nReprojected Bounding Box for Filtering (in {target_crs_csv}):")
        print(f"  Min Latitude (deg): {min_lat_deg}")
        print(f"  Max Latitude (deg): {max_lat_deg}")
        print(f"  Min Longitude (deg): {min_lon_deg}")
        print(f"  Max Longitude (deg): {max_lon_deg}")
        print("\nUse these values for --min_lat, --max_lat, --min_lon, --max_lon in filter_ocean_features_csv.py.")

    except Exception as e:
        print(f"An error occurred during reprojection: {e}")