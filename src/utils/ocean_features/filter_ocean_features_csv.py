import pandas as pd
from pathlib import Path
import argparse

def filter_csv_by_bbox(
    input_csv_path: Path,
    output_csv_path: Path,
    min_lat: float, max_lat: float,
    min_lon: float, max_lon: float,
    chunk_size: int = 100000 # Adjust chunk size based on your RAM
):
    """
    Reads a large CSV in chunks, filters rows by a geographic bounding box,
    and writes the filtered data to a new CSV.
    Assumes the CSV has 'lat' and 'lon' columns.
    """
    print(f"Filtering '{input_csv_path}' to '{output_csv_path}' with bbox:")
    print(f"  Lat: [{min_lat}, {max_lat}], Lon: [{min_lon}, {max_lon}]")

    is_first_chunk = True
    total_rows_read = 0
    total_rows_written = 0

    # Read the CSV in chunks
    for i, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunk_size)):
        total_rows_read += len(chunk)
        print(f"  Processing chunk {i+1}. Rows read so far: {total_rows_read}")

        # Filter the chunk by the bounding box
        filtered_chunk = chunk[
            (chunk['lat'] >= min_lat) & (chunk['lat'] <= max_lat) &
            (chunk['lon'] >= min_lon) & (chunk['lon'] <= max_lon)
        ]
        total_rows_written += len(filtered_chunk)

        # Write the filtered chunk to the output CSV
        # Use header=True only for the first chunk
        # Use mode='w' for the first chunk, 'a' for subsequent chunks
        mode = 'w' if is_first_chunk else 'a'
        header = is_first_chunk
        filtered_chunk.to_csv(output_csv_path, mode=mode, header=header, index=False)
        is_first_chunk = False

    if total_rows_written == 0:
        print("WARNING: No rows matched the bounding box. The output CSV might be empty.")

    print(f"\nFiltering complete. Total rows read: {total_rows_read}, Total rows written: {total_rows_written}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a large CSV by geographic bounding box.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the large input CSV file (e.g., /mnt/storagecube/joanna/ocean_features_projected.csv).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path for the filtered output CSV file (e.g., /mnt/storagecube/joanna/ocean_features_filtered.csv).")
    parser.add_argument("--min_lat", type=float, required=True)
    parser.add_argument("--max_lat", type=float, required=True)
    parser.add_argument("--min_lon", type=float, required=True)
    parser.add_argument("--max_lon", type=float, required=True)
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Number of rows to read at a time.")
    args = parser.parse_args()

    try:
        filter_csv_by_bbox(
            Path(args.input_csv),
            Path(args.output_csv),
            args.min_lat, args.max_lat,
            args.min_lon, args.max_lon,
            args.chunk_size
        )
    except Exception as e:
        print(f"An error occurred during filtering: {e}")