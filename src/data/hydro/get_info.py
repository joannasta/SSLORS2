import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import rowcol

# Replace 'your_file.tif' with the path to your .tif file
file_path = "sample.tif"

with rasterio.open(file_path) as src:
    # Basic information
    print("Number of bands:", src.count)
    print("Width, Height:", src.width, src.height)
    print("Metadata:", src.meta)
    print("Data type:", src.dtypes[0])
    print("CRS:", src.crs)
    
    # Bounding Box and Resolution
    print("Bounding Box:", src.bounds)
    print("Resolution (pixel size):", src.res)
    
    # No-data value
    print("No-data value:", src.nodata)
    
    # Per-Band Statistics
    for i in range(1, src.count + 1):
        band_data = src.read(i)
        print(f"\nBand {i} statistics:")
        print(f"  Min: {band_data.min()}")
        print(f"  Max: {band_data.max()}")
        print(f"  Mean: {band_data.mean()}")
        print(f"  Std Dev: {band_data.std()}")
        print(f"  Description: {src.descriptions[i - 1]}")
        
        # Plot histogram for each band
        plt.hist(band_data.flatten(), bins=50, alpha=0.5, label=f'Band {i}')

    # Display histograms
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of Pixel Values for Each Band')
    plt.show()

    # Example of getting pixel values at specific coordinates
    longitude, latitude = 718000, 7095000  # example coordinates
    row, col = rowcol(src.transform, longitude, latitude)
    pixel_values = src.read()[:, row, col]
    print(f"\nPixel values at ({longitude}, {latitude}):", pixel_values)
