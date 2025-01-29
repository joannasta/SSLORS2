import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Check if all .tif files have 256x256 dimensions
def check_tif_dimensions(directory, expected_size=(256, 256)):
    incorrect_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            file_path = os.path.join(directory, filename)
            with rasterio.open(file_path) as src:
                width, height = src.width, src.height
                if (width, height) != expected_size:
                    incorrect_files.append((filename, (width, height)))

    if incorrect_files:
        print("Files with incorrect dimensions:")
        for file, size in incorrect_files:
            print(f"{file}: {size}")
    else:
        print("All .tif files have the correct dimensions.")

# Specify the path to your directory containing .tif files
check_tif_dimensions('./src/data/hydro')

# Open the .tif file
path = "./src/data/hydro/sample.tif"
with rasterio.open(path) as src:
    # General metadata
    print(f"Driver: {src.driver}")
    print(f"Width: {src.width}, Height: {src.height}")
    print(f"Coordinate Reference System (CRS): {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Affine Transform: {src.transform}")
    
    # Band-specific metadata
    for i in range(1, src.count + 1):
        band = src.read(i)
        print(f"Band {i} - Data type: {src.dtypes[i-1]}")
        if src.descriptions:
            print(f"Description: {src.descriptions[i-1]}")
    
    # NoData value
    print(f"NoData Value: {src.nodata}")
    
    # Geospatial information
    print(f"Projection: {src.crs.to_proj4()}")
    print(f"Resolution: {src.res}")

    # Read RGB bands (assumes bands 1, 2, and 3 are Red, Green, Blue respectively)
    red = src.read(3)   # Red band
    green = src.read(2) # Green band
    blue = src.read(1)  # Blue band

# Stack the bands into an RGB image
rgb_image = np.dstack((red, green, blue))
print(rgb_image)

# Plot the image
plt.imshow(rgb_image)
plt.axis('off')  # Optional: Hide the axes for a cleaner image

# Save the plot as a PNG file
plt.savefig('sample_plot.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# Close the plot to free up memory
plt.close()
