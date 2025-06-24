import os
import numpy as np
import rasterio

# --- Configuration: IMPORTANT! Adjust this path to your MARIDA dataset's root ---
# This should point to the directory containing 'data', 'src', etc.
project_root = "/faststorage/joanna/marida/MARIDA" 

# Derived path for the 'data' directory, which contains 'splits' and 'patches'
data_path = project_root#os.path.join(project_root) 

# Define the problematic indices (0-indexed based on the order in val_X.txt)
# These are the indices that caused the NaN/Inf warnings in your previous logs.
# Added 287 explicitly as it also showed NaN/Inf in debug output.
problematic_indices = [170, 171, 287, 288, 299] 

# --- Script Logic ---

print(f"--- Starting MARIDA Data Inspector ---")
print(f"Project root assumed: {project_root}")
print(f"Data directory: {data_path}")
print(f"Indices to check: {problematic_indices}")

# 1. Get the list of ROI identifiers for the validation set
val_x_file = os.path.join(data_path, 'splits', 'val_X.txt')
roi_identifiers = []
try:
    with open(val_x_file, 'r') as f:
        roi_identifiers = [line.strip() for line in f if line.strip()]
    print(f"\nSuccessfully loaded {len(roi_identifiers)} ROI identifiers from {val_x_file}")
except FileNotFoundError:
    print(f"Error: The file '{val_x_file}' was not found.")
    print("Please ensure 'project_root' is set correctly and 'data/splits/val_X.txt' exists.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading '{val_x_file}': {e}")
    exit()

# 2. Iterate through problematic indices and inspect each corresponding .tif file
for idx in problematic_indices:
    print(f"\n--- Checking file for ROI index {idx} ---")

    # Check if the index is within valid bounds
    if idx < 0 or idx >= len(roi_identifiers):
        print(f"Warning: Index {idx} is out of bounds for the ROI list (length {len(roi_identifiers)}). Skipping.")
        continue

    roi = roi_identifiers[idx]
    print(f"  ROI Identifier: {roi}")

    # Construct the full path to the .tif file using the logic from MaridaDataset
    # Example roi: '11-6-18_16PCC_0'
    roi_parts = roi.split('_') # e.g., ['11-6-18', '16PCC', '0']
    
    # roi_folder: 'S2_11-6-18_16PCC'
    roi_folder = '_'.join(['S2'] + roi_parts[:-1]) 
    
    # roi_name: 'S2_11-6-18_16PCC_0'
    roi_name = '_'.join(['S2'] + roi_parts) 

    # The final .tif file path
    tif_file_path = os.path.join(data_path, 'patches', roi_folder, roi_name + '.tif')
    
    print(f"  Full .tif file path: {tif_file_path}")

    try:
        with rasterio.open(tif_file_path) as src:
            # Read all bands into a NumPy array (channels, height, width)
            img_data = src.read() 
            print(f"  Image shape: {img_data.shape} (Channels, Height, Width)")
            print(f"  Image data type: {img_data.dtype}")
            print(f"  Number of bands (channels): {src.count}")
            print(f"  CRS: {src.crs}, Transform: {src.transform}") # Useful for geospatial context

            # Check for NaNs and Infs across all bands
            nan_found = np.isnan(img_data).any()
            inf_found = np.isinf(img_data).any()

            if nan_found:
                print(f"  !!! NaNs DETECTED in this file !!!")
                nan_locations = np.argwhere(np.isnan(img_data))
                # Print a few example locations (band_index, row, col)
                print(f"    Example NaN locations (band, row, col): {nan_locations[:5]}...")
            
            if inf_found:
                print(f"  !!! Infs DETECTED in this file !!!")
                inf_locations = np.argwhere(np.isinf(img_data))
                # Print a few example locations (band_index, row, col)
                print(f"    Example Inf locations (band, row, col): {inf_locations[:5]}...")

            if not nan_found and not inf_found:
                print("  No NaNs or Infs detected in any band of this file.")

            # Specifically check the bands that your model uses (channels 1, 2, 3 in your code)
            # In a 0-indexed array like `img_data`, these correspond to indices 1, 2, 3.
            bands_used_by_model = [1, 2, 3] 
            print(f"  Checking model-relevant bands (0-indexed in array): {bands_used_by_model}")
            
            for band_idx in bands_used_by_model:
                if band_idx < img_data.shape[0]: # Ensure band index is valid
                    band_nan_found = np.isnan(img_data[band_idx,:,:]).any()
                    band_inf_found = np.isinf(img_data[band_idx,:,:]).any()
                    
                    if band_nan_found or band_inf_found:
                        print(f"    --- PROBLEM IN MODEL-RELEVANT BAND {band_idx + 1} (array index {band_idx}) ---")
                        if band_nan_found:
                            print(f"      NaNs found in band {band_idx + 1}.")
                        if band_inf_found:
                            print(f"      Infs found in band {band_idx + 1}.")
                else:
                    print(f"    Band index {band_idx} is out of range for this image (max {img_data.shape[0]-1} channels).")

    except rasterio.errors.RasterioIOError as e:
        print(f"  ERROR: Could not open file '{tif_file_path}'.")
        print(f"  Reason: {e}")
        print("  This might mean the file doesn't exist, is corrupted, or permissions are an issue.")
    except Exception as e:
        print(f"  An unexpected error occurred while processing '{tif_file_path}': {e}")

print("\n--- Data inspection complete. ---")
print("Examine the output above to identify specific files and band locations of NaNs/Infs.")
print("Once identified, you can proceed with data cleaning (e.g., imputation, masking) or investigate upstream data generation.")