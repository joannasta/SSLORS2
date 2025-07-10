import os
import numpy as np
import rasterio
from tqdm import tqdm # Import tqdm for progress bars

# --- Configuration: IMPORTANT! Adjust this path to your MARIDA dataset's root ---
# This should point to the directory containing 'data', 'src', etc.
project_root = "/data/joanna/MARIDA" 

# Derived path for the 'data' directory, which contains 'splits' and 'patches'
# Keeping your provided data_path = project_root
data_path = project_root 

# Note: The 'problematic_indices' list is no longer used, as we will check all files.
# It is kept here as a comment to acknowledge its original presence in your logic.
# problematic_indices = [170, 171, 287, 288, 299] 

# --- Script Logic ---

print(f"--- Starting MARIDA Data Inspector ---")
print(f"Project root assumed: {project_root}")
print(f"Data directory: {data_path}")
# print(f"Indices to check: {problematic_indices}") # Removed, as we check all

# List of split files to check
split_filenames = ['train_X.txt', 'val_X.txt', 'test_X.txt']

# Loop through each split file
for split_file_name in split_filenames:
    current_split_path = os.path.join(data_path, 'splits', split_file_name)
    split_name = split_file_name.replace('_X.txt', '').upper() # TRAIN, VAL, TEST

    print(f"\n\n=======================================================")
    print(f"       Checking all files in {split_name} set")
    print(f"       Loading from: {current_split_path}")
    print(f"=======================================================")

    roi_identifiers = []
    try:
        with open(current_split_path, 'r') as f:
            roi_identifiers = [line.strip() for line in f if line.strip()]
        print(f"\nSuccessfully loaded {len(roi_identifiers)} ROI identifiers from {current_split_path}")
    except FileNotFoundError:
        print(f"Error: The file '{current_split_path}' was not found. Skipping this split.")
        continue # Skip to the next split file
    except Exception as e:
        print(f"An unexpected error occurred while reading '{current_split_path}': {e}. Skipping this split.")
        continue # Skip to the next split file

    # Iterate through ALL ROI identifiers in the current split
    # Using tqdm for a progress bar
    for roi in tqdm(roi_identifiers, desc=f"Processing {split_name} files"):
        # The 'idx' from your original loop is implicit here as we iterate over 'roi_identifiers' directly.
        # If you needed the numerical index, 'enumerate(roi_identifiers)' would provide it.
        # print(f"\n--- Checking file for ROI index (implicit) ---") # Removed detailed print for every file's index

        # ROI Identifier is now 'roi' directly
        # print(f"  ROI Identifier: {roi}") # Removed for cleaner tqdm output, but 'roi' is available

        # Construct the full path to the .tif file using the logic from MaridaDataset
        # Example roi: '11-6-18_16PCC_0'
        roi_parts = roi.split('_') 
        
        # roi_folder: 'S2_11-6-18_16PCC'
        roi_folder = '_'.join(['S2'] + roi_parts[:-1]) 
        
        # roi_name: 'S2_11-6-18_16PCC_0'
        roi_name = '_'.join(['S2'] + roi_parts) 

        # The final .tif file path
        tif_file_path = os.path.join(data_path, 'patches', roi_folder, roi_name + '.tif')
        
        # print(f"  Full .tif file path: {tif_file_path}") # Removed for cleaner tqdm output

        try:
            with rasterio.open(tif_file_path) as src:
                # Read all bands into a NumPy array (channels, height, width)
                img_data = src.read() 
                # print(f"  Image shape: {img_data.shape} (Channels, Height, Width)") # Removed for cleaner tqdm output
                # print(f"  Image data type: {img_data.dtype}")
                # print(f"  Number of bands (channels): {src.count}")
                # print(f"  CRS: {src.crs}, Transform: {src.transform}") 

                # Check for NaNs and Infs across all bands
                nan_found = np.isnan(img_data).any()
                inf_found = np.isinf(img_data).any()

                if nan_found:
                    print(f"\n  !!! NaNs DETECTED in {roi} (File: {os.path.basename(tif_file_path)}) !!!")
                    nan_locations = np.argwhere(np.isnan(img_data))
                    print(f"    Example NaN locations (band, row, col): {nan_locations[:5]}...")
                
                if inf_found:
                    print(f"\n  !!! Infs DETECTED in {roi} (File: {os.path.basename(tif_file_path)}) !!!")
                    inf_locations = np.argwhere(np.isinf(img_data))
                    print(f"    Example Inf locations (band, row, col): {inf_locations[:5]}...")

                if not nan_found and not inf_found:
                    pass # We only print problematic files for clarity in a full scan
                    # print(f"  {roi}: No NaNs or Infs detected in any band of this file.")

                # Specifically check the bands that your model uses (channels 1, 2, 3 in your code)
                bands_used_by_model = [1, 2, 3] 
                # print(f"  Checking model-relevant bands (0-indexed in array): {bands_used_by_model}") # Removed for cleaner tqdm output
                
                for band_idx in bands_used_by_model:
                    if band_idx < img_data.shape[0]: 
                        band_nan_found = np.isnan(img_data[band_idx,:,:]).any()
                        band_inf_found = np.isinf(img_data[band_idx,:,:]).any()
                        
                        if band_nan_found or band_inf_found:
                            print(f"    --- PROBLEM IN MODEL-RELEVANT BAND {band_idx + 1} (array index {band_idx}) for {roi} ---")
                            if band_nan_found:
                                print(f"      NaNs found in band {band_idx + 1}.")
                            if band_inf_found:
                                print(f"      Infs found in band {band_idx + 1}.")
                    else:
                        print(f"    Band index {band_idx} is out of range for {roi} (max {img_data.shape[0]-1} channels).")

        except rasterio.errors.RasterioIOError as e:
            print(f"\n  ERROR: Could not open file '{tif_file_path}' for ROI {roi}.")
            print(f"  Reason: {e}")
            print("  This might mean the file doesn't exist, is corrupted, or permissions are an issue.")
        except Exception as e:
            print(f"\n  An unexpected error occurred while processing '{tif_file_path}' for ROI {roi}: {e}")

print("\n\n--- Comprehensive Data Inspection Complete. ---")
print("Examine the output above to identify specific files and band locations of NaNs/Infs.")
print("The tqdm progress bars will show you the overall progress for each split.")
print("Once identified, the imputation logic in your MaridaDataset should handle these during runtime.")