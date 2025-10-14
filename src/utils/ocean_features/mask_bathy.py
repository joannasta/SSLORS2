import pandas as pd
import numpy as np

def mask_bathymetry_beyond_secchi(input_csv_path, output_csv_path):
    """Load CSV, ensure numeric types, and set bathymetry to NaN where |bathy| > secchi."""
    df = pd.read_csv(input_csv_path)

    # Strip whitespace if columns are strings
    if df['bathy'].dtype == 'object':
        df['bathy'] = df['bathy'].str.strip()

    # Convert to numeric (invalid values -> NaN)
    df['bathy'] = pd.to_numeric(df['bathy'], errors='coerce')
    df['secchi'] = pd.to_numeric(df['secchi'], errors='coerce')

    # Mask: set bathy to NaN where both are valid and |bathy| > secchi
    df['bathy'] = df.apply(
        lambda row: np.nan if pd.notna(row['bathy']) and pd.notna(row['secchi']) and abs(row['bathy']) > row['secchi'] else row['bathy'],
        axis=1
    )
    # Save
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    input_file = "train_ocean_labels_3_clusters.csv"
    output_file = "ocean_features_nans_bathy.csv"

    mask_bathymetry_beyond_secchi(input_file, output_file)

    df_masked = pd.read_csv(output_file)
    print("\nFirst 5 rows of the modified CSV:")
    print(df_masked.head())
