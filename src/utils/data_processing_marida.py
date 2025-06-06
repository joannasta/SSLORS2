from pathlib import Path
import shutil
import os
import re
import numpy as np
class DatasetProcessor:
    def __init__(self, img_dir, depth_dir, output_dir, splits_dir=None,split_type="train"):
        self.img_dir = Path(img_dir)
        self.depth_dir = Path(depth_dir)
        self.output_dir = Path(output_dir)  # This will be the base output directory
        self.splits_dir = Path(splits_dir)
        self.split_type = split_type

        for mode in ["train", "val", "test"]:  # Process for each mode
            mode_output_dir = self.output_dir / mode # Create subdirectory for each mode
            if not mode_output_dir.exists():
                mode_output_dir.mkdir(parents=True, exist_ok=True)

            self.process_files_for_mode(mode, mode_output_dir) # Process each mode separately

    def process_files_for_mode(self, mode, mode_output_dir):
        """Processes files for a specific mode."""
        if mode == 'train':
            roi_file = self.splits_dir / 'train_X.txt'
        elif mode == 'test':
            roi_file = self.splits_dir / 'test_X.txt'
        elif mode == 'val':
            roi_file = self.splits_dir / 'val_X.txt'
        else:
            raise ValueError(f"Unknown split type: {mode}")

        if not roi_file.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_file}")

        try:
            ROIs = np.genfromtxt(roi_file, dtype='str', encoding='utf-8').tolist()
            if not isinstance(ROIs, list):
                ROIs = [ROIs]
        except Exception as e:
            raise RuntimeError(f"Error reading ROI file: {roi_file}. {e}")

        paired_files = []
        for roi in ROIs:
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            img_file = self.img_dir / roi_folder / (roi_name + '.tif')
            depth_file = self.depth_dir / roi_folder / (roi_name + '_cl.tif')

            if img_file.exists() and depth_file.exists(): #Check if both files exist
                paired_files.append((img_file, depth_file))
            else:
                print(f"Warning: File not found for ROI {roi}: {img_file} or {depth_file}")

        for img_file, depth_file in paired_files:
            shutil.copy2(img_file, mode_output_dir / img_file.name)
            shutil.copy2(depth_file, mode_output_dir / depth_file.name)

        print(f"Copied {len(paired_files)} paired files to {mode_output_dir}")
