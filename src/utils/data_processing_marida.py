
import shutil
import os
import re

from pathlib import Path
from config import train_images, test_images
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self,mode="train",ROIs=None ,output_dir=None,img_only_dir=None,depth_only_dir=None,split_type="train"):
        self.mode=mode
        self.ROIs = ROIs
        self.output_dir = Path(output_dir)
        self.img_only_dir = Path(img_only_dir)
        self.target_only_dir = Path(depth_only_dir)
        self.split_type = split_type

        self.all_img_files = self.collect_files(self.img_dir)
        self.all_target_files = self.collect_files(self.depth_dir)
        self.paired_files = self.match_files()

        # Check if the output directory already exists. If it does, do NOT create it.
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.copy_files = True
        else:
            self.copy_files = False
            print(f"Output directory '{self.output_dir}' already exists. Skipping file copy.")

        if self.copy_files:  # Only copy and create folders if copy_files is True
            self.copy_paired_files()

            if img_only_dir:  # Create img_only_dir AFTER copying
                self.create_img_folder(img_only_dir)
            else:
                print("img_only_dir is None. Skipping creation.")

            if self.target_only_dir:  # Create depth_only_dir AFTER copying
                self.create_target_folder(self.target_only_dir)
            else:
                print("target_only_dir is None. Skipping creation.")

        if self.copy_files:
            self.copy_paired_files()


    def collect_files(self, directory, extension="*.tif"):
        """Collects and sorts all .tif files in the given directory."""
        return sorted(directory.glob(extension))


    def match_files(self):
        path = '/data/joanna/MARIDA'
        for roi in tqdm(self.ROIs, desc='Load ' + self.mode + ' set to memory'):
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            roi_file = os.path.join(path, 'patches', roi_folder, roi_name + '.tif')
            roi_file_cl = os.path.join(path, 'patches', roi_folder,roi_name + '_cl.tif') # Get Class Mask
            



    def copy_paired_files(self):
        """Copies matched image-depth pairs to the output directory."""
        for img_path, depth_path in self.paired_files:
            shutil.copy(img_path, self.output_dir / img_path.name)
            shutil.copy(depth_path, self.output_dir / depth_path.name)
        print(f"Copied {len(self.paired_files)} paired files to {self.output_dir}")

    
    def create_img_folder(self, destination_folder):
        """Creates a new folder containing only the 'img' files."""
        dest_path = Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        for item in self.output_dir.iterdir():  # Iterate through all files in output dir
            if item.is_file() and re.match(r"S2_.+\.tif$", item.name) and not re.search(r"_cl\.tif$", item.name): # Match S2_..._.tif but not _cl.tif
                shutil.copy2(item, dest_path / item.name)
                print(f"Copied: {item.name}")

        print(f"Finished copying 'img' files to '{destination_folder}'.")

    def create_target_folder(self, destination_folder):
        """Creates a new folder containing only the 'depth' files."""
        dest_path = Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        for item in self.output_dir.iterdir():  # Iterate through all files in output dir
            if item.is_file() and re.match(r"S2_.+_cl\.tif$", item.name):  # Match S2_..._cl.tif
                shutil.copy2(item, dest_path / item.name)
                print(f"Copied: {item.name}")

        print(f"Finished copying target files to '{destination_folder}'.")