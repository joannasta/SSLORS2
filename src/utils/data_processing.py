# src/utils/data_processing.py

from pathlib import Path
import shutil
import os
import re
# Removed train_images, test_images import as they are passed now
# from config import train_images, test_images

class DatasetProcessor:
    def __init__(self, img_dir, depth_dir, output_dir, img_only_dir=None, depth_only_dir=None,
                 image_ids_to_process=None): # Changed from split_type
        self.img_dir = Path(img_dir)
        self.depth_dir = Path(depth_dir)
        self.output_dir = Path(output_dir)
        self.img_only_dir = img_only_dir
        self.depth_only_dir = depth_only_dir
        
        if image_ids_to_process is None:
            # If no specific IDs are given, assume all should be processed (e.g., for pre-processing raw data)
            self.image_ids_to_process = None 
            print("DatasetProcessor: No specific image IDs provided. Will process all found files.")
        else:
            self.image_ids_to_process = set(image_ids_to_process)
            print(f"DatasetProcessor: Initializing to process {len(self.image_ids_to_process)} specific image IDs.")


        self.all_img_files = self.collect_files(self.img_dir)
        self.all_depth_files = self.collect_files(self.depth_dir)
        
        self.paired_files = self.match_files()

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.copy_files = True
        else:
            self.copy_files = False
            print(f"Output directory '{self.output_dir}' already exists. Skipping initial file copy.")

        # Ensure processed_img and processed_depth directories are managed
        if self.img_only_dir:
            if not self.img_only_dir.exists():
                self.create_img_folder(self.img_only_dir)
            else:
                 print(f"Image-only directory '{self.img_only_dir}' already exists. Re-populating for current IDs.")
                 self.create_img_folder(self.img_only_dir) # Always ensure it's up-to-date with current filtered files


        if self.depth_only_dir:
            if not Path(self.depth_only_dir).exists():
                self.create_depth_folder(self.depth_only_dir)
            else:
                print(f"Depth-only directory '{self.depth_only_dir}' already exists. Re-populating for current IDs.")
                self.create_depth_folder(self.depth_only_dir)

        if self.copy_files:
            self.copy_paired_files()
        else:
            print(f"Files were not copied to {self.output_dir} (it existed). Ensure relevant 'img_only'/'depth_only' directories are correctly populated.")


    def collect_files(self, directory, extension="*.tif"):
        """Collects and sorts all .tif files in the given directory."""
        return sorted(directory.glob(extension))

    def extract_index(self, filename):
        """Extracts numeric index (as string) from a filename."""
        match = re.search(r"(\d+)\.tif$", str(filename))
        if match:
            return match.group(1) # Return as string to match IDs from config
        return None

    def match_files(self):
        """
        Matches image and depth files based on extracted indices,
        and filters them using self.image_ids_to_process if provided.
        """
        depth_dict = {}
        for depth_file in self.all_depth_files:
            depth_idx = self.extract_index(depth_file)
            if depth_idx is not None:
                depth_dict[depth_idx] = depth_file

        paired_files = []
        for img_file in self.all_img_files:
            img_idx = self.extract_index(img_file)
            
            # Filtering logic: if image_ids_to_process is set, only include files in that set
            if self.image_ids_to_process is not None and img_idx not in self.image_ids_to_process:
                continue # Skip this file if its ID is not in the desired set

            if img_idx is not None and img_idx in depth_dict and img_idx == self.extract_index(depth_dict[img_idx]):
                paired_files.append((img_file, depth_dict[img_idx]))

        print(f"DatasetProcessor matched {len(paired_files)} files based on provided IDs (or all files if no IDs provided).")
        return paired_files

    def copy_paired_files(self):
        """Copies matched image-depth pairs to the output directory."""
        if not self.paired_files:
            print("No paired files to copy after filtering.")
            return

        for img_path, depth_path in self.paired_files:
            shutil.copy(img_path, self.output_dir / img_path.name)
            shutil.copy(depth_path, self.output_dir / depth_path.name)
        print(f"Copied {len(self.paired_files)} paired files to {self.output_dir}.")

    def create_img_folder(self, destination_folder):
        """Creates or repopulates a folder containing only the 'img' files for the current set of IDs."""
        dest_path = Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files to ensure it only contains the current split's files
        for f in dest_path.iterdir():
            if f.is_file():
                os.remove(f)
        
        copied_count = 0
        for img_path, _ in self.paired_files: # Iterate through already filtered paired_files from self.match_files
            if re.match(r"img_(\d+)\.tif$", img_path.name):
                shutil.copy2(img_path, dest_path / img_path.name)
                copied_count += 1

        print(f"Finished copying {copied_count} 'img' files to '{destination_folder}'.")

    def create_depth_folder(self, destination_folder):
        """Creates or repopulates a folder containing only the 'depth' files for the current set of IDs."""
        dest_path = Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        # Clear existing files
        for f in dest_path.iterdir():
            if f.is_file():
                os.remove(f)

        copied_count = 0
        for _, depth_path in self.paired_files: # Iterate through already filtered paired_files
            if re.match(r"depth_(\d+)\.tif$", depth_path.name):
                shutil.copy2(depth_path, dest_path / depth_path.name)
                copied_count += 1

        print(f"Finished copying {copied_count} 'depth' files to '{destination_folder}'.")