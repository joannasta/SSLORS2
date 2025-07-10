import shutil
import os
import re

from pathlib import Path
from config import train_images, test_images

class DatasetProcessor:
    def __init__(self, img_dir, depth_dir, output_dir,img_only_dir=None,depth_only_dir=None,split_type="train"):
        self.img_dir = Path(img_dir)
        self.depth_dir = Path(depth_dir)
        self.output_dir = Path(output_dir)
        self.img_only_dir = img_only_dir
        self.depth_only_dir = depth_only_dir
        self.train_idx = train_images
        self.test_idx = test_images
        self.split_type = split_type

        # --- DEBUG PRINT: Showing loaded train/test indices ---
        print(f"\n--- DatasetProcessor Initialization Debug ---")
        print(f"Configured train_images (self.train_idx): {self.train_idx}")
        print(f"Configured test_images (self.test_idx): {self.test_idx}")
        print(f"Split type for this processor: {self.split_type}")
        print(f"Expected indices to use for matching: {self.train_idx if self.split_type == 'train' else self.test_idx}")
        print(f"--- End Initialization Debug ---\n")


        self.all_img_files = self.collect_files(self.img_dir)
        self.all_depth_files = self.collect_files(self.depth_dir)

        # --- DEBUG PRINT: Show collected file counts and samples ---
        print(f"Collected {len(self.all_img_files)} image files from {self.img_dir}")
        print(f"Sample image files (first 5): {[f.name for f in self.all_img_files[:5]]}")
        print(f"Collected {len(self.all_depth_files)} depth files from {self.depth_dir}")
        print(f"Sample depth files (first 5): {[f.name for f in self.all_depth_files[:5]]}")
        print(f"--- End File Collection Debug ---\n")

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
            if depth_only_dir:  # Create depth_only_dir AFTER copying
                self.create_depth_folder(depth_only_dir)
            else:
                print("depth_only_dir is None. Skipping creation.")

        if self.copy_files:
            self.copy_paired_files()


    def collect_files(self, directory, extension="*.tif"):
        """Collects and sorts all .tif files in the given directory."""
        return sorted(directory.glob(extension))

    def extract_index(self, filename):
        """Extracts numeric index from a filename (more robust)."""
        match = re.search(r"(\d+)\.tif$", str(filename))  # Improved regex
        if match:
            extracted_num = int(match.group(1))
            # --- DEBUG PRINT: Show extracted index ---
            # print(f"Debug: Extracted index '{extracted_num}' from filename: {filename.name}")
            return extracted_num
        # print(f"Debug: Failed to extract index from filename: {filename.name}")
        return None

    def match_files(self):
        """Matches image and depth files based on extracted indices."""
        depth_dict = {}
        indices_to_use = self.train_idx if self.split_type == 'train' else self.test_idx # Use correct indices

        print(f"\n--- Match Files Debug ---")
        print(f"Starting depth file matching. Target indices: {indices_to_use}")

        for depth_file in self.all_depth_files:
            depth_idx = self.extract_index(depth_file)
            if depth_idx is not None:
                # --- DEBUG PRINT: Show index check result ---
                is_in_config_lists = (str(depth_idx) in self.train_idx or str(depth_idx) in self.test_idx)
                print(f"  Processing depth file: {depth_file.name}, Extracted index: {depth_idx} (str: '{str(depth_idx)}')")
                print(f"    Is '{str(depth_idx)}' in train_images ({self.train_idx}) or test_images ({self.test_idx})? -> {is_in_config_lists}")

                if is_in_config_lists:
                    depth_dict[depth_idx] = depth_file
                    print(f"    SUCCESS: Added {depth_idx} to depth_dict.")
                else:
                    print(f"    SKIP: Index {depth_idx} not found in configured train/test lists.")
            # else:
                # print(f"  Skipping {depth_file.name}: No index extracted.")


        print(f"\nFinished depth file processing. Final depth_dict contains {len(depth_dict)} entries. Keys: {list(depth_dict.keys())}")


        paired_files = []
        print(f"\nStarting image file matching against depth_dict...")
        for img_file in self.all_img_files:
            img_idx = self.extract_index(img_file)
            if img_idx is not None:
                # --- DEBUG PRINT: Show image match attempt ---
                print(f"  Processing image file: {img_file.name}, Extracted index: {img_idx}")
                if img_idx in depth_dict: # Check if img_idx is a key in the depth_dict
                    # The redundant check 'img_idx == self.extract_index(depth_dict[img_idx])' is removed as it's always true if img_idx is a key
                    paired_files.append((img_file, depth_dict[img_idx]))
                    print(f"    MATCH FOUND: Paired {img_file.name} with {depth_dict[img_idx].name}.")
                else:
                    print(f"    SKIP: Image index {img_idx} not found in populated depth_dict.")
            # else:
                # print(f"  Skipping {img_file.name}: No index extracted.")

        print(f"\n--- End Match Files Debug ---")
        print(f"Total paired files found: {len(paired_files)}")
        return paired_files

    def copy_paired_files(self):
        """Copies matched image-depth pairs to the output directory."""
        # This function will still report 'Copied 0 paired files' if paired_files is empty
        for img_path, depth_path in self.paired_files:
            shutil.copy(img_path, self.output_dir / img_path.name)
            shutil.copy(depth_path, self.output_dir / depth_path.name)
        print(f"Copied {len(self.paired_files)} paired files to {self.output_dir}")

    
    def create_img_folder(self, destination_folder):
        """Creates a new folder containing only the 'img' files, sorted numerically."""
        dest_path = Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        img_files = [
            item for item in self.output_dir.iterdir()
            if item.is_file() and re.match(r"img_(\d+)\.tif$", item.name)
        ]

        sorted_img_files = sorted(img_files, key=lambda item: int(item.name.split("_")[1].split(".")[0]))

        for item in sorted_img_files:
            shutil.copy2(item, dest_path / item.name)
            # print(f"Copied: {item.name}") # Commented out to reduce verbose output if many files

        print(f"Finished copying 'img' files to '{destination_folder}'.")

    def create_depth_folder(self, destination_folder):
        """Creates a new folder containing only the 'depth' files, sorted numerically."""
        dest_path = Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        depth_files = [
            item for item in self.output_dir.iterdir()
            if item.is_file() and re.match(r"depth_(\d+)\.tif$", item.name)
        ]

        sorted_depth_files = sorted(depth_files, key=lambda item: int(item.name.split("_")[1].split(".")[0]))

        for item in sorted_depth_files:
            shutil.copy2(item, dest_path / item.name)
            # print(f"Copied: {item.name}") # Commented out to reduce verbose output if many files

        print(f"Finished copying 'depth' files to '{destination_folder}'.")