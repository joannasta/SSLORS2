

from pathlib import Path
import shutil
import os
import re
class DatasetProcessor:
    def __init__(self, img_dir, depth_dir, output_dir,img_only_dir=None,depth_only_dir=None):
        self.img_dir = Path(img_dir)
        self.depth_dir = Path(depth_dir)
        self.output_dir = Path(output_dir)
        self.img_only_dir = img_only_dir
        self.depth_only_dir = depth_only_dir

        self.all_img_files = self.collect_files(self.img_dir)
        self.all_depth_files = self.collect_files(self.depth_dir)
        self.paired_files = self.match_files()

        # Check if the output directory already exists. If it does, do NOT create it.
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
            self.copy_files = True #set flag to copy files
        else:
            self.copy_files = False #set flag not to copy files
            print(f"Output directory '{self.output_dir}' already exists. Skipping file copy.")

        if  not self.img_only_dir.exists(): 
            self.create_img_folder(self.img_only_dir)  
        else:
             print(f"Image-only directory '{self.img_only_dir}' already exists. Skipping creation.")

        if self.depth_only_dir:  # New: Handle depth_only_dir
            if not Path(self.depth_only_dir).exists():
                self.create_depth_folder(self.depth_only_dir) #New function
            else:
                print(f"Depth-only directory '{self.depth_only_dir}' already exists. Skipping creation.")



        if self.copy_files:
            self.copy_paired_files()


    def collect_files(self, directory, extension="*.tif"):
        """Collects and sorts all .tif files in the given directory."""
        return sorted(directory.glob(extension))

    def extract_index(self, filename):
        """Extracts numeric index from a filename (more robust)."""
        match = re.search(r"(\d+)\.tif$", str(filename))  # Improved regex
        if match:
            return int(match.group(1))
        return None

    def match_files(self):
        """Matches image and depth files based on extracted indices."""
        depth_dict = {}
        for depth_file in self.all_depth_files:
            depth_idx = self.extract_index(depth_file)
            if depth_idx is not None:
                depth_dict[depth_idx] = depth_file

        paired_files = []
        for img_file in self.all_img_files:
            img_idx = self.extract_index(img_file)
            if img_idx is not None and img_idx in depth_dict and img_idx == self.extract_index(depth_dict[img_idx]): # Check EXCAT match
                paired_files.append((img_file, depth_dict[img_idx]))

        return paired_files

    def copy_paired_files(self):
        """Copies matched image-depth pairs to the output directory."""
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
            print(f"Copied: {item.name}")

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
            print(f"Copied: {item.name}")

        print(f"Finished copying 'depth' files to '{destination_folder}'.")

