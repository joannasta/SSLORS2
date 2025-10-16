# src/utils/visualize_augmentations.py
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from torchvision import transforms
from torchvision.transforms import functional as F  # deterministic ops on tensors
from src.data.hydro.moco.hydro_dataloader_moco import HydroMoCoDataModule
from src.utils.two_crop_transformations import TwoCropsTransform
from config import get_Hydro_means_and_stds


def prepare_image(image: np.ndarray) -> np.ndarray:
    # clip to 99th percentile and scale to 0-255
    image = np.clip(image, 0, np.percentile(image, 99))
    maxv = image.max()
    if maxv == 0:
        return image.astype('uint8')
    return (image / maxv * 255).astype('uint8')


def tensor_to_numpy_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    # de-normalize channels 1..3 and convert BGR->RGB
    img = image_tensor.clone().detach().cpu()
    means, stds = get_Hydro_means_and_stds()
    img = img * stds[1:4, None, None] + means[1:4, None, None]
    img = img[[2, 1, 0], :, :].permute(1, 2, 0).numpy()
    return prepare_image(img)


def save_tensor_png(t: torch.Tensor, path: str):
    arr = tensor_to_numpy_rgb(t)
    Image.fromarray(arr).save(path)


def build_transform_basic() -> transforms.Compose:
    # keep original simple and consistent
    return transforms.Compose([transforms.Resize(256)])


def build_strong_random_pipeline() -> transforms.Compose:
    # clearly stronger random pipeline
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(256, scale=(0.05, 1.0)),                 # allow very small crops
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.7),
        transforms.RandomAffine(
            degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=(-20, 20)
        ),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.7),
        transforms.RandomRotation(degrees=60),
        transforms.GaussianBlur(kernel_size=31, sigma=(1.5, 3.5)),
    ])


def get_train_dataset(dm) -> torch.utils.data.Dataset:
    for name in ("train_dataset", "dataset_train", "train_ds", "ds_train"):
        if hasattr(dm, name):
            return getattr(dm, name)
    raise AttributeError("Train dataset not found in DataModule.")


def main():
    parser = argparse.ArgumentParser(description="Save original and stronger named augmentations as separate PNG files.")
    parser.add_argument("--data_dir", type=str, default="/mnt/storagecube/joanna/Hydro/", help="Path to raw data")
    parser.add_argument("--model_name", type=str, default="mae", help="Model name for DataModule")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the PNG files")
    args = parser.parse_args()

    # Ensure project root on path if run directly (optional safety)
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Dataset with basic transform (TwoCrops so we get a tuple; we take first view as "original")
    dm_basic = HydroMoCoDataModule(
        data_dir=args.data_dir,
        batch_size=32,
        transform=TwoCropsTransform(build_transform_basic()),
        model_name=args.model_name,
    )
    dm_basic.setup("fit")
    ds_basic = get_train_dataset(dm_basic)

    if len(ds_basic) == 0:
        print("Dataset is empty. Check --data_dir.")
        sys.exit(1)
    if not (0 <= args.sample_idx < len(ds_basic)):
        print(f"sample_idx {args.sample_idx} out of range [0, {len(ds_basic)-1}].")
        sys.exit(1)

    orig_view1, _ = ds_basic[args.sample_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    # Save original
    save_tensor_png(orig_view1, os.path.join(args.out_dir, "00_original.png"))

    # Deterministic, stronger single-op augmentations
    H, W = orig_view1.shape[-2], orig_view1.shape[-1]
    tx = int(0.20 * W)  # 20% width
    ty = int(0.20 * H)  # 20% height

    aug_ops = [
        ("01_randresizedcrop_strong.png", transforms.RandomResizedCrop(256, scale=(0.05, 0.4))),
        ("02_hflip.png", lambda x: F.hflip(x)),
        ("03_vflip.png", lambda x: F.vflip(x)),
        ("04_rotate_p60.png", lambda x: F.rotate(x, 60)),
        ("05_rotate_m60.png", lambda x: F.rotate(x, -60)),
        ("06_affine_t20_s130_shear15.png", lambda x: F.affine(x, angle=0, translate=(tx, ty), scale=1.3, shear=[-15.0, 15.0])),
        ("07_affine_t-20_s070_shear-15.png", lambda x: F.affine(x, angle=0, translate=(-tx, -ty), scale=0.7, shear=[15.0, -15.0])),
        ("08_perspective_strong.png", transforms.RandomPerspective(distortion_scale=0.40, p=1.0)),
        ("09_blur_strong.png", transforms.GaussianBlur(kernel_size=31, sigma=(2.5, 3.5))),
    ]

    for fname, aug in aug_ops:
        aug_img = aug(orig_view1) if callable(aug) else aug(orig_view1)
        save_tensor_png(aug_img, os.path.join(args.out_dir, fname))

    # Also save one strong random combo
    strong_random = build_strong_random_pipeline()
    save_tensor_png(strong_random(orig_view1), os.path.join(args.out_dir, "10_strong_random.png"))

    print(f"Saved images to: {args.out_dir}")


if __name__ == "__main__":
    main()