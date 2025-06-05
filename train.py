import argparse
import torch
import random
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.moco_geo import MoCoGeo
from src.models.moco import MoCo
from src.models.mae import MAE
from src.data.hydro.hydro_dataloader_moco_geo import HydroMoCoGeoDataModule
from src.data.hydro.hydro_dataloader_moco import HydroMoCoDataModule
from src.data.hydro.hydro_dataloader import HydroDataModule
from torchvision import transforms
from src.utils.mocogeo_utils import GaussianBlur, TwoCropsTransform

models = {
    "mae": MAE,
    "moco": MoCo,
    "moco-geo": MoCoGeo
}

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    """Main training function."""
    set_seed(args.seed)

    model = None
    datamodule = None
    transform = None # Initialize transform here to ensure scope

    # --- Initialize Model and Define Transform Pipeline based on selected model ---
    if args.model == "mae":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
        ])
        model = MAE(
            src_channels=12, # Assuming 12 channels for Hydro data
            mask_ratio=args.mask_ratio,
            decoder_dim=args.decoder_dim,
        )
        datamodule = HydroDataModule( # Assuming HydroDataModule for MAE
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers
        )
    elif args.model == "moco-geo":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            # Removed: ColorJitter line as per request
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = MoCoGeo(
            src_channels=12 # Assuming 12 channels for Hydro data
        )
        datamodule = HydroMoCoGeoDataModule( # Specific DataModule for MoCo-Geo
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers,
            csv_path="/home/joanna/SSLORS/src/data/hydro/train_geo_labels10_projected.csv" # Explicit CSV path
        )
        
    elif args.model == "moco":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            # Removed: ColorJitter line as per request
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = MoCo(
            src_channels=12 # Assuming 12 channels for Hydro data
        )
        datamodule = HydroMoCoDataModule( # Specific DataModule for MoCo
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers
        )
    
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(models.keys())}")

    # Ensure model and datamodule are initialized
    if model is None or datamodule is None:
        raise RuntimeError("Model or DataModule could not be initialized. Check model argument.")

    datamodule.setup("fit") # Prepare datasets for training and validation
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # --- Setup Trainer and Run Fit ---
    logger = TensorBoardLogger("results/trains", name=args.model)

    print(f"epochs: {args.epochs}")
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        val_check_interval=1.0
    )
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

# --- Argument Parsing ---
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train script for SSL models.")

    # General training arguments
    parser.add_argument("--accelerator", default="cpu", type=str, help="Training accelerator: 'cpu' or 'gpu'")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices to use for training")
    parser.add_argument("--train-batch-size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--val-batch-size", default=4, type=int, help="Batch size for validation")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--dataset", default="/faststorage/joanna/Hydro/raw_data", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, choices=models.keys(), default="mae", help="Model architecture")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")

    # MAE-specific arguments
    parser.add_argument("--mask-ratio", default=0.90, type=float, help="Masking ratio for MAE")
    parser.add_argument("--decoder-dim", default=512, type=int, help="Dimension of the MAE decoder")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)