import argparse
import torch
import random
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.moco_geo import MoCoGeo
from src.models.moco import MoCo
from src.models.mae import MAE
from src.models.mae_ocean import MAE_Ocean
from src.models.moco_geo_ocean import MoCoOceanFeatures
from src.data.hydro.hydro_dataloader_moco_geo import HydroMoCoGeoDataModule
from src.data.hydro.hydro_dataloader_moco_geo_ocean import HydroOceanFeaturesDataModule
from src.data.hydro.hydro_dataloader_moco import HydroMoCoDataModule
from src.data.hydro.hydro_dataloader_mae_ocean import HydroMaeOceanFeaturesDataModule
from src.data.hydro.hydro_dataloader import HydroDataModule
from torchvision import transforms
from src.utils.mocogeo_utils import GaussianBlur, TwoCropsTransform

models = {
    "mae": MAE,
    "mae_ocean" : MAE_Ocean,
    "moco": MoCo,
    "geo_aware": MoCoGeo,
    "ocean_aware": MoCoOceanFeatures,
}

def set_seed(seed):
    """Set the random seed for reproducibility."""
    print(f"DEBUG: Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    """Main training function."""
    print("DEBUG: Starting main training function.")
    set_seed(args.seed)

    model = None
    datamodule = None
    transform = None # Initialize transform here to ensure scope

    ocean_flag = args.ocean 
    print("Training model:",args.model)
    print("Using Ocean dataset:", ocean_flag)
    
    print(f"DEBUG: Initializing model and transform for model: {args.model}")
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
            num_workers=args.num_workers,
            ocean_flag=ocean_flag
        )
        print("DEBUG: MAE model and HydroDataModule initialized.")
        
    elif args.model == "mae_ocean":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
        ])
        model = MAE_Ocean(
            src_channels=12, 
            mask_ratio=args.mask_ratio,
            decoder_dim=args.decoder_dim,
        )
        datamodule = HydroMaeOceanFeaturesDataModule( 
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers,
            ocean_flag=ocean_flag
        )
        print("DEBUG: MAE_Ocean model and HydroDataModule initialized.")
        
    elif args.model == "geo_aware":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = MoCoGeo(
            src_channels=3 # Assuming 12 channels for Hydro data
        )
        datamodule = HydroMoCoGeoDataModule( # Specific DataModule for MoCo-Geo
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers,
            ocean_flag=ocean_flag
        )
        print("DEBUG: MoCoGeo model and HydroMoCoGeoDataModule initialized.")
    elif args.model == "ocean_aware":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = MoCoOceanFeatures(
            src_channels=3 # Assuming 12 channels for Hydro data
        )
        datamodule = HydroOceanFeaturesDataModule( # Specific DataModule for MoCo-Geo
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers,
            ocean_flag=ocean_flag
        )
        print("DEBUG: MoCoOceanFeatures model and HydroOceanFeaturesDataModule initialized.")
        
    elif args.model == "moco":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
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
            num_workers=args.num_workers,
            ocean_flag=ocean_flag
        )
        print("DEBUG: MoCo model and HydroMoCoDataModule initialized.")
    
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(models.keys())}")

    # Ensure model and datamodule are initialized
    if model is None or datamodule is None:
        raise RuntimeError("Model or DataModule could not be initialized. Check model argument.")

    print("DEBUG: Calling datamodule.setup('fit').")
    datamodule.setup("fit") # Prepare datasets for training and validation
    print("DEBUG: Getting train_dataloader.")
    train_dataloader = datamodule.train_dataloader()
    print("DEBUG: Getting val_dataloader.")
    val_dataloader = datamodule.val_dataloader()

    print("DEBUG: Setting up TensorBoardLogger.")
    # --- Setup Trainer and Run Fit ---
    logger = TensorBoardLogger("results/trains", name=args.model)

    print(f"DEBUG: Initializing Trainer with epochs: {args.epochs}")
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        val_check_interval=1.0
    )
    print("DEBUG: Calling trainer.fit().")
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    print("DEBUG: Training finished.")

# --- Argument Parsing ---
def parse_args():
    """Parse command-line arguments."""
    print("DEBUG: Parsing command-line arguments.")
    parser = argparse.ArgumentParser(description="Train script for SSL models.")

    # General training arguments
    parser.add_argument("--accelerator", default="cpu", type=str, help="Training accelerator: 'cpu' or 'gpu'")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices to use for training")
    parser.add_argument("--train-batch-size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--val-batch-size", default=4, type=int, help="Batch size for validation")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--dataset", default="/data/joanna/Hydro", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, choices=models.keys(), default="mae", help="Model architecture")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--ocean", default=True, type=bool, help="Flag to indicate ocean dataset")

    # MAE-specific arguments
    parser.add_argument("--mask-ratio", default=0.90, type=float, help="Masking ratio for MAE")
    parser.add_argument("--decoder-dim", default=512, type=int, help="Dimension of the MAE decoder")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

    args = parser.parse_args()
    print("DEBUG: Arguments parsed.")
    return args

if __name__ == "__main__":
    print("DEBUG: Script started. Entering __main__ block.")
    args = parse_args()
    main(args)
