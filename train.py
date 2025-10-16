import argparse
import torch
import random
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.geography_aware import GeographyAware
from src.models.moco import MoCo
from src.models.mae import MAE
from src.models.mae_ocean import MAE_Ocean
from src.models.ocean_aware import OceanAware
from src.data.hydro.geography_aware.hydro_dataloader_geography_aware import HydroGeographyAwareDataModule
from src.data.hydro.ocean_aware.hydro_dataloader_ocean_aware import HydroOceanAwareDataModule
from src.data.hydro.moco.hydro_dataloader_moco import HydroMoCoDataModule
from src.data.hydro.mae_ocean.hydro_dataloader_mae_ocean import HydroMaeOceanFeaturesDataModule
from src.data.hydro.mae.hydro_dataloader import HydroDataModule
from torchvision import transforms
from src.utils.two_crop_transformations import GaussianBlur, TwoCropsTransform

models = {
    "mae": MAE,
    "mae_ocean" : MAE_Ocean,
    "moco": MoCo,
    "geo_aware": GeographyAware,
    "ocean_aware": OceanAware,
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
    transform = None 

    csv_file_path = args.csv_file
    limit_files=args.limit_files
    
    print("Training model:",args.model)
    print("Limit Files to Train on same Subset as Ocean Features:", limit_files)
    
    # Initialize Model and Define Transform Pipeline based on selected model 
    
    if args.model == "mae":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
        ])
        model = MAE(
            src_channels=12, 
            mask_ratio=args.mask_ratio,
            decoder_dim=args.decoder_dim,
        )
        datamodule = HydroDataModule( 
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers,
            limit_files=limit_files
        )
        
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
            csv_file_path=csv_file_path,
            limit_files=limit_files
        )
        
    elif args.model == "geo_aware":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = GeographyAware(
            src_channels=3 
        )
        datamodule = HydroGeographyAwareDataModule(
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            num_workers=args.num_workers,
            csv_file_path=csv_file_path,
            limit_files=limit_files
        )
    elif args.model == "ocean_aware":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = OceanAware(
            src_channels=3 
        )
        datamodule = HydroOceanAwareDataModule( 
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers,
            csv_file_path=csv_file_path,
            limit_files=limit_files
        )

    elif args.model == "moco":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([GaussianBlur(sigma_range=[.1, 2.])], p=0.5), 
            transforms.RandomHorizontalFlip(),
        ]
        transform = TwoCropsTransform(transforms.Compose(augmentations))
        model = MoCo(
            src_channels=12
        )
        datamodule = HydroMoCoDataModule(
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            num_workers=args.num_workers,
            limit_files=limit_files
        )
    
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(models.keys())}")
    
    # Setup Dataloader and Trainer

    datamodule.setup("fit") 
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    logger = TensorBoardLogger("results/trains", name=args.model)
    
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

def parse_args():
    ''' Training setup and arguments'''
    parser = argparse.ArgumentParser(description="Train script for SSL models.")
    
    parser.add_argument("--accelerator", default="cpu", type=str, help="Training accelerator: 'cpu' or 'gpu'")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices to use for training")
    parser.add_argument("--train-batch-size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--val-batch-size", default=4, type=int, help="Batch size for validation")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--dataset", default="/data/joanna/Hydro", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, choices=models.keys(), default="mae", help="Model architecture")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--csv_file", default="/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv", type=str, help="Path to ocean features csv")
    parser.add_argument("--limit_files", default=False, type=bool, help="Bool to limit files to train on same subset as ocean features")
    parser.add_argument("--mask-ratio", default=0.90, type=float, help="Masking ratio for MAE")
    parser.add_argument("--decoder-dim", default=512, type=int, help="Dimension of the MAE decoder")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
