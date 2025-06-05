import argparse
import torch
import random
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint # Import ModelCheckpoint
from src.models.moco_geo import MoCoGeo
from src.models.moco import MoCo
from src.models.mae import MAE
from PIL import ImageFilter # Not explicitly used in the transforms here, but kept
from src.data.hydro.hydro_dataloader_moco_geo import HydroMoCoGeoDataModule
from src.data.hydro.hydro_dataloader_moco import HydroMoCoDataModule
from src.data.hydro.hydro_dataloader import HydroDataModule
from pytorch_lightning.callbacks import ProgressBar # Already imported, but kept
from torchvision import transforms
from src.utils.mocogeo_utils import GaussianBlur, TwoCropsTransform

# Define available models
models = {
    "mae": MAE,
    "moco": MoCo,
    "moco-geo": MoCoGeo
}

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    # Set random seed
    set_seed(args.seed)

    # Select the model
    model_class = models[args.model]
    if args.model == "mae":
        model = model_class(
            src_channels=11, # Assuming your data has 11 channels
            mask_ratio=args.mask_ratio,
            decoder_dim=args.decoder_dim,
        )
    else:
        model = model_class(src_channels=11) # Assuming other models also take src_channels

    # Define transforms based on the model
    # Initialize transform to None, then set it based on model type
    transform = None

    if args.model == "moco-geo":
        # If 'aug_plus' is always True, you can remove the 'if aug_plus:' line.
        # Otherwise, make it a configurable argument.
        aug_plus = True
        if aug_plus: # This block is always executed if model is moco-geo
            augmentations = [
                transforms.Resize(224 * 2),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) # not strengthened
                ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(), # IMPORTANT: Add ToTensor if your dataset does not do it
                # transforms.Normalize(mean=[...], std=[...]) # Add normalization for 11 channels
            ]
            transform = TwoCropsTransform(transforms.Compose(augmentations))
            
        datamodule = HydroMoCoGeoDataModule(
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            model_name = args.model,
            num_workers=args.num_workers # Pass num_workers from args
        )

    elif args.model == "moco":
        augmentations = transforms.Compose([
            transforms.Resize(256 * 2),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([
                # T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomApply(
                [GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # <-- This is crucial!
            # Normalization will be applied in __getitem__ (as per your comment)
        ])
        transform = TwoCropsTransform(transforms.Compose(augmentations)) # <-- Your transform is now `TwoCropsTransform`
        datamodule = HydroMoCoDataModule(
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform, # <-- This `transform` (TwoCropsTransform) is passed
            model_name = args.model,
            num_workers=args.num_workers
        )
        
    elif args.model == "mae": # Specific transform for MAE
        transform = transforms.Compose([
            transforms.Resize(256), # Or the exact input size your MAE model expects
            transforms.CenterCrop(256), # Or the exact input size your MAE model expects
            #transforms.ToTensor(), # IMPORTANT: Add ToTensor if your dataset does not do it
            # transforms.Normalize(mean=[...], std=[...]) # Add normalization for 11 channels
        ])
        # Setup the DataModule
        datamodule = HydroDataModule(
            data_dir=args.dataset,
            batch_size=args.train_batch_size,
            transform=transform,
            num_workers=args.num_workers # Pass num_workers from args
        )

    else:
        # Fallback for any other model, or if no specific transform is defined
        print(f"Warning: No specific transform defined for model: {args.model}. Using a default.")
        transform = transforms.Compose([
            transforms.ToTensor(), # Fallback: Ensure data is a tensor
            # transforms.Normalize(mean=[...], std=[...]) # Add normalization for 11 channels
        ])

    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # Setup TensorBoard logger
    logger = TensorBoardLogger("results/trains", name=args.model) # Log under the model name

    # Setup Callbacks
    callbacks = [
        # Saves the best model based on validation loss
        ModelCheckpoint(
            monitor='val_loss', # Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            mode='min',         # 'min' for loss, 'max' for accuracy
            dirpath=f"checkpoints/{args.model}", # Directory to save checkpoints
            filename='{epoch:02d}-{val_loss:.2f}', # Checkpoint filename format
            save_top_k=1,       # Save only the best model
            verbose=True,
        ),
        ProgressBar() # Keeping ProgressBar as it was
    ]

    print(f"epochs: {args.epochs}")
    # Setup the Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        val_check_interval=1.0, # Ensures validation happens every epoch
        callbacks=callbacks # Add the callbacks list here
    )
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train script for SSL models.")

    # General training arguments
    parser.add_argument("--accelerator", default="gpu", type=str, help="Training accelerator: 'cpu' or 'gpu'") # Changed default to 'gpu' as it's more common for training
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