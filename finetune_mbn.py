import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, Tuple
from torchvision import transforms

# from src.utils.finetuning_utils import write_geotiff, read_geotiff
from src.models.finetuning.mbn.mae_finetuning_mbn import MAEFineTuning
from src.models.mae import MAE
from src.models.moco import MoCo
from src.models.moco_geo import MoCoGeo
from src.data.magicbathynet.mbn_dataloader import MagicBathyNetDataModule
import torch


class BathymetryPredictor:
    def __init__(
        self,
        pretrained_weights_path: str,
        data_dir: str,
        model_type: str = "mae", 
        output_dir: str = "./inference_results/bathymetry",
        batch_size: int = 16,
        resize_to: Tuple[int, int] = (3, 256, 256), # This tuple looks like (C, H, W)
        location: Optional[str] = "agia_napa",
        epochs=10
    ):
        self.full_finetune = True # Set to True for finetuning
        self.random = False
        self.ssl = False
        self.location = location
        self.model_type = model_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

        # Load pre-trained model based on model_type
        print(f"Loading pretrained model of type: {self.model_type} from {pretrained_weights_path}")
        if self.model_type.lower() == "mae":
            self.pretrained_model = MAE.load_from_checkpoint(
                pretrained_weights_path,
                strict=False # Use strict=False if the checkpoint has extra keys
            )
        elif self.model_type.lower() == "moco":
            self.pretrained_model = MoCo.load_from_checkpoint(
                pretrained_weights_path,
                strict=False
            )

        elif self.model_type.lower() == "mocogeo":
            self.pretrained_model = MoCoGeo.load_from_checkpoint(
                pretrained_weights_path,
                strict=False
            )
        # Initialize data module with transformations
        self.data_module = MagicBathyNetDataModule(
            root_dir=data_dir,
            batch_size=batch_size,
            transform=transforms.Compose([  
                transforms.ToTensor()]),
            pretrained_model=self.pretrained_model,
            location=self.location,
            full_finetune=self.full_finetune,
            random=self.random, 
            ssl=self.ssl
        )
        
        self.model = MAEFineTuning(
            location=location,
            full_finetune=self.full_finetune, 
            random=False,
            ssl=self.ssl,
            model_type = self.model_type,
            pretrained_model=self.pretrained_model
        )

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, max_epochs: int = 10) -> pl.Trainer:
        logger = TensorBoardLogger("results/inference", name="finetuning_logs")

        trainer = Trainer(
            accelerator=self.device.type,
            devices=1,
            max_epochs=max_epochs,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            #val_check_interval=1.0,
            # Ensure no batch limits are set here that could interfere
            #limit_train_batches=1.0, 
            #limit_val_batches=1.0,
        )

        # Perform model training
        trainer.fit(self.model, datamodule=self.data_module)
        trainer.test(self.model, datamodule=self.data_module)
        return trainer


def main():
    # Set up argument parser for configurable parameters
    parser = argparse.ArgumentParser(description="Bathymetry Prediction Pipeline")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Type of accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Training batch size") # Changed default to 1
    parser.add_argument("--val-batch-size", type=int, default=1, help="Validation batch size") # Changed default to 1
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--model", type=str, default="mae",
                        help="Type of pretrained model to load (mae, moco, mocogeo)") # New argument
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weights",default="/home/jovyan/SSLORS/SSLORS2/results/trains/training_logs/3-channels/checkpoints/epoch=99-step=132700.ckpt" , help="Path to pretrained weights checkpoint")
    parser.add_argument("--data_dir", default="/home/jovyan/SSLORS/mbn/MagicBathyNet", help="Path to data directory")
    parser.add_argument("--output_dir", default="./inference_results", help="Output directory")
    parser.add_argument("--resize-to", type=int, nargs=2, default=(256, 256), help="Resize images (height width)")
    parser.add_argument("--location", type=str, default="puck_lagoon", help="location")

    # Parse arguments and initialize predictor
    args = parser.parse_args()
    model_type = args.model.lower()
    location = args.location
    # Pass the new 'model_type' argument to BathymetryPredictor
    predictor = BathymetryPredictor(
        pretrained_weights_path=args.weights,
        data_dir=args.data_dir,
        model_type=model_type, # Pass the model type
        output_dir=args.output_dir,
        resize_to=tuple(args.resize_to),
        batch_size=args.train_batch_size,
        location=location,
        epochs=args.epochs
    )
        # Run prediction workflow
    predictor.train(max_epochs=args.epochs) 

if __name__ == "__main__":
    main()
