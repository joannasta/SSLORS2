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
import torchvision.transforms.functional as F
import random

#from src.utils.finetuning_utils import write_geotiff, read_geotiff 
from src.models.finetuning.marida.marida_mae import MAEFineTuning
from src.data.marida import marida_dataloader

    
class MarineDebrisPredictor:
    def __init__(
        self, 
        pretrained_weights_path: str, 
        data_dir: str, 
        output_dir: str = "./inference_results/marine_debris",
        batch_size: int = 32,
        resize_to: Tuple[int, int] = (3, 256, 256)
    ):

        # Validate input paths
        self._validate_paths(pretrained_weights_path, data_dir)
        
        # Determine computational device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize data module with transformations
        self.data_module = marida_dataloader.MaridaDataModule(
            root_dir=data_dir,
            batch_size=batch_size
        )
        
        # Load pre-trained model
        self.model = MAEFineTuning.load_from_checkpoint(
            pretrained_weights_path, 
            strict=False).to(self.device)
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _validate_paths(self, weights_path: str, data_dir: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
    
    def train(self, max_epochs: int = 100) -> pl.Trainer:
        # Configure TensorBoard logger for tracking
        logger = TensorBoardLogger("results/inference/marine_debris", name="finetuning_logs")

        # Initialize trainer with specific configurations
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=max_epochs,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            val_check_interval=1.0,
            limit_train_batches=1, 
            limit_val_batches=0,  # Skip validation
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
    parser.add_argument("--train-batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--model", type=str, default="mae", help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weights", required=True, help="Path to pretrained weights")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--output_dir", default="./inference_results", help="Output directory")
    parser.add_argument("--resize-to", type=int, nargs=2, default=(256, 256), help="Resize images")

    # Parse arguments and initialize predictor
    args = parser.parse_args()
    predictor = MarineDebrisPredictor(
        pretrained_weights_path=args.weights,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resize_to=tuple(args.resize_to)
    )
    
    predictor.train()


if __name__ == "__main__":
    main()