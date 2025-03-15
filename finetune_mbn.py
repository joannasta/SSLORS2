import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import pytorch_lightning as pl
import torch.multiprocessing as mp

from scipy import ndimage
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, Tuple
from torchvision import transforms

#from src.utils.finetuning_utils import write_geotiff, read_geotiff 
from src.models.finetuning.mbn.mae_finetuning_mbn import MAEFineTuning
from src.models.mae import MAE
from src.data.magicbathynet.mbn_dataloader import MagicBathyNetDataModule

class BathymetryPredictor:
    def __init__(
        self, 
        pretrained_weights_path: str, 
        data_dir: str, 
        output_dir: str = "./inference_results/bathymetry",
        batch_size: int = 16,
        resize_to: Tuple[int, int] = (3, 256, 256)
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.location = "puck_lagoon"
        self.pretrained_model = MAE.load_from_checkpoint(
            pretrained_weights_path, 
            strict=False
        )
        print(f"RUN TEST FOR LOCATION: {self.location}")
        self.model = MAEFineTuning(pretrained_model=self.pretrained_model,location=self.location)
        self.data_module = MagicBathyNetDataModule(
            root_dir=data_dir,
            batch_size=batch_size,
            transform=transforms.Compose([  
                transforms.ToTensor()]),
            pretrained_model=self.pretrained_model,
            location=self.location
        )

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
            #val_check_interval=1.0
            #limit_train_batches=1, 
            #limit_val_batches=1,  # Skip validation
        )

        # Perform model training
        trainer.fit(self.model, datamodule=self.data_module)
        print("Testing starts")
        trainer.test(self.model, datamodule=self.data_module)
        return trainer


def main():
    # Set up argument parser for configurable parameters
    parser = argparse.ArgumentParser(description="Bathymetry Prediction Pipeline")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Type of accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--model", type=str, default="mae", help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weights", required=True, help="Path to pretrained weights")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--output_dir", default="./inference_results", help="Output directory")
    parser.add_argument("--resize-to", type=int, nargs=2, default=(256, 256), help="Resize images")

    args = parser.parse_args()
    predictor = BathymetryPredictor(
        pretrained_weights_path=args.weights,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resize_to=tuple(args.resize_to),
        batch_size=args.train_batch_size
    )

    predictor.train()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
