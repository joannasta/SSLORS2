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

from src.models.finetuning.mbn.mae_finetuning_mbn import MAEFineTuning
from src.models.mae import MAE
from src.models.moco_geo import MoCoGeo
from src.models.moco import MoCo
from src.data.magicbathynet.mbn_dataloader import MagicBathyNetDataModule


class BathymetryPredictor:
    def __init__(
        self, 
        pretrained_weights_path: str, 
        data_dir: str, 
        output_dir: str = "./inference_results/bathymetry",
        batch_size: int = 16,
        resize_to: Tuple[int, int] = (256, 256),
        model_name: str = "mae",
        src_channels: int = 12,
        num_geo_clusters: int = 100,
        location: str = "agia_napa",
        full_finetune: bool = True,
        random: bool = False,
        ssl: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.location = location
        self.model_name = model_name
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl
        
        self.pretrained_model = None
        self.encoder_backbone = None

        if self.model_name == "mae":
            self.pretrained_model = MAE.load_from_checkpoint(
                pretrained_weights_path, 
                strict=False
            )
            self.encoder_backbone = self.pretrained_model.backbone
        elif self.model_name == "moco":
            self.pretrained_model = MoCo.load_from_checkpoint(
                pretrained_weights_path,
                strict=False,
                src_channels=src_channels 
            )
            self.encoder_backbone = self.pretrained_model.backbone
        elif self.model_name == "moco-geo":
            self.pretrained_model = MoCoGeo.load_from_checkpoint(
                pretrained_weights_path,
                strict=False,
                src_channels=src_channels,
                num_geo_classes=num_geo_clusters, 
            )
            self.encoder_backbone = self.pretrained_model.backbone
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}.")
        
        if self.encoder_backbone:
            self.encoder_backbone.eval()
            for param in self.encoder_backbone.parameters():
                param.requires_grad = False
        else:
            raise RuntimeError(f"Could not extract backbone for model: {self.model_name}")

        self.data_module = MagicBathyNetDataModule(
            root_dir=data_dir,
            batch_size=batch_size,
            transform=transforms.Compose([  
                transforms.ToTensor(),
                transforms.Resize(resize_to, antialias=True),
            ]),
            location=self.location,
            pretrained_model=self.pretrained_model,
            full_finetune=self.full_finetune,
            random=self.random,
            ssl=self.ssl,
        )

        self.model = MAEFineTuning(location=self.location,pretrained_weights=self.encoder_backbone)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, max_epochs: int = 100) -> pl.Trainer:
        logger = TensorBoardLogger("results/inference", name="finetuning_logs")

        trainer = Trainer(
            accelerator=self.device.type,
            devices=1,
            max_epochs=max_epochs,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            val_check_interval=1.0, 
        )

        trainer.fit(self.model, datamodule=self.data_module)
        trainer.test(self.model, datamodule=self.data_module) 
        return trainer


def main():
    parser = argparse.ArgumentParser(description="Bathymetry Prediction Pipeline")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Type of accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default="mae")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./inference_results")
    parser.add_argument("--resize-to", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--src-channels", type=int, default=12)
    parser.add_argument("--num-geo-clusters", type=int, default=100)
    parser.add_argument("--location", type=str, default="agia_napa")
    parser.add_argument("--full_finetune", type=bool, default=True)
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--ssl", type=bool, default=False)


    args = parser.parse_args()
    
    
    predictor = BathymetryPredictor(
        pretrained_weights_path=args.weights,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resize_to=tuple(args.resize_to),
        batch_size=args.train_batch_size,
        model_name=args.model,
        location=args.location,
        src_channels=args.src_channels,
        num_geo_clusters=args.num_geo_clusters,
        full_finetune=args.full_finetune,
        random=args.random,
        ssl=args.ssl
    )
    
    predictor.train(max_epochs=args.epochs)

if __name__ == "__main__":
    main()