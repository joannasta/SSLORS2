import argparse
import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.moco_geo import MoCoGeo
from src.models.cmae import CMAE
from src.models.moco import MoCo
from src.models.dino import DINO_LIT
from src.models.mae import MAE
from src.data.hydro.hydro_dataloader import HydroDataModule  # Correct import
from src.losses import cross_entropy, infonce
from pytorch_lightning.callbacks import ProgressBar




# Define available models and losses
models = {
    "mae": MAE,  
    "moco": MoCo,
    "dino": DINO_LIT,
    "cmae": CMAE,
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
            src_channels=3,
            mask_ratio=0.9,
            decoder_dim=args.decoder_dim,
        )
    else:
        model = model_class(src_channels=12) 
    
    

    # Setup the DataModule
    datamodule = HydroDataModule(
        data_dir=args.dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform =  T.Compose([
            T.RandomResizedCrop(
                    256,
                    scale=(0.67, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                #T.Lambda(lambda x: x / 10000.0),
                #T.Normalize(mean=self.band_means, std=self.band_stds)
            ]
        ) 
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # Setup TensorBoard logger
    logger = TensorBoardLogger("results/trains", name="training_logs")

    print(f"epochs: {args.epochs}")
    # Setup the Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        val_check_interval=1.0  # Ensure validation happens every epoch
        )
    trainer.fit(model, train_dataloader,val_dataloaders=val_dataloader) 


    
def parse_args():
    """
    Parse command-line arguments.
    """
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
