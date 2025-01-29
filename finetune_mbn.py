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

#from src.utils.finetuning_utils import write_geotiff, read_geotiff 
from src.models.finetuning.mbn.mae_finetuning_mbn import MAEFineTuning
from src.data.magicbathynet.mbn_dataloader import MagicBathyNetDataModule


class BathymetryPredictor:
    def __init__(
        self, 
        pretrained_weights_path: str, 
        data_dir: str, 
        output_dir: str = "./inference_results/bathymetry",
        batch_size: int = 32,
        modality: str = "s2",
        resize_to: Tuple[int, int] = (3, 256, 256)
    ):

        # Validate input paths
        self._validate_paths(pretrained_weights_path, data_dir)
        
        # Determine computational device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize data module with transformations
        self.data_module = MagicBathyNetDataModule(
            root_dir=data_dir,
            batch_size=batch_size,
            modality=modality,
            transform=transforms.Compose([  
                transforms.ToTensor(),
            ])
        )
        
        # Load pre-trained model
        self.model = MAEFineTuning.load_from_checkpoint(
            pretrained_weights_path, 
            strict=False
        ).to(self.device)
        
        
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
        logger = TensorBoardLogger("results/inference", name="finetuning_logs")

        # Initialize trainer with specific configurations
        trainer = Trainer(
            accelerator=self.device.type,
            devices=1,
            max_epochs=max_epochs,
            logger=logger,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            val_check_interval=1.0,
            limit_train_batches=1, 
            limit_val_batches=1,  # Skip validation
        )

        # Perform model training
        trainer.fit(self.model, datamodule=self.data_module)
        trainer.test(self.model, datamodule=self.data_module)
        return trainer
    
    def evaluate(self) -> float:
        self.model.eval()
        all_preds, all_gts = [], []
        
        with torch.no_grad():
            for batch in self.data_module.test_dataloader():
                images, depth = batch
                preds = self.model([images, depth])
                all_preds.append(preds.cpu().numpy())
                all_gts.append(depth.cpu().numpy())
        
        # Calculate Root Mean Square Error
        all_preds, all_gts = np.concatenate(all_preds), np.concatenate(all_gts)
        return np.sqrt(np.mean((all_preds - all_gts) ** 2))
    
    def predict_and_save(self, norm_param_depth: float = -30.443, reference_geotiff: Optional[str] = None):
        self.model.eval()
        crop_size = 256
        WINDOW_SIZE = (18, 18)
        norm_param_depth = -30.443   #-30.443 FOR AGIA NAPA, -11 FOR PUCK LAGOON
        ratio = crop_size / WINDOW_SIZE[0]
        
        for batch_idx, batch in enumerate(self.data_module.test_dataloader()):
            inputs, target = batch
            test_ids = self.data_module.test_images
            
            with torch.no_grad():
                preds = self.model([inputs, target])
                
            
            # Save predictions as images
            for idx, pred in enumerate(preds):                
                # Denormalize image
                img = pred.cpu().numpy() * norm_param_depth
                img = scipy.ndimage.zoom(img, (1/ratio, 1/ratio), order=1)
                #img = np.squeeze(img)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(img, cmap='viridis')
                plt.title(f"Predicted Depth for Image {test_ids[idx]}")
                plt.colorbar(label='Depth')
                plt.savefig(os.path.join(self.output_dir, f'prediction_{test_ids[idx]}.png'))
                print("IMAGE HAS BEEN SAVED")
                plt.close()

                # Commented out geotiff writing for now
                # if reference_geotiff:
                #     _, ref_dataset = read_geotiff(reference_geotiff, 3)
                #     write_geotiff(
                #         os.path.join(self.output_dir, f'inference_tile_{test_ids[idx]}.tif'), 
                #         pred_img, 
                #         ref_dataset
                #     )


def main():
    # Set up argument parser for configurable parameters
    parser = argparse.ArgumentParser(description="Bathymetry Prediction Pipeline")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Type of accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=4, help="Validation batch size")
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
    predictor = BathymetryPredictor(
        pretrained_weights_path=args.weights,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resize_to=tuple(args.resize_to)
    )

    # Run prediction workflow
    predictor.train()
    #print(f"Test RMSE: {predictor.evaluate()}")
    #predictor.predict_and_save()


if __name__ == "__main__":
    main()
