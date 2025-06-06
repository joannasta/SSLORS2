import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .mbn_dataset import MagicBathyNetDataset


class MagicBathyNetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, transform=None, cache=False,pretrained_model=None,location="agia_napa",
                 full_finetune=True, random=False, ssl=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.cache = cache
        self.pretrained_model = pretrained_model
        self.location=location
        self.full_finetune = full_finetune
        self.random = random
        self.ssl = ssl

    def setup(self, stage=None):
        # Use stage to load data depending on the task
        if stage == 'fit' or stage is None:
            # Initialize train and validation datasets for training
            self.train_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='train',
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl 
            )
            
            self.val_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='val',
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl
            )

        if stage == 'test' or stage is None:
            # Initialize test dataset for evaluation
            self.test_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='test',
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl
            )

        if stage == 'predict':
            # You can set up a different dataset for prediction if needed
            self.predict_dataset = MagicBathyNetDataset(
                root_dir=self.root_dir,
                transform=self.transform,
                split_type='predict',  # Adjust if there's a different mode
                cache=self.cache,
                pretrained_model=self.pretrained_model,
                location=self.location,
                full_finetune=self.full_finetune,
                random=self.random,
                ssl=self.ssl
            )


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
