# utils/training_utils.py
import torch
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer

def setup_finetuning(labeled_data_dir, bands, train_batch_size, val_batch_size, num_workers):
    """
    Prepare the smaller labeled dataset for fine-tuning.
    """
    # Load the labeled dataset
    labeled_dataset = LabeledHydroDataset(
        path_dataset=labeled_data_dir,
        bands=bands,
        transform=T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=bands.mean, std=bands.std),
        ])
    )

    # Split the dataset for fine-tuning (train/val split)
    total_size = len(labeled_dataset)
    val_size = int(total_size * 0.2)  # 20% for validation
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Setup DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader


def fine_tune_model(model, pretrained_weights_path, train_dataloader, val_dataloader, learning_rate=1e-4, epochs=10):
    """
    Fine-tune a pretrained model using a smaller labeled dataset.
    """
    # Load pretrained weights
    model.load_state_dict(torch.load(pretrained_weights_path))

    # Optionally freeze the backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False  # Freeze backbone layers

    # Setup loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # or other suitable loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",  # Change based on your setup
        devices=1,
        val_check_interval=1.0  # Evaluate after each epoch
    )

    # Fine-tune the model
    trainer.fit(model, train_dataloader, val_dataloader)
