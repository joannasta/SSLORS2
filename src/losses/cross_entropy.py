import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        """
        Computes the cross-entropy loss between predictions and targets.
        
        Parameters:
        - predictions (torch.Tensor): Model predictions (logits) of shape (batch_size, num_classes).
        - targets (torch.Tensor): True labels of shape (batch_size).
        
        Returns:
        - loss (torch.Tensor): The computed cross-entropy loss.
        """
        return self.criterion(predictions, targets)
