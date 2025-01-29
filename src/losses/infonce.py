import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Initializes the InfoNCELoss with a specified temperature.
        
        Parameters:
        - temperature (float): Scaling factor for logits, typically a small value like 0.07.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Computes the InfoNCE loss for contrastive learning.
        
        Parameters:
        - anchor (torch.Tensor): Anchor (query) embeddings of shape (batch_size, embedding_dim).
        - positive (torch.Tensor): Positive (key) embeddings of shape (batch_size, embedding_dim).
        - negatives (torch.Tensor): Negative samples embeddings of shape (batch_size, num_negatives, embedding_dim).
        
        Returns:
        - loss (torch.Tensor): The computed InfoNCE loss.
        """
        # Normalize all embeddings to unit vectors
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Compute similarity scores
        positive_score = torch.sum(anchor * positive, dim=-1) / self.temperature
        negative_scores = torch.bmm(anchor.unsqueeze(1), negatives.permute(0, 2, 1)) / self.temperature
        negative_scores = negative_scores.squeeze(1)  # Shape (batch_size, num_negatives)

        # Concatenate positive and negative scores
        logits = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1)  # Shape (batch_size, 1 + num_negatives)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)  # 0 for the positive sample
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss
