import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out

class UNet_Marida(nn.Module):
    def __init__(self, input_channels=11, out_channels=11, hidden_channels=16, embedding_dim=128,model_type="mae"):
        super(UNet_Marida, self).__init__()

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.embedding_projector = nn.Linear(768, 256)

        self.moco_projection =  nn.Linear(in_features=512, out_features=128 * 16 * 16)
        self.mocogeo_projection =  nn.Linear(in_features=512, out_features=128 * 16 * 16)

        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True))

        self.down1 = Down(self.hidden_channels, 2 * self.hidden_channels)
        self.down2 = Down(2 * self.hidden_channels, 4 * self.hidden_channels)
        self.down3 = Down(4 * self.hidden_channels, 8 * self.hidden_channels)
        self.down4 = Down(8 * self.hidden_channels, 8 * self.hidden_channels)

        self.up1 = Up(256 + (8 * self.hidden_channels), 4 * self.hidden_channels)
        self.up2 = Up(8 * self.hidden_channels, 2 * self.hidden_channels)
        self.up3 = Up(4 * self.hidden_channels, self.hidden_channels)
        self.up4 = Up(2 * self.hidden_channels, self.hidden_channels)

        self.outc = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        self.embedding_dim = embedding_dim
        self.combined_projection = nn.Linear(256, 256) # Corrected in_features to 256


    def forward(self, image,x_embedding):
        print("image shape before:", image.shape)
        #image = image.permute(0,3,1,2)
        print("image shape after:", image.shape)
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        processed_x_embedding = None

        if self.model_type == "mae":
            projected_embedding = self.embedding_projector(x_embedding)
            if projected_embedding.shape[0] == 1 and projected_embedding.shape[1] == x5.shape[0]:
                 projected_embedding = projected_embedding.squeeze(0)
            patch_tokens = projected_embedding[:, 1:, :]
            batch_size, num_patches, features = patch_tokens.shape
            spatial_size = int(num_patches**0.5)
            spatial_embedding = patch_tokens.permute(0, 2, 1).reshape(
                batch_size, features, spatial_size, spatial_size
            )
            processed_x_embedding = F.interpolate(
                spatial_embedding,
                size=(x5.shape[2], x5.shape[3]),
                mode='bilinear',
                align_corners=False
            )

        elif self.model_type == "moco":
            x_embedding_flat = self.moco_projection(x_embedding)
            processed_x_embedding = x_embedding_flat.view(
                x5.shape[0], x5.shape[1], x5.shape[2], x5.shape[3]
            )

        elif self.model_type == "mocogeo":
            x_embedding_flat = self.mocogeo_projection(x_embedding)
            if x_embedding_flat.shape[0] == 1 and x_embedding_flat.shape[1] == x5.shape[0]:
                x_embedding_flat = x_embedding_flat.squeeze(0)
            processed_x_embedding = x_embedding_flat.view(
                x5.shape[0], x5.shape[1], x5.shape[2], x5.shape[3]
            )

        combined = torch.cat([processed_x_embedding, x5], dim=1)
        batch_size, channels_combined, height, width = combined.shape
        combined_reshaped = combined.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels_combined)
        combined_projected = self.combined_projection(combined_reshaped).reshape(batch_size, 256, height, width)

        x6 = self.up1(combined_projected, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.outc(x9)
        return logits