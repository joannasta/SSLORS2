import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels=192, latent_size=32):
        super(UNet_bathy, self).__init__()
        self.n_channels = in_channels
        self.n_outputs = out_channels
        self.latent_channels = latent_channels  # Latent channels
        self.latent_size = latent_size  # Spatial size of the latent feature map
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Fully connected layer, dynamically computed
        self.fc = nn.Linear(
            in_features=3072,
            out_features=self.latent_channels * self.latent_size * self.latent_size,
        )

        # Channel projection for latent feature adjustment
        self.channel_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256)
        )

        self.decoder = nn.Sequential(
            Up(256, 128),
            Up(128, 64),
            Up(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )


    def forward(self,x, images):
        images = images.float()  # Already normalized by dataset
        batch_size, channels, features = x.shape  # (7, 13, 768)
        target_height = 32
        target_width = features // target_height  # Ensure compatibilitys
        x = x.view(batch_size, channels, target_height, target_width)  # (7, 13, 32, 24)
        # Interpolate to (32, 32) for U-Net compatibility
        x = F.interpolate(x, size=(32, 32), mode="nearest")
        # Project channels to match encoder
        x = self.channel_projection(x)
        images = images.to(self._device)
        x1 = self.encoder[0](images)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x4 = x
        x = self.decoder[0](x4, x3)
        x = self.decoder[1](x, x2)
        x = self.decoder[2](x, x1)
        output = self.decoder[3](x)
        
        return output    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculate the difference in shape and pad x2 if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
