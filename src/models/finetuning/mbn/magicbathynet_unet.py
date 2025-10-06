import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels_up, in_channels_skip, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_up, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + in_channels_skip, out_channels)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)
        diff_y = x_skip.size()[2] - x_up.size()[2]
        diff_x = x_skip.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels, model_type="mae", full_finetune=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.full_finetune = full_finetune

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        bottleneck_input_channels = 256
            
        if model_type == "mae" or model_type == "mae_ocean":
            bottleneck_input_channels += 256
            self.mae_pre_proj = nn.Linear(768, 512) 
            self.mae_spatial_proj = nn.Linear(512, 256 * 32 * 32)
        elif model_type == "moco":
            bottleneck_input_channels += 256
            self.moco_pre_proj = nn.Linear(512, 512) 
            self.moco_spatial_proj = nn.Linear(512, 256 * 32 * 32)
        elif model_type in ["geo_aware","ocean_aware"]:
            bottleneck_input_channels += 512
        else:
            raise NotImplementedError(f"Model type '{model_type}' not supported.")
        
        self.feature_linear = nn.Linear(bottleneck_input_channels, 512)

        self.bottleneck = DoubleConv(512, 512)

        self.up1 = Up(512, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, embedding, images):
        images = images.to(self.device)

        x1 = self.inc(images)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        fused_features = None
        
        if self.model_type == "mae" or self.model_type== "mae_ocean":
            if embedding.dim() == 4:
                cls_feat = embedding.squeeze(1)[:, 0, :]
            elif embedding.dim() == 3:
                cls_feat = embedding[:, 0, :]
            elif embedding.dim() == 2 and embedding.shape[-1] == 768:
                cls_feat = embedding
            else:
                raise ValueError(f"Unexpected embedding dimensions for {self.model_type}: {embedding.shape}.")   
            proj_emb = self.mae_pre_proj(cls_feat)
            spatial_flat = self.mae_spatial_proj(proj_emb)
            
            spatial_2d = spatial_flat.view(embedding.shape[0], 256, 32, 32)
            upsampled_emb = F.interpolate(spatial_2d, size=x4.shape[2:], mode='bilinear', align_corners=False)
            fused_features = torch.cat([x4, upsampled_emb], dim=1)

        elif self.model_type == "moco":
            proj_emb = self.moco_pre_proj(embedding)
            spatial_flat = self.moco_spatial_proj(proj_emb)
            spatial_2d = spatial_flat.view(embedding.shape[0], 256, 32, 32)
            upsampled_emb = F.interpolate(spatial_2d, size=x4.shape[2:], mode='bilinear', align_corners=False)
            fused_features = torch.cat([x4, upsampled_emb], dim=1)

        elif self.model_type in ["geo_aware","ocean_aware"]:
            if embedding.dim() > 2:
                embedding = embedding.flatten(start_dim=1) 
            
            emb_expanded = embedding.unsqueeze(2).unsqueeze(3)
            emb_expanded = emb_expanded.expand(-1, -1, x4.shape[2], x4.shape[3])
            fused_features = torch.cat([x4, emb_expanded], dim=1)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not supported during forward pass.")

        bs, ch_fused, h_fused, w_fused = fused_features.shape
        fused_reshaped = fused_features.permute(0, 2, 3, 1).reshape(bs * h_fused * w_fused, ch_fused)

        bottleneck_in = self.feature_linear(fused_reshaped).reshape(bs, 512, h_fused, w_fused)
        bottleneck_out = self.bottleneck(bottleneck_in)

        x = self.up1(bottleneck_out, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        output = self.outc(x)
        return output