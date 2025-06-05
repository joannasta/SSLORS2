"""PyTorch Lightning implementation of CMAE (Contrastive Masked Autoencoder)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional
import math
import torchvision.transforms as transforms


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=256, patch_size=16, in_chans=12, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class TransformerBlock(nn.Module):
    """Transformer encoder block with LayerNorm before attention/MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(*[self.norm1(x)]*3)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    """Vision Transformer Encoder"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = self._get_pos_embed(self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)
        nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, embed_dim):
        """Get sinusoidal positional embeddings"""
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        pos_embed = self._get_2d_sincos_pos_embed(embed_dim, grid_size)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=True):
        """Get 2D sine-cosine positional embedding"""
        grid_h = grid_w = grid_size
        grid_h, grid_w = torch.meshgrid(
            torch.arange(grid_h), torch.arange(grid_w), indexing='ij'
        )
        grid = torch.stack([grid_h, grid_w], dim=-1).float()
        grid = grid.reshape([grid_h.shape[0] * grid_h.shape[1], 2])

        embed_dim = embed_dim // 2
        omega = torch.arange(embed_dim).float() / embed_dim
        omega = 1. / (10000**omega)

        out = torch.einsum('m,d->md', grid.flatten(), omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        pos_embed = torch.cat([emb_sin, emb_cos], dim=1)
        if cls_token:
            pos_embed = torch.cat([torch.zeros([1, pos_embed.shape[1]]), pos_embed], dim=0)
        return pos_embed

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

class MAEDecoder(nn.Module):
    """Masked Autoencoder Decoder"""
    def __init__(self, num_patches=196, patch_size=16, in_chans=3,
                 embed_dim=768, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize decoder position embedding
        decoder_pos_embed = self._get_pos_embed(self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, embed_dim):
        grid_size = int(math.sqrt(self.decoder_pos_embed.shape[1] - 1))
        pos_embed = self._get_2d_sincos_pos_embed(embed_dim, grid_size)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=True):
        grid_h = grid_w = grid_size
        grid_h, grid_w = torch.meshgrid(
            torch.arange(grid_h), torch.arange(grid_w), indexing='ij'
        )
        grid = torch.stack([grid_h, grid_w], dim=-1).float()
        grid = grid.reshape([grid_h.shape[0] * grid_h.shape[1], 2])

        embed_dim = embed_dim // 2
        omega = torch.arange(embed_dim).float() / embed_dim
        omega = 1. / (10000**omega)

        out = torch.einsum('m,d->md', grid.flatten(), omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        pos_embed = torch.cat([emb_sin, emb_cos], dim=1)
        if cls_token:
            pos_embed = torch.cat([torch.zeros([1, pos_embed.shape[1]]), pos_embed], dim=0)
        return pos_embed

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token
        return x

class CMAE(pl.LightningModule):
    """Contrastive Masked Autoencoder main model adapted for Hydro dataset"""
    def __init__(self, img_size=256, patch_size=16, in_chans=12, embed_dim=768,
                 depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False, temperature=0.07):
        super().__init__()
        self.save_hyperparameters()

        # Online encoder
        self.encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio)

        # Target encoder (with EMA updates)
        self.target_encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Decoders for pixel reconstruction and feature prediction
        self.pixel_decoder = MAEDecoder(
            num_patches=(img_size // patch_size) ** 2,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.feature_decoder = MAEDecoder(
            num_patches=(img_size // patch_size) ** 2,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=2,  # Shallower for feature decoder
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        # Projector and predictor for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 256)
        )

        self.predictor = nn.Sequential(
            nn.Linear(256, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 256)
        )

        self.norm_pix_loss = norm_pix_loss
        self.temperature = temperature
        self.momentum = 0.996  # EMA momentum for target network


    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.hparams.patch_size
        h = w = self.hparams.img_size // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_train(self, imgs, aug_imgs):
        """Forward pass for training"""
        # Online encoder path
        latent_s, mask_s, ids_restore_s = self.encoder(imgs)

        # Target encoder path (no gradients)
        with torch.no_grad():
            latent_t, mask_t, ids_restore_t = self.target_encoder(aug_imgs)
            latent_t = latent_t.detach()  # Stop gradients

        # Reconstruction path
        pred_pixel = self.pixel_decoder(latent_s, ids_restore_s)
        pred_feature = self.feature_decoder(latent_s, ids_restore_s)

        # Contrastive path
        proj_s = self.projector(torch.mean(pred_feature, dim=1, keepdim=True))
        proj_t = self.projector(torch.mean(latent_t[:, 1:, :], dim=1, keepdim=True))

        # Predictor
        pred_s = self.predictor(proj_s)

        return pred_pixel, mask_s, pred_s, proj_t

    def forward(self, imgs):
        """Forward pass for inference"""
        return self.encoder(imgs)

    def training_step(self, batch, batch_idx):
        """Lightning training step"""
        imgs, aug_imgs = batch['img'], batch['img_t']

        # Forward pass
        pred_pixel, mask_s, pred_s, proj_t = self.forward_train(imgs, aug_imgs)

        # Reconstruction loss
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        recon_loss = (pred_pixel - target) ** 2
        recon_loss = recon_loss.mean(dim=-1)
        recon_loss = (recon_loss * mask_s).sum() / mask_s.sum()

        # Contrastive loss
        pred_s = F.normalize(pred_s.squeeze(dim=1), dim=1, p=2)
        proj_t = F.normalize(proj_t.squeeze(dim=1), dim=1, p=2)

        # Gather targets from all GPUs if using distributed training
        if self.trainer.world_size > 1:
            proj_t_all = self.all_gather(proj_t)
            proj_t = proj_t_all.view(-1, proj_t.shape[-1])

        # Compute similarity and supervised contrastive loss
        sim = torch.matmul(pred_s, proj_t.T) / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        contra_loss = F.cross_entropy(sim, labels)

        # Combined loss
        loss = recon_loss + contra_loss

        # Log metrics
        self.log('train/recon_loss', recon_loss, sync_dist=True)
        self.log('train/contra_loss', contra_loss, sync_dist=True)
        self.log('train/loss', loss, sync_dist=True)

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        """Update target network with momentum encoder after each training step"""
        with torch.no_grad():
            for online_param, target_param in zip(self.encoder.parameters(),
                                                self.target_encoder.parameters()):
                target_param.data = target_param.data * self.momentum + \
                                  online_param.data * (1. - self.momentum)

    def configure_optimizers(self):
        """Configure optimizers for training"""
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': [], 'lr': self.hparams.lr, 'weight_decay': 0.05},  # Regular params
            {'params': [], 'lr': 0.0, 'weight_decay': 0.0}  # No weight decay params
        ]

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Skip LayerNorm and bias terms for weight decay
            if 'ln' in name or 'norm' in name or 'bias' in name or \
               'pos_embed' in name or 'cls_token' in name:
                param_groups[1]['params'].append(p)
            else:
                param_groups[0]['params'].append(p)

        optimizer = torch.optim.AdamW(param_groups)

        # Cosine learning rate schedule with warmup
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,  # 5% warmup
                anneal_strategy='cos',
                cycle_momentum=False
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]


