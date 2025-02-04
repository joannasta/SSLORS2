import os
import torch
import json
import numpy as np
from torch import nn

from utils import time


def get_save_dir(args):
    if args.checkpoint is None:
        save_dir = f"{args.save_dir}{time.get_date_time()}-{get_run_name(args)}/"
    else:
        save_dir = f"{os.path.dirname(args.checkpoint)}/"
    return save_dir


def get_run_name(args):
    save_dir = f"{args.model}-{args.loss}-{args.mode}-{args.learning_rate}"
    return save_dir

# Copyright (c) OpenMMLab. All rights reserved.
import torch

def denormalize(original_images, reconstructed_images,masked_images,min_value,max_value):
    original_images = (original_images  * (max_value - min_value)) +  (min_value)
    reconstructed_images = (reconstructed_images  * (max_value - min_value)) +  (min_value)
    masked_images = (masked_images  * (max_value - min_value)) +  (min_value)
    return original_images,reconstructed_images,masked_images
        

def extract_bgr(original_images, reconstructed_images,masked_images):
    return original_images[1:4, :, :],reconstructed_images[ 1:4, :, :],masked_images[ 1:4, :, :]

def bgr2rgb(original_images, reconstructed_images,masked_images):
    original_rgb = original_images[[2, 1, 0], :, :] 
    reconstructed_rgb = reconstructed_images[ [2, 1, 0], :, :] 
    masked_rgb = masked_images[[2, 1, 0], :, :] 

    return original_rgb,reconstructed_rgb,masked_rgb


def image_clipping(original_rgb, reconstructed_rgb, masked_rgb):
    
    print("original rgb",type(original_rgb),original_rgb.shape)
    
    original_rgb = original_rgb.detach().cpu().numpy()
    reconstructed_rgb = reconstructed_rgb.detach().cpu().numpy()
    masked_rgb = masked_rgb.detach().cpu().numpy()


    # Clip values to the 99th percentile for better contrast
    original_rgb = np.clip(original_rgb, 0, np.percentile(original_rgb, 99))
    original_rgb = original_rgb / original_rgb.max()  # Normalize to [0, 1]
    
    reconstructed_rgb = np.clip(reconstructed_rgb, 0, np.percentile(reconstructed_rgb, 99))
    reconstructed_rgb = reconstructed_rgb / reconstructed_rgb.max()  # Normalize to [0, 1]
    
    masked_rgb = np.clip(masked_rgb, 0, np.percentile(masked_rgb, 99))
    masked_rgb = masked_rgb / masked_rgb.max()  # Normalize to [0, 1]
    
    return original_rgb, reconstructed_rgb, masked_rgb

def build_2d_sincos_position_embedding(patches_resolution,
                                       embed_dims,
                                       temperature=10000.,
                                       cls_token=False):
    """The function is to build position embedding for model to obtain the
    position information of the image patches."""

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
        dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb



class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not torch.isnan(val):
            val = val.data
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

