import random
from PIL import ImageFilter
from torchvision.transforms import GaussianBlur as GB
import torch
import torchvision.transforms as T
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    def __init__(self, sigma_range=(0.1, 2.0), kernel_size=5):
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        gaussian_blur_transform = T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma_range)
        return gaussian_blur_transform(img)