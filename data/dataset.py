import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T

from typing import Callable, Optional

import parameters as params
from utils import fftconv2d
from .PSFPrecomputer import PSFPrecomputer

class ImageDataset(Dataset):
    def __init__(
        self,
        image_dir         : str,
        psf_bank_path     : str = None,
        num_blurs_per_img : int = 1,
        z_range           : tuple = (params.z_near, params.z_far),
        transform         : Optional[Callable] = None,
        random_seed       : int = 42,
        verbose           : bool = False
    ):
        self.image_dir          = image_dir
        self.image_paths        = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.z_range            = z_range
        self.psf_bank_path      = psf_bank_path
        self.num_blurs_per_img  = num_blurs_per_img

        if verbose:
            print("ImageDataset created.")
            print(f"  Number of images      : {len(self.image_paths)}")
            print(f"  Z range               : {z_range}")
            print(f"  Blurs per image       : {num_blurs_per_img}")

        self.precomp = PSFPrecomputer()
        if self.psf_bank_path is not None:
            self.precomp.load_psf_bank(psf_bank_path)
            if verbose:
                print(f"  PSF loaded from       : {psf_bank_path}")
        else:
            self.precomp.compute_psf_bank(z_range=z_range, z_step=0.01)
            self.precomp.save_psf_bank(psf_bank_path)
            if verbose:
                print(f"  PSF computed to       : {psf_bank_path}")

        self.transform = transform
        self.random_seed = random_seed
        self.verbose = verbose

        self.channel_order = ['r', 'g', 'b']
        self.total_samples = len(self.image_paths) * num_blurs_per_img

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_idx, blur_idx = divmod(idx, self.num_blurs_per_img)
        img_path = self.image_paths[img_idx]
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)

        # Deterministic depth sampling
        seed = hash((img_idx, blur_idx, self.random_seed)) % (2**32)    # random seed for reproducibility
        rng = np.random.RandomState(seed)                               # random number generator
        z = np.round(rng.uniform(self.z_range[0], self.z_range[1]), 2)  # random depth in z_range

        # Get PSFs from precomputed bank
        psfs = self.precomp[z]  

        # Convolve each channel
        blurred_np = np.zeros_like(img_np)
        for i, ch in enumerate(self.channel_order):
            blurred_np[:, :, i] = fftconv2d(
                torch.tensor(img_np[:, :, i]), 
                torch.tensor(psfs[ch])
            ).real.numpy()

        if self.transform:
            blurred_np = self.transform(blurred_np)
            img_np = self.transform(img_np)

        return z, blurred_np, img_np  

if __name__ == "__main__":
    dataset = ImageDataset(
        image_dir="data/train",
        psf_bank_path="data/psf_bank.npz",
        num_blurs_per_img=3,
        verbose=True
    )
    z, blurred_tensor, sharp_tensor = dataset[0]
    print(f"Sample z: {z}, blurred shape: {blurred_tensor.shape}, sharp shape: {sharp_tensor.shape}")