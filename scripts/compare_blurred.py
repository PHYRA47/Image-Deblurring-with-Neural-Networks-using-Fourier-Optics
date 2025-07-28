import numpy as np
import torch
import cv2
import torch
import matplotlib.pyplot as plt

from util import fftconv2d
from dataset.PSFPrecomputer import PSFPrecomputer

# ==== 1. Load Ground Truth Image ====
image_path = "/Users/mii/Documents/DSII/_instruction/parrots256.png"
image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)        # BGR format
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)      # Convert to RGB
image_rgb = image_rgb.astype(np.float32) / 255.0            # Normalize to [0,1]

# ==== 2. Initialize PSFPrecomputer and load PSF bank ====
psf_precomputer = PSFPrecomputer()
# psf_precomputer.compute_psf_bank(z_range=(1.84, 2.20), z_step=0.01)
# psf_precomputer.save_psf_bank("dataset/psf_bank.npz")
psf_precomputer.load_psf_bank("dataset/psf_bank.npz")

# === 3. Choose a z and get PSFs ===
z = 2.00
psfs = psf_precomputer[z]  # dict with 'r', 'g', 'b'

# === 4. Convolve each channel ===
blurred = np.zeros_like(image_rgb)
for i, ch in enumerate(['r', 'g', 'b']):
    blurred[:, :, i] = fftconv2d(
        torch.from_numpy(image_rgb[:, :, i]), 
        torch.from_numpy(psfs[ch])
    ).numpy()

# === 5. Display ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Blurred (z = {z:.2f}m)")
plt.imshow(blurred)
plt.axis("off")
plt.tight_layout()
plt.show()

# from terminal in the root directory:
# python -m scripts.compare_blurred