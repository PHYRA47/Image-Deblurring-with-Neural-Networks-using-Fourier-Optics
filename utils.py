import yaml
import torch 
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from torchmetrics.functional import structural_similarity_index_measure as ssim

# === Data ===

def fftconv2d(image, kernel, mode="same"):
    """
    2D convolution using FFT.
    image and kernel should have same ndim.

    the Fourier transform of a convolution equals the multiplication of the Fourier transforms of the convolved signals.
    A similar property holds for the Laplace and z-transforms.
    However, it does not, in general, hold for the discrete Fourier transform.
    Instead, multiplication of discrete Fourier transforms corresponds to the 'circular convolution' of the corresponding time-domain signals.
    In order to compute the linear convolution of two sequences using the DFT, the sequences must be zero-padded to a length equal to the sum of the lengths of the two sequences minus one, i.e. N+M-1.
    """


    imH, imW = image.shape[-2], image.shape[-1]
    kH, kW = kernel.shape[-2], kernel.shape[-1]
    # zero-padded to a length equal to the sum of the lengths of the two sequences minus one, i.e. N+M-1
    size = (imH + kH - 1, imW + kW - 1)

    Fimage = torch.fft.fft2(image, s=size)
    Fkernel = torch.fft.fft2(kernel, s=size)

    Fconv = Fimage * Fkernel

    conv = torch.fft.ifft2(Fconv)

    if mode == "same":
        conv = conv[..., (kH // 2) : imH + (kH // 2), (kW // 2) : imW + (kW // 2)]
    if mode == "valid":
        conv = conv[..., kH - 1 : imH, kW - 1 : imW]
    # otherwise, full
    return conv

def show_blur_sharp_with_z(blurred_tensor, sharp_tensor, z=None):
    """
    Display blurred and sharp images side by side, with z in the title.
    Args:
        z: float, the depth value
        blurred_tensor: torch.Tensor, shape (C, H, W)
        sharp_tensor: torch.Tensor, shape (C, H, W)
    """
    # Convert tensors to numpy arrays and move channel to last dimension
    blurred_img = blurred_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    sharp_img = sharp_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    # Clip values to [0,1] for display
    blurred_img = blurred_img.clip(0, 1)
    sharp_img = sharp_img.clip(0, 1)
    # Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(blurred_img)
    if z is not None:
        plt.title(f"Blurred (z={z:.2f}m)")
    else:
        plt.title("Blurred")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(sharp_img)
    plt.title("Sharp (Ground Truth)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_config(yaml_path: str) -> SimpleNamespace:
    """
    Load a YAML config file and convert it to a nested SimpleNamespace object.
    Allows dot-notation access to config values.
    """
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(i) for i in d]
        else:
            return d

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return dict_to_namespace(config_dict)
