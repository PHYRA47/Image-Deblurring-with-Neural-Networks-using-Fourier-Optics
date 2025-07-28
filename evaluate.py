# Evaluation function
import torch
import numpy as np
from model import get_model
from data.PSFPrecomputer import PSFPrecomputer
from utils import fftconv2d
from PIL import Image
import torchvision.transforms as transforms

def evaluate(sharp_image: np.ndarray, z: float, ckpt_path: str) -> np.ndarray:
    # sharp_image: the ground truth image to be evaluated with a shape of (H, W, 3)
    # z: the object depth in meters
    # returns: deblurred image with a shape of (H, W, 3) through neural network
    computer = PSFPrecomputer()
    psfs = computer[z]

    blurred = np.zeros_like(sharp_image) # default dtype is 
    for i, ch in enumerate(['r', 'g', 'b']):
        blurred[:, :, i] = fftconv2d(
            torch.tensor(sharp_image[:, :, i]),
            torch.tensor(psfs[ch])
        ).real

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = get_model(ckpt['config'].MODEL.name).to(device)
    print(f'Model loaded: {ckpt["config"].MODEL.name}')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    transform = transforms.ToTensor()
    blurred_tensor = transform(blurred).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(blurred_tensor)
    deblurred_image = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    deblurred_image = np.clip(deblurred_image, 0, 1)

    # Compute metrics
    from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
    sharp_tensor = transform(sharp_image).unsqueeze(0)
    deblurred_tensor = transform(deblurred_image).unsqueeze(0)
    mse = np.mean((deblurred_image - sharp_image) ** 2)
    psnr = peak_signal_noise_ratio(deblurred_tensor, sharp_tensor, data_range=1.0).item()
    ssim = structural_similarity_index_measure(deblurred_tensor, sharp_tensor, data_range=1.0).item()
    print(f"MSE: {mse:.6f}, PSNR: {psnr:.3f}, SSIM: {ssim:.4f}")

    return deblurred_image

# Example usage
if __name__ == "__main__":
    SHARP_IMG_PATH = "/home/denegasf/repo/negasa-fromsa-teshome-msc-thesis/src/DSII/_instruction/parrots256.png"
    CKPT_PATH = "/home/denegasf/repo/negasa-fromsa-teshome-msc-thesis/src/DSII/logs/zrange1/UNET_0727_165934/checkpoints/best_psnr.pth"
    z = 2.0 
    sharp_image = np.array(Image.open(SHARP_IMG_PATH)).astype(np.float32) / 255.0

    deblurred_image = evaluate(sharp_image, z, CKPT_PATH)
    print(f"Deblurred image shape: {deblurred_image.shape}")
    
    # Save or visualize deblurred_image as needed
    import matplotlib.pyplot as plt
    deblurred_image_clipped = np.clip(deblurred_image, 0, 1)
    plt.imshow(deblurred_image_clipped)
    plt.axis("off")
    plt.show()
    # Optionally save the image
    plt.imsave("deblurred_result.png", deblurred_image_clipped)
