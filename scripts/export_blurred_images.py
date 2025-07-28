import os
import torch
from torchvision.utils import save_image
from dataset.main import ImageDataset

def export_blurred_images(
    image_dir: str,
    psf_bank_path: str,
    output_dir: str,
    num_blurs_per_img: int = 1,
    save_sharp: bool = False,
    verbose: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    if save_sharp:
        os.makedirs(os.path.join(output_dir, "sharp"), exist_ok=True)

    dataset = ImageDataset(
        image_dir=image_dir,
        psf_bank_path=psf_bank_path,
        num_blurs_per_img=num_blurs_per_img,
        verbose=verbose
    )

    for idx in range(len(dataset)):
        z, blurred_tensor, sharp_tensor = dataset[idx]
        base_name = os.path.splitext(os.path.basename(dataset.image_paths[idx // num_blurs_per_img]))[0]

        blur_out_path = os.path.join(output_dir, f"{base_name}_z{z:.2f}_blurred.png")
        save_image(blurred_tensor, blur_out_path)

        if save_sharp:
            sharp_out_path = os.path.join(output_dir, "sharp", f"{base_name}_sharp.png")
            save_image(sharp_tensor, sharp_out_path)

        if verbose:
            print(f"[{idx}] Saved blurred: {blur_out_path}")

    print("✅ All blurred images exported.")


if __name__ == "__main__":
    export_blurred_images(
        image_dir="dataset/test",
        psf_bank_path="dataset/psf_bank.npz",
        output_dir="exported_blurred",
        num_blurs_per_img=3,
        save_sharp=True,
        verbose=True
    )