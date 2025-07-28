#!/usr/bin/env python3
# Simple batch inference script for deblurring model

import os
import sys
import time
import csv
import random
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import get_model
from data.loader import get_dataloader
from utils import load_config
from tqdm import tqdm


# --- Configuration ---
EXPERIMENT_PATH      = "logs/zrange1/DNCNN_0727_170005"
CHECKPOINT_NAME      = "best_psnr.pth" # best_ssim.pth, best_psnr.pth
INFERENCE_INPUT_DIR  = "data/test"
BATCH_SIZE           = 8
SAVE_DEBLURRED       = True
SAVE_CSV             = True
SAVE_SAMPLES_PLOT    = True
NUM_SAMPLES_PLOT     = 5
RESULT_SUBDIR        = "inference_results" # Directory to save results
DEVICE               = "cpu"  # "cuda" or "cpu"
VERBOSE              = True


def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    mse = torch.mean((pred - target) ** 2).item()
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
    with torch.no_grad():
        psnr_val = psnr_fn(pred.unsqueeze(0), target.unsqueeze(0)).item()
        ssim_val = ssim_fn(pred.unsqueeze(0), target.unsqueeze(0)).item()
    return mse, psnr_val, ssim_val


def plot_random_samples(df: pd.DataFrame, loader, model, device, results_dir):
    import matplotlib.pyplot as plt
    samples = df.sample(min(len(df), NUM_SAMPLES_PLOT))
    plt.figure(figsize=(12, 4 * len(samples)))
    for idx, row in enumerate(samples.itertuples(), start=1):
        batch_idx = int(row.batch_idx)
        i = int(row.index_in_batch)
        z = row.z
        _, blurred, sharp = loader.dataset[batch_idx * loader.batch_size + i]
        blurred = blurred.cpu()
        sharp = sharp.cpu()
        with torch.no_grad():
            pred = model(blurred.unsqueeze(0).to(device)).cpu().squeeze(0)
        mse, psnr_v, ssim_v = row.mse, row.psnr, row.ssim
        b_img = np.clip(np.transpose(blurred.numpy(), (1, 2, 0)), 0.0, 1.0)
        p_img = np.clip(np.transpose(pred.numpy(),    (1, 2, 0)), 0.0, 1.0)
        s_img = np.clip(np.transpose(sharp.numpy(),   (1, 2, 0)), 0.0, 1.0)
        plt.subplot(len(samples), 3, (idx - 1) * 3 + 1)
        plt.imshow(b_img)
        plt.title(f"Blurred (z={z})")
        plt.axis('off')
        plt.subplot(len(samples), 3, (idx - 1) * 3 + 2)
        plt.imshow(p_img)
        plt.title(f"Deblurred\nMSE={mse:.4f}  PSNR={psnr_v:.2f}  SSIM={ssim_v:.3f}")
        plt.axis('off')
        plt.subplot(len(samples), 3, (idx - 1) * 3 + 3)
        plt.imshow(s_img)
        plt.title("Ground Truth")
        plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"sample_grid_{datetime.now().strftime('%H%M%S')}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    if VERBOSE:
        print(f"Sample grid saved to: {out_path}")


def main():

    print(f"Experiment      : {EXPERIMENT_PATH}")
    print(f"Checkpoint      : {CHECKPOINT_NAME}")
    print(f"Inference Dir   : {INFERENCE_INPUT_DIR}\n")

    config_path = os.path.join(EXPERIMENT_PATH, "config.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)
    cfg = load_config(config_path)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if VERBOSE:
        print(f"Loading inference DataLoader on '{INFERENCE_INPUT_DIR}' (batch_size={BATCH_SIZE})...")
    loader = get_dataloader(
        train_dir=INFERENCE_INPUT_DIR,
        test_dir=INFERENCE_INPUT_DIR,
        psf_bank_path=cfg.DATA.psf_bank_path,
        train_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        verbose=VERBOSE
    )
    test_loader = loader['test']
    print(f"Loaded {len(test_loader.dataset)} samples ({len(test_loader)} batches)\n")

    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    results_dir = os.path.join(EXPERIMENT_PATH, RESULT_SUBDIR, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    if SAVE_DEBLURRED:
        deblur_dir = os.path.join(results_dir, 'deblurred_images')
        os.makedirs(deblur_dir, exist_ok=True)
    if SAVE_CSV:
        csv_path = os.path.join(results_dir, 'metrics.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_name','batch_idx','index_in_batch','z','mse','psnr','ssim'])

    model = get_model(cfg.MODEL.name).to(device)
    ckpt_path = os.path.join(EXPERIMENT_PATH, 'checkpoints', CHECKPOINT_NAME)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if VERBOSE:
        print(f"Loading weights from: {ckpt_path}")
    start = time.time()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded in {time.time() - start:.2f}s\n")

    print(f"Running inference on {len(test_loader.dataset)} images...")
    count = 0
    start_all = time.time()
    for batch_idx, (z, blurred, sharp) in enumerate(tqdm(test_loader, desc="Inference", unit="batch")):
        blurred, sharp = blurred.to(device), sharp.to(device)
        with torch.no_grad():
            output = model(blurred)
        output_cpu = output.cpu()
        sharp_cpu  = sharp.cpu()
        z_vals     = z.cpu().numpy()
        for i in range(output_cpu.size(0)):
            img_pred = output_cpu[i]
            img_sharp= sharp_cpu[i]
            z_val    = float(z_vals[i])
            mse_v, psnr_v, ssim_v = compute_metrics(img_pred, img_sharp)
            if SAVE_DEBLURRED:
                fname = f"deblur_{batch_idx:04d}_{i}.png"
                save_image(img_pred, os.path.join(deblur_dir, fname))
                image_name = fname
            else:
                image_name = f"idx_{batch_idx:04d}_{i}"
            if SAVE_CSV:
                csv_writer.writerow([image_name, batch_idx, i, z_val,
                                     f"{mse_v:.6f}", f"{psnr_v:.3f}", f"{ssim_v:.4f}"])
            count += 1
    elapsed = time.time() - start_all
    print(f"\nProcessed {count} images in {elapsed:.2f}s ({(elapsed/count):.3f}s per image)")
    if SAVE_CSV:
        csv_file.close()
        print(f"Metrics CSV saved to: {csv_path}")
    if SAVE_SAMPLES_PLOT and SAVE_CSV:
        df = pd.read_csv(csv_path)
        plot_random_samples(df, test_loader, model, device, results_dir)
    print("\nInference completed.")
    print(f"Check results under: {results_dir}\n")

if __name__ == '__main__':
    main()