import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image import PeakSignalNoiseRatio as psnr
import torchvision

from model import get_model
from data.loader import get_dataloader
from utils import load_config

from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"

# ─── Train Epoch ─────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, epoch, device, loss_fn, 
                writer=None, log_frequency=10):
    model.train()
    running_loss = 0.0
    num_batch = 0
    base_step = epoch * len(dataloader)

    for batch_idx, (_, blurred, sharp) in enumerate(tqdm(dataloader, desc=f"Train {epoch+1}")):
        blurred, sharp = blurred.to(device), sharp.to(device)
        optimizer.zero_grad()
        output = model(blurred)
        loss = loss_fn(output, sharp)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        num_batch += 1

        if writer and (batch_idx + 1) % log_frequency == 0:
            step = base_step + batch_idx + 1
            writer.add_scalar(f"Train/Learning_Rate", optimizer.param_groups[0]['lr'], step)
            # writer.add_scalar(f"Train/Batch_Loss", loss.item(), step)

    epoch_loss = running_loss / num_batch if num_batch > 0 else 0.0
    if writer:
        writer.add_scalar(f"Train/Epoch_Loss", epoch_loss, epoch)
        
    print(f"  ➤ Train Loss : {epoch_loss:.8f}")
    return epoch_loss

# ─── Validate Epoch ──────────────────────────────────────────────────────
def validate_epoch(model, dataloader, epoch, device, loss_fn, metrics_fn, writer=None):
    model.eval()
    running_loss = 0.0
    num_batch = 0

    with torch.no_grad():
        for batch_idx, (_, blurred, sharp) in enumerate(tqdm(dataloader, desc=f"Valid {epoch+1}")):
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)
            loss = loss_fn(output, sharp)
            running_loss += loss.item()
            num_batch += 1

            for name, metric in metrics_fn.items():
                metric.update(output, sharp)

            if writer and batch_idx == 0:
                n = min(6, blurred.size(0))  # log up to 6 samples
                grid_blurred = torchvision.utils.make_grid(blurred[:n], nrow=n, normalize=True)
                grid_sharp = torchvision.utils.make_grid(sharp[:n], nrow=n, normalize=True)
                grid_output = torchvision.utils.make_grid(output[:n], nrow=n, normalize=True)

                grid_concat = torch.cat([grid_blurred, grid_output, grid_sharp], dim=1)
                writer.add_image("Validation/Examples", grid_concat, epoch)

    metric_results = {}
    for name, metric in metrics_fn.items():
        metric_results[name] = metric.compute()
        metric.reset() # Reset for next epoch

    epoch_loss = running_loss / num_batch if num_batch > 0 else 0.0
    if writer:
        writer.add_scalar("Val/Epoch_Loss", epoch_loss, epoch)
        writer.add_scalar("Val/PSNR", metric_results['psnr'], epoch)
        writer.add_scalar("Val/SSIM", metric_results['ssim'], epoch)


    print(f"  ➤ Val Loss   : {epoch_loss:.8f}")
    print(f"  ➤ Val PSNR   : {metric_results['psnr']:.8f}")
    print(f"  ➤ Val SSIM   : {metric_results['ssim']:.8f}")
    return epoch_loss, metric_results

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    cfg = load_config(CONFIG_PATH)

    # ─── Setup Experiment Directory ─────────────────────────────────────
    if cfg.EXPERIMENT.exp_name is None:
        cfg.EXPERIMENT.exp_name = cfg.MODEL.name.upper()
        if cfg.EXPERIMENT.auto_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime('%m%d_%H%M%S')
            cfg.EXPERIMENT.exp_name += f"_{timestamp}"

    if cfg.LOGGING.verbose:
        print(f"[INFO] Experiment Name  : {cfg.EXPERIMENT.exp_name}")
        print(f"[INFO] Save Checkpoints : {cfg.LOGGING.save_checkpoints}")
        print(f"[INFO] Base Directory   : {cfg.EXPERIMENT.base_dir}")

    exp_dir = os.path.join(cfg.EXPERIMENT.base_dir, cfg.EXPERIMENT.exp_name)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    # ─── Device Setup ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device           : {device}")

    # ─── Set Seed ───────────────────────────────────────────────────────
    if cfg.EXPERIMENT.seed is not None:
        torch.manual_seed(cfg.EXPERIMENT.seed)
        print(f"[INFO] Seed             : {cfg.EXPERIMENT.seed}")

    # ─── Back Up Config ─────────────────────────────────────────────────
    if cfg.LOGGING.save_config:
        config_backup = os.path.join(exp_dir, "config.yaml")
        shutil.copy(CONFIG_PATH, config_backup)
        print(f"[INFO] Config saved to  : {config_backup}")

    # ─── TensorBoard Writer ─────────────────────────────────────────────
    writer = None
    if cfg.LOGGING.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard"))
        print(f"[INFO] TensorBoard      : {writer.log_dir}")

    # ─── Load Datasets ──────────────────────────────────────────────────
    dataloaders = get_dataloader(
        train_dir=cfg.DATA.train_dir,
        test_dir=cfg.DATA.test_dir,
        psf_bank_path=cfg.DATA.psf_bank_path,
        train_batch_size=cfg.TRAIN.train_batch_size,
        test_batch_size=cfg.TRAIN.test_batch_size,
        shuffle=True,
        verbose=False,
        num_workers=cfg.DATA.num_workers)

    train_loader, val_loader = dataloaders['train'], dataloaders['test']
    if cfg.LOGGING.verbose:
        print(f"[INFO] Train Samples    : {len(train_loader) * cfg.TRAIN.train_batch_size}")
        print(f"[INFO] Test Samples     : {len(val_loader) * cfg.TRAIN.test_batch_size}")
        print(f"[INFO] Train Batch Size : {cfg.TRAIN.train_batch_size}")
        print(f"[INFO] Test Batch Size  : {cfg.TRAIN.test_batch_size}")

    # ─── Model ──────────────────────────────────────────────────────────
    model = get_model(cfg.MODEL.name).to(device)
    if cfg.LOGGING.verbose:
        print(f"[INFO] Model            : {cfg.MODEL.name.upper()}")
        print(f"[INFO] Number of params : {sum(p.numel() for p in model.parameters())}")

    # ─── Loss and Metrics ───────────────────────────────────────────────
    loss_fn = nn.MSELoss().to(device)
    metrics_fn = {
        "psnr": psnr().to(device),
        "ssim": ssim().to(device)
    }

    # ─── Optimizer ──────────────────────────────────────────────────────
    if cfg.OPTIMIZER.type.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.OPTIMIZER.base_lr,
            weight_decay=cfg.OPTIMIZER.weight_decay,
            betas=(cfg.OPTIMIZER.beta1, cfg.OPTIMIZER.beta2)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.OPTIMIZER.type}")
    print(f"[INFO] Using optimizer  : {cfg.OPTIMIZER.type}")

    # ─── Scheduler ──────────────────────────────────────────────────────
    if cfg.SCHEDULER.type.lower() == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.SCHEDULER.step_size, 
            gamma=cfg.SCHEDULER.gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.SCHEDULER.type}")
    if cfg.LOGGING.verbose:
        print(f"[INFO] Using scheduler  : {cfg.SCHEDULER.type}")

    # ─── Training Loop ──────────────────────────────────────────────────
    best_psnr = float(0.0); best_psnr_epoch = None
    best_ssim = float(0.0); best_ssim_epoch = None

    print(f"[INFO] Training Start   : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_start_time = time.time()

    for epoch in range(cfg.TRAIN.epochs):
        print(f"[INFO] Epoch {epoch + 1}/{cfg.TRAIN.epochs}")
        epoch_start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, device,
            loss_fn, writer, log_frequency=cfg.LOGGING.print_frequency
        )

        if epoch % cfg.TRAIN.test_every == 0 or epoch == cfg.TRAIN.epochs - 1:
            val_loss, val_metrics = validate_epoch(
                model, val_loader, epoch, device, loss_fn, metrics_fn, writer
            )
            
            is_best_psnr = False
            is_best_ssim = False
            if epoch + 1 >= cfg.LOGGING.save_best_after:
                current_psnr = val_metrics['psnr']
                current_ssim = val_metrics['ssim']

                is_best_psnr = current_psnr > best_psnr
                is_best_ssim = current_ssim > best_ssim

                if is_best_psnr:
                    best_psnr = current_psnr
                    best_psnr_epoch = epoch + 1
                    print(f"⭐️ New best PSNR: {best_psnr:.2f} (epoch {best_psnr_epoch})")

                if is_best_ssim:
                    best_ssim = current_ssim
                    best_ssim_epoch = epoch + 1
                    print(f"⭐️ New best SSIM: {best_ssim:.4f} (epoch {best_ssim_epoch})")

            if cfg.LOGGING.save_checkpoints:
                ckpt_state = {
                    'model_name':           cfg.MODEL.name,
                    'epoch':                epoch + 1,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss':           train_loss,
                    'val_loss':             val_loss,
                    'val_metrics':          val_metrics,
                    'config':               cfg
                }
                ckpt_dir = os.path.join(exp_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)

                last_path = os.path.join(ckpt_dir, 'last.pth')
                torch.save(ckpt_state, last_path)

                if epoch + 1 >= cfg.LOGGING.save_best_after:
                    if is_best_psnr:
                        best_psnr_path = os.path.join(ckpt_dir, 'best_psnr.pth')
                        torch.save(ckpt_state, best_psnr_path)
                        print(f" ✅ Best PSNR checkpoint saved.")
                    if is_best_ssim:
                        best_ssim_path = os.path.join(ckpt_dir, 'best_ssim.pth')
                        torch.save(ckpt_state, best_ssim_path)
                        print(f" ✅ Best SSIM checkpoint saved.")

                if cfg.LOGGING.verbose:
                    print(f" ✅ Latest checkpoint saved.")
        
        if writer and cfg.TENSORBOARD.log_timing:
            writer.add_scalar('Timing/Epoch_Duration', time.time() - epoch_start_time, epoch)
            
    if writer and cfg.TENSORBOARD.log_hparams:
        hparam = {
            'model': cfg.MODEL.name,
            'batch_size': cfg.TRAIN.train_batch_size,
            'lr': cfg.OPTIMIZER.base_lr,
            'weight_decay': cfg.OPTIMIZER.weight_decay,
        }
        metrics = {'PSNR': best_psnr, 'SSIM': best_ssim}
        writer.add_hparams(hparam, metrics)
        writer.close()
    
    print(f"[INFO] Training completed in {time.time() - train_start_time:.2f} seconds")
    print(f"[INFO] Best PSNR: {best_psnr:.2f} (epoch {best_psnr_epoch})")
    print(f"[INFO] Best SSIM: {best_ssim:.4f} (epoch {best_ssim_epoch})")

if __name__ == "__main__":
    main()