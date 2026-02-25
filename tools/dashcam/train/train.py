#!/usr/bin/env python3
"""Training script for single-camera driving vision model.

Usage:
  python tools/dashcam/train/train.py \
    --data-dirs data/training/carla_001 data/training/carla_002 \
    --epochs 100 --batch-size 16 --lr 1e-3

  # Early stopping: stop if val_loss doesn't improve for 15 epochs
  python tools/dashcam/train/train.py \
    --data-dirs data/training/carla_001 \
    --early-stop 15

  # Monitor training from another terminal:
  python tools/dashcam/train/monitor.py checkpoints/training_log.csv

Flow:
  1. Build DrivingDataset + random split (95/5)
  2. DataLoader (num_workers=4, pin_memory)
  3. DrivingVisionModel -> CUDA
  4. AdamW + CosineAnnealingLR (warmup 5 epochs)
  5. Training loop with gradient clipping, logging, validation, checkpointing
  6. CSV metrics log written per epoch for monitoring
  7. Optional early stopping based on val_loss patience
  8. Auto-export ONNX at end of training
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from openpilot.tools.dashcam.train.config import ModelConfig, TrainConfig
from openpilot.tools.dashcam.train.dataset import DrivingDataset
from openpilot.tools.dashcam.train.losses import DrivingLoss
from openpilot.tools.dashcam.train.model import DrivingVisionModel


CSV_COLUMNS = ['epoch', 'lr', 'train_loss', 'val_loss', 'lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose', 'road_transform', 'time_s']


def get_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
  """Cosine annealing with linear warmup."""
  warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=cfg.warmup_epochs)
  cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs)
  return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])


def train_one_epoch(
  model: nn.Module, loader: DataLoader, criterion: DrivingLoss, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, cfg: TrainConfig
) -> float:
  model.train()
  total_loss = 0.0
  n_batches = 0

  for batch_idx, (inputs, targets) in enumerate(loader):
    inputs = inputs.to(device, non_blocking=True)
    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

    optimizer.zero_grad()
    preds = model(inputs)
    loss, sub_losses = criterion(preds, targets)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    total_loss += loss.item()
    n_batches += 1

    if (batch_idx + 1) % cfg.log_every == 0:
      sub_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in sub_losses.items())
      print(f"  [E{epoch + 1} B{batch_idx + 1}/{len(loader)}] loss={loss.item():.4f} | {sub_str}")

  return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: DrivingLoss, device: torch.device) -> tuple[float, dict[str, float]]:
  model.eval()
  total_loss = 0.0
  sub_totals: dict[str, float] = {}
  n_batches = 0

  for inputs, targets in loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

    preds = model(inputs)
    loss, sub_losses = criterion(preds, targets)

    total_loss += loss.item()
    for k, v in sub_losses.items():
      sub_totals[k] = sub_totals.get(k, 0.0) + v.item()
    n_batches += 1

  n = max(n_batches, 1)
  avg_subs = {k: v / n for k, v in sub_totals.items()}
  return total_loss / n, avg_subs


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, path: str):
  torch.save(
    {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'val_loss': val_loss,
    },
    path,
  )
  print(f"  Checkpoint saved: {path}")


class CSVLogger:
  """Append-mode CSV logger for training metrics."""

  def __init__(self, path: str, columns: list[str]):
    self.path = path
    self.columns = columns
    write_header = not os.path.exists(path)
    self.file = open(path, 'a', newline='')
    self.writer = csv.DictWriter(self.file, fieldnames=columns)
    if write_header:
      self.writer.writeheader()
      self.file.flush()

  def log(self, row: dict):
    self.writer.writerow(row)
    self.file.flush()

  def close(self):
    self.file.close()


class EarlyStopping:
  """Stop training when val_loss doesn't improve for `patience` epochs."""

  def __init__(self, patience: int, min_delta: float = 0.0):
    self.patience = patience
    self.min_delta = min_delta
    self.best_loss = float('inf')
    self.wait = 0

  def step(self, val_loss: float) -> bool:
    """Returns True if training should stop."""
    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.wait = 0
    else:
      self.wait += 1
    return self.wait >= self.patience

  def status(self) -> str:
    return f"best={self.best_loss:.4f}, no_improve={self.wait}/{self.patience}"


def main():
  parser = argparse.ArgumentParser(description='Train driving vision model')
  parser.add_argument('--data-dirs', nargs='+', required=True, help='Directories containing NPZ training data')
  parser.add_argument('--output-dir', default='checkpoints', help='Output directory for checkpoints')
  parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
  parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
  parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
  parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
  parser.add_argument('--no-export', action='store_true', help='Skip ONNX export after training')
  parser.add_argument('--early-stop', type=int, default=0, help='Early stopping patience (0=disabled)')
  parser.add_argument('--min-delta', type=float, default=0.001, help='Minimum val_loss improvement to count as progress (default: 0.001)')
  args = parser.parse_args()

  model_cfg = ModelConfig()
  train_cfg = TrainConfig()
  if args.epochs is not None:
    train_cfg.epochs = args.epochs
  if args.batch_size is not None:
    train_cfg.batch_size = args.batch_size
  if args.lr is not None:
    train_cfg.lr = args.lr

  os.makedirs(args.output_dir, exist_ok=True)

  # Dataset
  print(f"Loading data from: {args.data_dirs}")
  dataset = DrivingDataset(args.data_dirs)
  print(f"Total samples: {len(dataset)}")

  val_size = max(1, int(len(dataset) * train_cfg.val_split))
  train_size = len(dataset) - val_size
  train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

  train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True, drop_last=True)
  val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, pin_memory=True)

  print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")

  # Model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = DrivingVisionModel(model_cfg).to(device)
  n_params = sum(p.numel() for p in model.parameters())
  print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

  # Optimizer + scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
  scheduler = get_lr_scheduler(optimizer, train_cfg)
  criterion = DrivingLoss(model_cfg, train_cfg)

  start_epoch = 0
  best_val_loss = float('inf')

  # Resume
  if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt.get('val_loss', float('inf'))
    print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.4f}")

  # CSV logger
  csv_path = os.path.join(args.output_dir, "training_log.csv")
  csv_logger = CSVLogger(csv_path, CSV_COLUMNS)
  print(f"Metrics log: {csv_path}")

  # Early stopping
  early_stop = EarlyStopping(args.early_stop, args.min_delta) if args.early_stop > 0 else None
  if early_stop:
    print(f"Early stopping: patience={args.early_stop} epochs, min_delta={args.min_delta}")

  # Training loop
  print(f"\nStarting training: {train_cfg.epochs} epochs, lr={train_cfg.lr}, bs={train_cfg.batch_size}")
  for epoch in range(start_epoch, train_cfg.epochs):
    t0 = time.monotonic()
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, train_cfg)
    val_loss, val_subs = validate(model, val_loader, criterion, device)
    scheduler.step()

    dt = time.monotonic() - t0
    lr = optimizer.param_groups[0]['lr']
    val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_subs.items())
    es_str = f" | early_stop: {early_stop.status()}" if early_stop else ""
    print(f"Epoch {epoch + 1}/{train_cfg.epochs} ({dt:.1f}s) lr={lr:.6f} train={train_loss:.4f} val={val_loss:.4f} | {val_str}{es_str}")

    # Log to CSV
    csv_logger.log(
      {
        'epoch': epoch + 1,
        'lr': f"{lr:.8f}",
        'train_loss': f"{train_loss:.6f}",
        'val_loss': f"{val_loss:.6f}",
        **{k: f"{v:.6f}" for k, v in val_subs.items()},
        'time_s': f"{dt:.1f}",
      }
    )

    # Save periodic checkpoint
    if (epoch + 1) % train_cfg.save_every == 0:
      save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt"))

    # Save best
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(args.output_dir, "best.pt"))

    # Early stopping check
    if early_stop and early_stop.step(val_loss):
      print(f"\nEarly stopping triggered at epoch {epoch + 1} (no improvement for {args.early_stop} epochs)")
      print(f"Best val_loss: {early_stop.best_loss:.4f}")
      break

  # Save final
  save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(args.output_dir, "final.pt"))

  csv_logger.close()

  # Export ONNX (fp32 + fp16)
  if not args.no_export:
    print("\nExporting ONNX...")
    from openpilot.tools.dashcam.train.export_onnx import convert_onnx_to_fp16, export_onnx

    onnx_path = os.path.join(args.output_dir, "driving_vision.onnx")
    export_onnx(model, onnx_path, device)
    fp16_path = os.path.join(args.output_dir, "driving_vision_fp16.onnx")
    convert_onnx_to_fp16(onnx_path, fp16_path)

  print("Training complete.")


if __name__ == "__main__":
  main()
