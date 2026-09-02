"""
train_cobb_regressor_matched.py

Receptive-field-matched version of train_cobb_regressor.py. Architecture
mirrors the original CobbRegressor exactly except that conv-block depth
and pool strides are scaled to handle non-square latent shapes.

Key design decisions preserved from the original:
  * Paired conv blocks (Conv->GN->GELU->Conv->GN->GELU->MaxPool(2)) per
    downsample level, NOT single stride-2 convs.
  * Channel schedule (32, 64, 128, 128). Extra blocks beyond original 4
    keep the 128 width — strict superset of the original.
  * Final AdaptiveAvgPool2d((6, 6)) normalises output to (6, 6) regardless
    of input shape. We don't need to bring everything to 3x3 — AAP handles
    spatial normalisation.
  * FC head preserves spatial structure: Flatten of 128x6x6 -> Linear(4608, 256)
    -> GELU -> Dropout(0.3) -> Linear(256, 3). Global average pooling
    before the FC throws away the WHERE-along-the-spine signal that Cobb
    regression needs.
  * Horizontal flip augmentation, dropout=0.3, lr=3e-4, weight_decay=1e-4,
    batch_size=32, no early stopping (200 epochs).

The only deviation for non-square latents: when one dimension is larger
than the other, MaxPool2d gets asymmetric stride/kernel like (2, 1) so
only the larger axis halves at that block. This keeps the feature map
proportional going into the AAP.

Usage:
    python train_cobb_regressor_matched.py \\
        --latents ./latents_pretrained_4x.npz \\
        --out ./cobb_matched_4x.npz \\
        --baseline-results ./cobb_pretrained_4x.npz \\
        --seed 42
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def compute_pool_schedule(H: int, W: int,
                            target_pre_aap: int = 6) -> list:
    """Decide on pool strides per block to bring (H, W) down toward
    (~target_pre_aap, ~target_pre_aap) before the AdaptiveAvgPool2d((6, 6))
    head.

    Strategy: stride-(2, 2) pools as long as both H and W are above the
    target. Once a dimension reaches it, use stride-1 on that axis. Continue
    until both are at or below target_pre_aap.

    Returns: list of (pool_h, pool_w) tuples — kernel/stride for each
    block's MaxPool2d.
    """
    pools = []
    while H > target_pre_aap or W > target_pre_aap:
        ph = 2 if H > target_pre_aap else 1
        pw = 2 if W > target_pre_aap else 1
        if (ph, pw) == (1, 1):
            break
        pools.append((ph, pw))
        H = math.ceil(H / ph)
        W = math.ceil(W / pw)
    return pools


class PairedConvBlock(nn.Module):
    """Two conv-GN-GELU pairs followed by a configurable MaxPool.

    Exactly mirrors the original CobbRegressor block structure except
    that MaxPool2d takes a tuple kernel/stride instead of fixed 2.
    """
    def __init__(self, c_in: int, c_out: int, pool: tuple):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
        )

    def forward(self, x):
        return self.body(x)


class CobbRegressorMatched(nn.Module):
    """Spatial-preserving Cobb regressor with conv depth scaled to input.

    For 96x96 input: pool_schedule = [(2,2)]*4 → feature map 6x6 going
    into the AAP. Identical to the original CobbRegressor.

    For non-square inputs: extra asymmetric pools reduce the longer axis
    until both dims reach ~6 before the AAP. Channel widths reuse 128 for
    any blocks beyond the original 4.
    """
    def __init__(self, in_channels: int, in_h: int, in_w: int,
                 target_pre_aap: int = 6):
        super().__init__()
        pool_schedule = compute_pool_schedule(in_h, in_w, target_pre_aap)
        n_blocks = len(pool_schedule)

        base_channels = [32, 64, 128, 128]
        if n_blocks > len(base_channels):
            channels = base_channels + [128] * (n_blocks - len(base_channels))
        else:
            channels = base_channels[:n_blocks]

        layers = []
        prev_ch = in_channels
        for ch, pool in zip(channels, pool_schedule):
            layers.append(PairedConvBlock(prev_ch, ch, pool))
            prev_ch = ch
        layers.append(nn.AdaptiveAvgPool2d((6, 6)))
        self.conv = nn.Sequential(*layers)

        # FC head — preserves spatial structure (matches original exactly)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_ch * 6 * 6, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

        # For reporting
        self.pool_schedule = pool_schedule
        self.channels = channels
        self.final_channels = prev_ch

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(z))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for z, y in loader:
            preds.append(model(z.to(device)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--baseline-results", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz = np.load(args.latents, allow_pickle=True)
    latents, angles, splits, uuids = (
        npz["latents"], npz["angles"], npz["splits"], npz["uuids"])
    print(f"Loaded {len(latents)} latents of shape {latents.shape[1:]} from {args.latents}")

    def take(name):
        m = splits == name
        return latents[m], angles[m], uuids[m]
    z_train, y_train, _ = take("train")
    z_val, y_val, _ = take("val")
    z_test, y_test, uuid_test = take("test")
    print(f"  train: {len(z_train)} | val: {len(z_val)} | test: {len(z_test)}")

    # Predict-the-train-mean diagnostic
    train_mean = y_train.mean(0, keepdims=True)
    mean_baseline_test = float(np.abs(y_test - train_mean).mean())
    print(f"\n[diagnostic] Predict-the-train-mean baseline test MAE: "
          f"{mean_baseline_test:.2f}°")
    print("            (regressor must beat this to demonstrate it's learning)")

    # Standardise targets using train stats
    y_mean = y_train.mean(0, keepdims=True)
    y_std = y_train.std(0, keepdims=True) + 1e-6
    def norm(y): return (y - y_mean) / y_std

    def loader(z, y, shuffle):
        ds = TensorDataset(torch.from_numpy(z).float(),
                           torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    train_loader = loader(z_train, norm(y_train), True)
    val_loader = loader(z_val, norm(y_val), False)
    test_loader = loader(z_test, norm(y_test), False)

    in_c, in_h, in_w = latents.shape[1:]
    model = CobbRegressorMatched(in_channels=in_c, in_h=in_h, in_w=in_w).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nRegressor architecture (receptive-field matched, original-style):")
    print(f"  Input shape:         ({in_c}, {in_h}, {in_w})")
    print(f"  Pool schedule:       {model.pool_schedule}")
    print(f"  Channel schedule:    {model.channels}")
    print(f"  Final conv channels: {model.final_channels} (pooled to 6x6 by AAP)")
    print(f"  Total parameters:    {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_mae = float("inf")
    best_state = None
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for z, y in train_loader:
            z, y = z.to(device), y.to(device)
            if torch.rand(()) < 0.5:
                z = torch.flip(z, dims=[-1])
            loss = F.smooth_l1_loss(model(z), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        scheduler.step()

        preds_n, targets_n = evaluate(model, val_loader, device)
        preds = preds_n * y_std + y_mean
        targets = targets_n * y_std + y_mean
        val_mae = float(np.abs(preds - targets).mean())

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d} | train_loss={np.mean(train_losses):.3f} "
                  f"| val_MAE={val_mae:.2f}° | best_val_MAE={best_val_mae:.2f}°")

    print(f"\nBest val MAE: {best_val_mae:.2f}°")
    model.load_state_dict(best_state)

    preds_n, targets_n = evaluate(model, test_loader, device)
    test_preds = preds_n * y_std + y_mean
    test_targets = targets_n * y_std + y_mean
    per_image_mae = np.abs(test_preds - test_targets).mean(axis=1)
    per_angle_mae = np.abs(test_preds - test_targets).mean(axis=0)

    print("\n=== Test set results ===")
    print(f"  Mean-baseline MAE:     {mean_baseline_test:.2f}° (must beat this)")
    print(f"  Overall MAE:           {per_image_mae.mean():.2f}° "
          f"(± {per_image_mae.std(ddof=1):.2f})")
    print(f"  Proximal-thoracic MAE: {per_angle_mae[0]:.2f}°")
    print(f"  Main-thoracic MAE:     {per_angle_mae[1]:.2f}°")
    print(f"  Thoracolumbar MAE:     {per_angle_mae[2]:.2f}°")
    if per_image_mae.mean() >= mean_baseline_test - 0.3:
        print("  ⚠  Regressor barely beats predict-the-mean. Latents not "
              "providing useful signal at this head capacity.")

    np.savez(args.out, test_preds=test_preds, test_targets=test_targets,
             test_uuids=uuid_test, per_image_mae=per_image_mae,
             per_angle_mae=per_angle_mae,
             best_val_mae=np.array(best_val_mae),
             mean_baseline_mae=np.array(mean_baseline_test))
    print(f"\nWrote {args.out}")

    if args.baseline_results:
        b = np.load(args.baseline_results, allow_pickle=True)
        b_uuids = b["test_uuids"]
        order = [list(b_uuids).index(u) for u in uuid_test]
        b_per_image = b["per_image_mae"][order]
        delta = b_per_image - per_image_mae
        wins = (delta > 0).sum()
        print(f"\n=== Paired comparison vs {args.baseline_results.name} ===")
        print(f"  Baseline overall MAE: {b_per_image.mean():.2f}°")
        print(f"  This run overall MAE: {per_image_mae.mean():.2f}°")
        print(f"  Δ (baseline − this):  {delta.mean():+.2f}° "
              f"(this run wins on {wins}/{len(delta)} images)")


if __name__ == "__main__":
    main()
