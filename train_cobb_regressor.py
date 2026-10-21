"""
train_cobb_regressor.py: train a CNN regression head on frozen MedVAE latents to predict the three Cobb angles.

The script is run on each of latents_pretrained.npz and latents_finetuned.npz
with identical hyperparameters, so that only the encoder varies.

Key design notes:
  * The head does not collapse spatial information early. Cobb angle
    prediction depends on where the curvature lies along the spine, so global
    average pooling too early discards the relevant signal.
  * Reports a "predict-the-train-mean" baseline as a sanity check. If the
    regressor does not beat this, the head is failing to extract signal from
    the latents.
  * Mild augmentation (small horizontal flip, since spinal curves are
    sometimes mirrored).

Inputs:  --latents .npz with latents, angles, splits, uuids (e.g. from
         extract_latents.py); optionally --baseline-results from an earlier
         run of this script for a paired comparison.
Outputs: --out .npz with test predictions, targets, uuids and per-image /
         per-angle MAE; results printed to stdout.

Usage:
  python train_cobb_regressor.py --latents ./latents_pretrained.npz \\
      --out ./cobb_pretrained.npz --seed 42

  python train_cobb_regressor.py --latents ./latents_finetuned.npz \\
      --out ./cobb_finetuned.npz \\
      --baseline-results ./cobb_pretrained.npz --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CobbRegressor(nn.Module):
    """Spatial-preserving head: conv stack downsamples 96x96 -> 6x6, then MLP.

    About 1.76M params: 0.58M in the conv stack and 1.18M in the MLP head
    (4608 -> 256 -> 3). The conv stack keeps a 6x6 spatial grid, because
    average-pooling to 1x1 early would remove spatial structure.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # 96x96 -> 48x48 -> 24x24 -> 12x12 -> 6x6
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.GELU(),
                nn.MaxPool2d(2),
            )
        self.conv = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128),
            # Adaptive pool to (6, 6) so this architecture works for both
            # 96×96 latents (4× model), where the pool is the identity, and
            # 48×48 latents (8× model), where the conv stack ends at 3×3 and
            # the pool expands it nearest-neighbour-style to 6×6 so the
            # Linear below has its expected 4608-input shape.
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

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

    # Predict-the-train-mean diagnostic. If the regressor does not beat this on
    # the test set, the head cannot extract signal from the latents.
    train_mean = y_train.mean(0, keepdims=True)
    mean_baseline_test = float(np.abs(y_test - train_mean).mean())
    print(f"\n[diagnostic] Predict-the-train-mean baseline test MAE: "
          f"{mean_baseline_test:.2f}°")
    print("            (the regressor needs to beat this to show that it is learning)")

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

    model = CobbRegressor(in_channels=latents.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nRegressor params: {n_params:,}")

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
            # Mild augmentation: horizontal flip the latent half the time.
            # A mirror flip reverses the Cobb-angle sign, but BoostNet reports
            # magnitudes, so the labels need no change.
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
    print(f"  Mean-baseline MAE:     {mean_baseline_test:.2f}° (reference to beat)")
    print(f"  Overall MAE:           {per_image_mae.mean():.2f}° "
          f"(± {per_image_mae.std(ddof=1):.2f})")
    print(f"  Major curve (col 0) MAE: {per_angle_mae[0]:.2f}°")
    print(f"  Col 1 MAE:             {per_angle_mae[1]:.2f}°")
    print(f"  Col 2 MAE:             {per_angle_mae[2]:.2f}°")
    if per_image_mae.mean() >= mean_baseline_test - 0.3:
        print(f"  WARNING: test MAE ({per_image_mae.mean():.2f}°) is not more than 0.3° below the "
              f"predict-the-mean baseline ({mean_baseline_test:.2f}°). The latents may carry little usable signal at this head capacity.")

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
        print(f"  Δ (baseline - this):  {delta.mean():+.2f}° "
              f"(this run wins on {wins}/{len(delta)} images)")


if __name__ == "__main__":
    main()
