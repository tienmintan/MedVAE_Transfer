"""
train_cobb_regressor_whole.py

Phase 3-redux step 2 — train a Cobb angle regressor on whole-image latents
produced by extract_latents_whole.py.

Architecture: 2D CNN, closely matched to the previous 96x96 baseline regressor.
    Input: (B, Cz, 96, 192) latent (Cz=1 for medvae_4_1_2d)
    Five stride-2 conv blocks bring the spatial down: 96×192 → 48×96 →
      24×48 → 12×24 → 6×12 → 3×6, channels 32→64→128→128→192.
    Adaptive average pool to (3, 3), then small MLP → 3 angles.

Total params: ~680k. Architecturally as close as possible to the previous
600k-param 96x96 baseline — same depth, same conv-block design — just with
slightly more channels in the last layer to handle the rectangular input,
and an adaptive pool to bring everything to a fixed shape before the head.

Why this architecture vs. the Phase 3 (tiled) regressor:
  - Smaller input (18k positions vs 110k) means a smaller regressor is
    appropriate. Phase 3's 1.2M-param regressor overfit on 409 training
    images; this one should overfit less because there's less to overfit on.
  - The architecture is now genuinely comparable to the previous 96x96
    baseline, making the encoder comparison cleaner: same regressor shape,
    different encoded representation.

Training: smooth-L1 on standardised labels, AdamW with cosine LR schedule,
early stopping on val MAE. Same hyperparameters as Phase 3 by default,
override via CLI if needed.

Usage:
    python train_cobb_regressor_whole.py \\
        --latents-dir ./latents_whole \\
        --labels-csv  ./medvae/data/boostnet_with_angles.csv \\
        --out-dir     ./cobb_regressor_whole
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class CobbRegressorWhole(nn.Module):
    """Whole-image latent (96x192) → 3 Cobb angles.

    Architectural twin of the previous 96x96 baseline regressor — same depth,
    same conv-block design, an adaptive pool before the MLP head so the
    head's input dimension is fixed regardless of small input variations.
    """
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
                nn.Dropout2d(dropout),
            )
        self.conv1 = block(in_channels, 32)    # 96×192 → 48×96
        self.conv2 = block(32,          64)    # 48×96  → 24×48
        self.conv3 = block(64,         128)    # 24×48  → 12×24
        self.conv4 = block(128,        128)    # 12×24  → 6×12
        self.conv5 = block(128,        192)    # 6×12   → 3×6
        self.pool  = nn.AdaptiveAvgPool2d((3, 3))   # → 3×3
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x); x = self.pool(x)
        return self.head(x)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class CobbLatentDataset(Dataset):
    """Loads pre-extracted whole-image latents and Cobb angle labels.

    Identical interface to the tiled-pipeline version, just operating on
    smaller (1, 96, 192) latent tensors.
    """
    def __init__(self, latents_npz: Path, labels_df: pd.DataFrame,
                 angle_cols: list,
                 label_mean=None, label_std=None):
        data = np.load(latents_npz, allow_pickle=True)
        self.latents = data["latents"]              # (N, Cz, 192, 96)
        self.uuids   = data["uuids"].astype(str)
        self.angle_cols = angle_cols
        labels = labels_df.set_index("image_uuid").loc[self.uuids]
        self.labels = labels[self.angle_cols].values.astype(np.float32)
        self.label_mean = label_mean
        self.label_std  = label_std

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int):
        z = torch.from_numpy(self.latents[idx])
        y = self.labels[idx]
        if self.label_mean is not None:
            y_std = (y - self.label_mean) / self.label_std
        else:
            y_std = y
        return z, torch.from_numpy(y_std).float(), torch.from_numpy(y).float()


# ----------------------------------------------------------------------------
# Train / eval loop
# ----------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device,
              label_mean, label_std, train: bool):
    model.train() if train else model.eval()
    losses, preds_raw, trues_raw = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for z, y_std, y_raw in loader:
            z, y_std = z.to(device), y_std.to(device)
            pred_std = model(z)
            loss = criterion(pred_std, y_std)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pred_raw = pred_std.detach().cpu().numpy() * label_std + label_mean
            preds_raw.append(pred_raw)
            trues_raw.append(y_raw.numpy())

    preds = np.concatenate(preds_raw, axis=0)
    trues = np.concatenate(trues_raw, axis=0)
    mae_per_angle = np.abs(preds - trues).mean(axis=0)
    mae_all = np.abs(preds - trues).mean()
    return np.mean(losses), mae_all, mae_per_angle, preds, trues


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True, type=Path)
    p.add_argument("--labels-csv",  required=True, type=Path)
    p.add_argument("--out-dir",     required=True, type=Path)
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--patience",    type=int, default=30,
                   help="Epochs to wait for val MAE improvement before stopping.")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--angle-cols",  nargs=3,
                   default=["angle_proximal_thoracic", "angle_main_thoracic",
                            "angle_thoracolumbar"],
                   help="Names of the 3 Cobb angle columns in the labels CSV.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    labels = pd.read_csv(args.labels_csv)
    missing_cols = [c for c in args.angle_cols if c not in labels.columns]
    if missing_cols:
        raise ValueError(
            f"Labels CSV missing angle columns: {missing_cols}. "
            f"Available columns: {labels.columns.tolist()}")
    train_labels = labels[labels["split"] == "train"][args.angle_cols].values
    label_mean = train_labels.mean(axis=0).astype(np.float32)
    label_std  = train_labels.std(axis=0).astype(np.float32)
    print(f"Train label mean: {label_mean.round(2)}  std: {label_std.round(2)}")

    train_ds = CobbLatentDataset(args.latents_dir / "latents_train.npz", labels,
                                 args.angle_cols, label_mean, label_std)
    val_ds   = CobbLatentDataset(args.latents_dir / "latents_val.npz",   labels,
                                 args.angle_cols, label_mean, label_std)
    test_ds  = CobbLatentDataset(args.latents_dir / "latents_test.npz",  labels,
                                 args.angle_cols, label_mean, label_std)
    print(f"Sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    z0, _, _ = train_ds[0]
    Cz, H, W = z0.shape
    print(f"Latent shape: Cz={Cz} H={H} W={W}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = CobbRegressorWhole(in_channels=Cz, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: whole-image regressor, {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    test_raw = test_ds.labels
    mean_baseline_mae = np.abs(test_raw - label_mean).mean()
    print(f"Mean-baseline test MAE: {mean_baseline_mae:.2f}°")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    epochs_since_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae, train_per, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device,
            label_mean, label_std, train=True)
        val_loss, val_mae, val_per, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, device,
            label_mean, label_std, train=False)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
            "val_loss": val_loss, "val_mae": val_mae,
            "val_mae_angle1": val_per[0], "val_mae_angle2": val_per[1],
            "val_mae_angle3": val_per[2],
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_since_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            marker = " *"
        else:
            epochs_since_improve += 1
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or marker:
            print(f"  [epoch {epoch:3d}] train MAE {train_mae:.3f}°  "
                  f"val MAE {val_mae:.3f}° (per-angle {val_per.round(2)}){marker}")

        if epochs_since_improve >= args.patience:
            print(f"  Early stop at epoch {epoch} (no val improvement in "
                  f"{args.patience} epochs). Best val MAE: {best_val_mae:.3f}°")
            break

    model.load_state_dict(torch.load(args.out_dir / "best.pt"))
    _, test_mae, test_per, test_preds, test_trues = run_epoch(
        model, test_loader, optimizer, criterion, device,
        label_mean, label_std, train=False)

    print(f"\n=== Test set ===")
    print(f"  MAE all angles:       {test_mae:.3f}°")
    print(f"  Per-angle MAE:        {test_per.round(3)}")
    print(f"  Mean-baseline MAE:    {mean_baseline_mae:.2f}°")
    print(f"  Previous best (pretrained encoder, aspect):       7.50°")
    print(f"  Previous (aspect-finetuned encoder, aspect):      7.58°")
    print(f"  Previous (patch encoder, tile-and-stitch, 2d):    8.16°")

    pred_df = pd.DataFrame({
        "image_uuid":  test_ds.uuids,
        "pred_angle1": test_preds[:, 0], "true_angle1": test_trues[:, 0],
        "pred_angle2": test_preds[:, 1], "true_angle2": test_trues[:, 1],
        "pred_angle3": test_preds[:, 2], "true_angle3": test_trues[:, 2],
    })
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    print(f"  Wrote {args.out_dir / 'test_predictions.csv'}")

    # Save .npz in the same schema as the previous Cobb runs for easy comparison
    per_image_mae = np.abs(test_preds - test_trues).mean(axis=1).astype(np.float32)
    per_angle_mae = np.abs(test_preds - test_trues).mean(axis=0).astype(np.float32)
    np.savez(
        args.out_dir / "cobb_whole.npz",
        test_preds=test_preds.astype(np.float32),
        test_targets=test_trues.astype(np.float32),
        test_uuids=np.array(test_ds.uuids),
        per_image_mae=per_image_mae,
        per_angle_mae=per_angle_mae,
        best_val_mae=np.array(float(best_val_mae)),
        mean_baseline_mae=np.array(float(mean_baseline_mae)),
    )
    print(f"  Wrote {args.out_dir / 'cobb_whole.npz'}")

    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({"label_mean":    label_mean.tolist(),
                   "label_std":     label_std.tolist(),
                   "n_params":      n_params,
                   "best_val_mae":  float(best_val_mae),
                   "test_mae":      float(test_mae)}, f, indent=2)
    print(f"  Saved history to {args.out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
