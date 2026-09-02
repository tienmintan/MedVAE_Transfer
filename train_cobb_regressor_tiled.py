"""
train_cobb_regressor_tiled.py

Phase 3 step 2 — train a Cobb angle regressor on the patch-encoded latent
grids produced by extract_latents_tiled.py.

Architecture: 1D-along-spine.
    Input: (B, Cz, 576, 192) latent grid (Cz=1 for medvae_4_1_2d)
    1. Mean-pool across the x-axis (left-right of spine) → (B, Cz, 576)
       Rationale: Cobb angle is a vertical curvature property. Pooling
       across x first throws away left-right detail (irrelevant for Cobb)
       while preserving the along-spine signal that matters.
    2. 1D conv stack down the spine: 576 → 288 → 144 → 72 → 36
       Channels 64 → 128 → 128 → 256. Stride-2 convs, BatchNorm, GELU.
    3. Flatten and MLP head → 3 angles.

Total params: ~400k. Smaller than the previous 2D regressor (~600k) but
better matched to the problem geometry.

Training:
    - Smooth-L1 loss across all 3 angles
    - AdamW, lr 1e-3, weight decay 1e-3, cosine schedule
    - Batch 32, 200 epochs
    - Dropout 0.3
    - Early-stop on val MAE plateau (patience 30 epochs)
    - Standardize labels by training-set mean/std for stability

Usage:
    python train_cobb_regressor_tiled.py \\
        --latents-dir ./latents_tiled \\
        --labels-csv  ./medvae/data/boostnet_with_angles.csv \\
        --out-dir     ./cobb_regressor_tiled
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

class CobbRegressor1D(nn.Module):
    """Latent grid → 3 Cobb angles via 1D convs along the spine axis.

    Input shape: (B, Cz, H, W) where H is along-spine, W is across-spine.
    We mean-pool across W first, then run 1D convs down H.
    """
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        # After mean-pool over x, shape is (B, in_channels, 576)
        # Run 1D conv stack. Strides reduce spatial extent by half each block.
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(cout),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.conv1 = block(in_channels, 64)    # 576 → 288
        self.conv2 = block(64,         128)    # 288 → 144
        self.conv3 = block(128,        128)    # 144 → 72
        self.conv4 = block(128,        256)    # 72 → 36
        # Final spatial: 256 channels x 36 = 9216
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 36, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cz, H, W) → mean-pool across W → (B, Cz, H)
        x = x.mean(dim=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.head(x)


class CobbRegressor2D(nn.Module):
    """Latent grid → 3 Cobb angles via 2D stride-2 conv blocks, matched in
    architecture to the previous 384x384-latent baseline regressor.

    The previous baseline used 5 stride-2 blocks on a 96x96 latent
    (96→48→24→12→6 spatial) with ~600k params. Here the input is 576x192,
    so we add one more stride-2 stage to get to comparable final spatial
    extent (9x3), keeping per-stage channel counts modest so total
    parameters stay near the baseline.

    Architecture:
        576x192 (Cz)   stride-2 block (32 ch)  → 288x96
                       stride-2 block (64 ch)  → 144x48
                       stride-2 block (96 ch)  →  72x24
                       stride-2 block (128 ch) →  36x12
                       stride-2 block (128 ch) →  18x6
                       stride-2 block (192 ch) →   9x3
        flatten (9*3*192 = 5184) → Linear → 3
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
        self.conv1 = block(in_channels, 32)    # 576×192 → 288×96
        self.conv2 = block(32,          64)    # 288×96  → 144×48
        self.conv3 = block(64,          96)    # 144×48  → 72×24
        self.conv4 = block(96,         128)    # 72×24   → 36×12
        self.conv5 = block(128,        128)    # 36×12   → 18×6
        self.conv6 = block(128,        192)    # 18×6    → 9×3
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 9 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.head(x)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class CobbLatentDataset(Dataset):
    """Loads pre-extracted latents and Cobb angle labels.

    Labels are standardized (subtract mean, divide std) for training stability;
    predictions are un-standardized at evaluation time.
    """
    def __init__(self, latents_npz: Path, labels_df: pd.DataFrame,
                 angle_cols: list,
                 label_mean=None, label_std=None):
        data = np.load(latents_npz, allow_pickle=True)
        self.latents = data["latents"]                      # (N, Cz, H, W)
        self.uuids   = data["uuids"].astype(str)
        self.angle_cols = angle_cols
        # Align labels to UUIDs
        labels = labels_df.set_index("image_uuid").loc[self.uuids]
        self.labels = labels[self.angle_cols].values.astype(np.float32)  # (N, 3)
        # Standardization (computed externally for the train split, reused for val/test)
        self.label_mean = label_mean
        self.label_std  = label_std

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int):
        z = torch.from_numpy(self.latents[idx])    # (Cz, H, W)
        y = self.labels[idx]
        if self.label_mean is not None:
            y_std = (y - self.label_mean) / self.label_std
        else:
            y_std = y
        return z, torch.from_numpy(y_std).float(), torch.from_numpy(y).float()


# ----------------------------------------------------------------------------
# Train / eval loop
# ----------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, label_mean, label_std,
              train: bool):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_pred_raw = []
    all_true_raw = []
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
            # Un-standardize for raw-degree MAE reporting
            pred_raw = pred_std.detach().cpu().numpy() * label_std + label_mean
            all_pred_raw.append(pred_raw)
            all_true_raw.append(y_raw.numpy())

    preds = np.concatenate(all_pred_raw, axis=0)
    trues = np.concatenate(all_true_raw, axis=0)
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
    p.add_argument("--arch", choices=["1d", "2d"], default="2d",
                   help="Regressor architecture. '2d' is the apples-to-apples "
                        "comparison against the previous 384x384 baseline; "
                        "'1d' is the along-spine variant.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    labels = pd.read_csv(args.labels_csv)
    # Verify the requested angle columns exist
    missing_cols = [c for c in args.angle_cols if c not in labels.columns]
    if missing_cols:
        raise ValueError(
            f"Labels CSV missing angle columns: {missing_cols}. "
            f"Available columns: {labels.columns.tolist()}")
    # Compute label mean/std from training labels only (no test leakage)
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

    # Inspect dimensions
    z0, _, _ = train_ds[0]
    Cz, H, W = z0.shape
    print(f"Latent shape: Cz={Cz} H={H} W={W}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    if args.arch == "1d":
        model = CobbRegressor1D(in_channels=Cz, dropout=args.dropout).to(device)
    else:
        model = CobbRegressor2D(in_channels=Cz, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.arch.upper()} regressor, {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    # Mean-baseline test MAE (predict train-mean for all test items): same
    # number we computed in the previous round. Print for reference.
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

    # Evaluate best on test set
    model.load_state_dict(torch.load(args.out_dir / "best.pt"))
    _, test_mae, test_per, test_preds, test_trues = run_epoch(
        model, test_loader, optimizer, criterion, device,
        label_mean, label_std, train=False)

    print(f"\n=== Test set ===")
    print(f"  MAE all angles:       {test_mae:.3f}°")
    print(f"  Per-angle MAE:        {test_per.round(3)}")
    print(f"  Mean-baseline MAE:    {mean_baseline_mae:.2f}°")
    print(f"  Previous best (pretrained encoder, aspect):  7.50°")
    print(f"  Previous (aspect-finetuned encoder, aspect): 7.58°")

    # Save predictions for downstream Bland-Altman / paired comparison.
    # Two formats:
    #  - CSV (human-readable, inspect with pandas)
    #  - NPZ matching the baseline cobb_pretrained.npz schema so the
    #    comparison script can treat both files identically.
    pred_df = pd.DataFrame({
        "image_uuid":  test_ds.uuids,
        "pred_angle1": test_preds[:, 0], "true_angle1": test_trues[:, 0],
        "pred_angle2": test_preds[:, 1], "true_angle2": test_trues[:, 1],
        "pred_angle3": test_preds[:, 2], "true_angle3": test_trues[:, 2],
    })
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    print(f"  Wrote {args.out_dir / 'test_predictions.csv'}")

    per_image_mae = np.abs(test_preds - test_trues).mean(axis=1).astype(np.float32)
    per_angle_mae = np.abs(test_preds - test_trues).mean(axis=0).astype(np.float32)
    np.savez(
        args.out_dir / "cobb_tiled.npz",
        test_preds=test_preds.astype(np.float32),
        test_targets=test_trues.astype(np.float32),
        test_uuids=np.array(test_ds.uuids),
        per_image_mae=per_image_mae,
        per_angle_mae=per_angle_mae,
        best_val_mae=np.array(float(best_val_mae)),
        mean_baseline_mae=np.array(float(mean_baseline_mae)),
    )
    print(f"  Wrote {args.out_dir / 'cobb_tiled.npz'}")

    # Save training history
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({"label_mean": label_mean.tolist(),
                   "label_std":  label_std.tolist(),
                   "n_params":   n_params,
                   "best_val_mae": float(best_val_mae),
                   "test_mae":   float(test_mae)}, f, indent=2)
    print(f"  Saved history to {args.out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
