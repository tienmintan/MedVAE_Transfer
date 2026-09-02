"""
train_cobb_regressor_aspect.py

Cobb regressor on aspect-padded latents (produced by extract_latents_aspect.py).
Used to compare MedVAE variants at different compression ratios while holding
the inference pipeline constant (the original Phase 1 setup that produced
the 7.50° baseline).

Architecture mirrors the Phase 1 baseline regressor — same conv-block design,
same depth, just adapts to whatever spatial extent the latent has:

    Input: (B, Cz, H, W)
        Cz=1 H=W=96 for medvae_4_1_2d  (the 7.50° baseline)
        Cz=1 H=W=48 for medvae_8_1_2d  (the new experiment)
        Cz=3 H=W=96 for medvae_4_3_2d  (a possible follow-up)
        Cz=4 H=W=48 for medvae_8_4_2d  (a possible follow-up)
    Conv stack: 4 stride-2 blocks with channels {32, 64, 128, 128}.
    Adaptive avg pool to (3, 3) regardless of input shape.
    MLP head: Linear(128*9, 128) → Linear(128, 3).

This keeps the regressor architecturally constant across encoder variants,
with the adaptive pool absorbing the spatial-extent differences. Total
params: ~250k-300k depending on Cz. Smaller than the Phase 1 regressor's
600k because we're using adaptive pooling more aggressively; if we wanted
exact param-match we could add a fifth conv block, but the adaptive design
is more honest to the comparison.

Usage:
    # Match the Phase 1 baseline run
    python train_cobb_regressor_aspect.py \\
        --latents-dir ./latents_aspect_4_1 \\
        --labels-csv  ./medvae/data/boostnet_with_angles.csv \\
        --out-dir     ./cobb_regressor_aspect_4_1

    # Run the new 8x experiment
    python train_cobb_regressor_aspect.py \\
        --latents-dir ./latents_aspect_8_1 \\
        --labels-csv  ./medvae/data/boostnet_with_angles.csv \\
        --out-dir     ./cobb_regressor_aspect_8_1
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

class CobbRegressorAspect(nn.Module):
    """Architecturally fixed regressor — adapts to input shape via the pool."""
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
                nn.Dropout2d(dropout),
            )
        self.conv1 = block(in_channels, 32)
        self.conv2 = block(32,          64)
        self.conv3 = block(64,         128)
        self.conv4 = block(128,        128)
        self.pool  = nn.AdaptiveAvgPool2d((3, 3))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.conv2(x)
        x = self.conv3(x); x = self.conv4(x)
        x = self.pool(x)
        return self.head(x)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class CobbLatentDataset(Dataset):
    def __init__(self, latents_npz: Path, labels_df: pd.DataFrame,
                 angle_cols: list,
                 label_mean=None, label_std=None):
        data = np.load(latents_npz, allow_pickle=True)
        self.latents = data["latents"]
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
# Train/eval loop (same as the other Cobb scripts)
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
    p.add_argument("--patience",    type=int, default=30)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--angle-cols",  nargs=3,
                   default=["angle_proximal_thoracic", "angle_main_thoracic",
                            "angle_thoracolumbar"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    labels = pd.read_csv(args.labels_csv)
    for c in args.angle_cols:
        if c not in labels.columns:
            raise ValueError(f"Missing column {c!r} in labels CSV.")
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

    model = CobbRegressorAspect(in_channels=Cz, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: aspect regressor, {n_params/1e3:.1f}k params, Cz={Cz}")

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
        train_loss, train_mae, _, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device,
            label_mean, label_std, train=True)
        val_loss, val_mae, val_per, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, device,
            label_mean, label_std, train=False)
        scheduler.step()

        history.append({"epoch": epoch,
                        "train_loss": train_loss, "train_mae": train_mae,
                        "val_loss": val_loss, "val_mae": val_mae,
                        "val_mae_angle1": val_per[0],
                        "val_mae_angle2": val_per[1],
                        "val_mae_angle3": val_per[2]})

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
            print(f"  Early stop at epoch {epoch}. Best val MAE: {best_val_mae:.3f}°")
            break

    model.load_state_dict(torch.load(args.out_dir / "best.pt"))
    _, test_mae, test_per, test_preds, test_trues = run_epoch(
        model, test_loader, optimizer, criterion, device,
        label_mean, label_std, train=False)

    print(f"\n=== Test set ===")
    print(f"  MAE all angles:       {test_mae:.3f}°")
    print(f"  Per-angle MAE:        {test_per.round(3)}")
    print(f"  Mean-baseline MAE:    {mean_baseline_mae:.2f}°")
    print(f"  Phase 1 baseline (pretrained 4x/1ch, 384x384):   7.50°")

    pred_df = pd.DataFrame({
        "image_uuid":  test_ds.uuids,
        "pred_angle1": test_preds[:, 0], "true_angle1": test_trues[:, 0],
        "pred_angle2": test_preds[:, 1], "true_angle2": test_trues[:, 1],
        "pred_angle3": test_preds[:, 2], "true_angle3": test_trues[:, 2],
    })
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False)

    per_image_mae = np.abs(test_preds - test_trues).mean(axis=1).astype(np.float32)
    per_angle_mae = np.abs(test_preds - test_trues).mean(axis=0).astype(np.float32)
    np.savez(
        args.out_dir / "cobb_aspect.npz",
        test_preds=test_preds.astype(np.float32),
        test_targets=test_trues.astype(np.float32),
        test_uuids=np.array(test_ds.uuids),
        per_image_mae=per_image_mae,
        per_angle_mae=per_angle_mae,
        best_val_mae=np.array(float(best_val_mae)),
        mean_baseline_mae=np.array(float(mean_baseline_mae)),
    )
    print(f"  Wrote {args.out_dir / 'cobb_aspect.npz'}")

    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({"label_mean":    label_mean.tolist(),
                   "label_std":     label_std.tolist(),
                   "latent_shape":  [int(Cz), int(H), int(W)],
                   "n_params":      n_params,
                   "best_val_mae":  float(best_val_mae),
                   "test_mae":      float(test_mae)}, f, indent=2)


if __name__ == "__main__":
    main()
