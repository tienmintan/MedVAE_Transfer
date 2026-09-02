"""
train_diffusion.py

Train a conditional latent diffusion model in SD-VAE or MedVAE latent space.

Two data modes supported:

  v1 (original): --latents-dir + --labels-csv
      Expects a directory with latents_train.npz / latents_val.npz files
      (separate per-split, no angles inside), joined with a CSV of Cobb
      angles by image_uuid. This is what extract_latents_whole.py and
      friends produce.

  v2 (unified): --latents-files file1.npz file2.npz ...
      Each file has latents / image_ids / splits / angles (optional) /
      aspect_pad in a single .npz. Multiple files are concatenated.
      boostnet_test entries are dropped from training automatically.
      This is what extract_latents_vindr.py and extract_latents_boostnet_512.py
      produce.

Two training modes supported:

  Conditional (default): standard FiLM Cobb conditioning with CFG dropout
      at rate --cond-drop-prob. Requires angles in the data.

  Unconditional (--unconditional): bypass Cobb conditioning. Every sample
      uses the null embedding. Used for the unconditional pretraining
      phase on the pooled VinDr+BoostNet corpus.

Use --resume-from to load model + EMA weights from a previous checkpoint.
Optimizer and LR schedule are reset (fresh warmup for the new phase).

Two-stage pipeline (the recommended usage for the VinDr-extended pipeline):

    # Stage 1: unconditional pretrain on pooled VinDr + BoostNet
    python train_diffusion.py \\
        --latents-files ./latents_vindr/latents_vindr.npz \\
                        ./latents_boostnet_512/latents_boostnet_512.npz \\
        --unconditional \\
        --out-dir ./diffusion_ckpt_pretrain \\
        --epochs 100 --batch-size 32 --mixed-precision

    # Stage 2: conditional fine-tune on BoostNet
    python train_diffusion.py \\
        --latents-files ./latents_boostnet_512/latents_boostnet_512.npz \\
        --resume-from ./diffusion_ckpt_pretrain/final.pt \\
        --out-dir ./diffusion_ckpt_finetune \\
        --epochs 100 --batch-size 32 --mixed-precision
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusion_model import UNet, GaussianDiffusion, EMA


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class CobbLatentDataset(Dataset):
    """Latents + standardised Cobb angles, joined by uuid.

    Latents are kept in CPU memory (~30 MB total for 481 samples at
    256x96x1 float32). Cobb angles are standardised using train statistics.
    """
    def __init__(self, latents: np.ndarray, cobb: np.ndarray,
                 cobb_mean: np.ndarray, cobb_std: np.ndarray,
                 hflip_p: float = 0.5):
        self.latents = latents
        self.cobb_raw = cobb
        self.cobb_std = (cobb - cobb_mean) / cobb_std
        self.hflip_p = hflip_p

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        z = torch.from_numpy(self.latents[idx]).float()
        c = torch.from_numpy(self.cobb_std[idx]).float()

        # Horizontal flip augmentation in latent space — width axis is the
        # lateral (left-right) anatomical axis, so flipping is a valid
        # augmentation. Cobb magnitudes are sign-invariant (we always
        # measure the absolute angle), so labels are unchanged.
        if torch.rand(1).item() < self.hflip_p:
            z = torch.flip(z, dims=[-1])

        return z, c


def load_data(latents_dir: Path, labels_csv: Path):
    """Load train+val latents, join with Cobb angles from CSV."""
    train_npz = np.load(latents_dir / "latents_train.npz", allow_pickle=True)
    val_npz   = np.load(latents_dir / "latents_val.npz",   allow_pickle=True)

    train_lat = train_npz["latents"]
    train_uuids = train_npz["uuids"].astype(str)
    val_lat = val_npz["latents"]
    val_uuids = val_npz["uuids"].astype(str)

    labels = pd.read_csv(labels_csv).set_index("image_uuid")
    angle_cols = ["angle_proximal_thoracic", "angle_main_thoracic",
                  "angle_thoracolumbar"]
    for c in angle_cols:
        if c not in labels.columns:
            raise ValueError(f"Missing column {c!r} in {labels_csv}")

    train_angles = labels.loc[train_uuids, angle_cols].values.astype(np.float32)
    val_angles   = labels.loc[val_uuids,   angle_cols].values.astype(np.float32)

    pooled_latents = np.concatenate([train_lat, val_lat], axis=0)
    pooled_angles  = np.concatenate([train_angles, val_angles], axis=0)

    print(f"Train + val pooled: {len(pooled_latents)} samples")
    print(f"  Latent shape per sample: {pooled_latents.shape[1:]}")
    print(f"  Angle range: PT [{pooled_angles[:,0].min():.1f}, "
          f"{pooled_angles[:,0].max():.1f}], MT "
          f"[{pooled_angles[:,1].min():.1f}, {pooled_angles[:,1].max():.1f}], "
          f"TL [{pooled_angles[:,2].min():.1f}, {pooled_angles[:,2].max():.1f}]")

    return pooled_latents, pooled_angles


def load_data_from_files(files: list, require_angles: bool):
    """Load latents from one or more .npz files in the unified v2 schema.

    Expected per-file fields:
      latents    : (N, C, H, W)
      image_ids  : (N,) strings
      splits     : (N,) strings (e.g. "boostnet_train", "vindr_train", ...)
      angles     : (N, 3) — optional; required if require_angles=True

    All "boostnet_test" entries are filtered out (held back for downstream
    evaluation). Everything else is used for training.

    For files without angles (e.g. the unlabeled VinDr pool used during
    unconditional pretraining), angles are filled with zeros and a
    ValueError is raised if require_angles=True.

    Returns (latents, angles) where latents is fp32, angles is fp32.
    """
    all_lat, all_ang, all_splits = [], [], []
    for f in files:
        data = np.load(f, allow_pickle=True)
        lat = data["latents"]
        splits = data["splits"].astype(str)
        if "angles" in data.files:
            ang = data["angles"].astype(np.float32)
        else:
            if require_angles:
                raise ValueError(
                    f"{f} has no 'angles' field but conditional training "
                    f"requires angles. Use --unconditional or supply a "
                    f"file with angles.")
            ang = np.zeros((len(lat), 3), dtype=np.float32)

        # Filter: drop boostnet_test entirely. Keep everything else.
        keep = (splits != "boostnet_test")
        n_dropped = len(lat) - keep.sum()
        all_lat.append(lat[keep])
        all_ang.append(ang[keep])
        all_splits.append(splits[keep])

        print(f"  Loaded {f}: {len(lat)} entries, {keep.sum()} kept "
              f"({n_dropped} dropped as boostnet_test)")

    pooled_latents = np.concatenate(all_lat, axis=0).astype(np.float32)
    pooled_angles  = np.concatenate(all_ang, axis=0)
    pooled_splits  = np.concatenate(all_splits, axis=0)

    print(f"\nPooled training set: {len(pooled_latents)} samples")
    print(f"  Latent shape per sample: {pooled_latents.shape[1:]}")
    print(f"  Split breakdown: " +
          ", ".join(f"{s}={(pooled_splits == s).sum()}"
                    for s in sorted(set(pooled_splits.tolist()))))
    if require_angles:
        print(f"  Angle range: "
              f"PT [{pooled_angles[:,0].min():.1f}, "
              f"{pooled_angles[:,0].max():.1f}], "
              f"MT [{pooled_angles[:,1].min():.1f}, "
              f"{pooled_angles[:,1].max():.1f}], "
              f"TL [{pooled_angles[:,2].min():.1f}, "
              f"{pooled_angles[:,2].max():.1f}]")

    return pooled_latents, pooled_angles


# ----------------------------------------------------------------------------
# Periodic sample generation (visual sanity check during training)
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, diffusion, ema, sampling_angles: torch.Tensor,
                     latent_shape, cfg_scale: float, device, n_steps: int = 50):
    """Generate samples at fixed target angles for visual inspection."""
    model.eval()
    backup = ema.store(model)
    ema.copy_to(model)
    samples = diffusion.sample_ddim(model, latent_shape, sampling_angles,
                                    cfg_scale=cfg_scale, n_steps=n_steps,
                                    device=device)
    ema.restore(model, backup)
    model.train()
    return samples.detach().cpu()


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Data sources — two mutually exclusive modes
    p.add_argument("--latents-dir", type=Path, default=None,
                   help="Old schema: directory with latents_train.npz + "
                        "latents_val.npz files (joined with --labels-csv). "
                        "Use --latents-files instead for the v2 schema.")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="Required with --latents-dir.")
    p.add_argument("--latents-files", type=Path, nargs="+", default=None,
                   help="New unified schema: one or more .npz files each "
                        "containing latents/image_ids/splits/angles. "
                        "Multiple files are pooled. boostnet_test entries "
                        "are dropped automatically.")
    # Training mode
    p.add_argument("--unconditional", action="store_true",
                   help="Ignore Cobb angles entirely — use null embedding "
                        "for every sample. Required for the unconditional "
                        "pretraining phase on data without Cobb labels.")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Path to a checkpoint to resume from. Model and EMA "
                        "weights are loaded; optimizer and LR schedule are "
                        "reset. Use for conditional fine-tuning after "
                        "unconditional pretrain.")
    # Output
    p.add_argument("--out-dir",     required=True, type=Path)
    # Training hyperparams
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--ema-decay",   type=float, default=0.9999)
    p.add_argument("--cond-drop-prob", type=float, default=0.15,
                   help="Probability of dropping Cobb conditioning during "
                        "training (enables classifier-free guidance). "
                        "Ignored in --unconditional mode (always drop).")
    p.add_argument("--sample-every", type=int, default=20,
                   help="Generate visual samples every N epochs.")
    p.add_argument("--sample-cfg-scale", type=float, default=3.0,
                   help="CFG scale for periodic samples (1=no guidance).")
    p.add_argument("--T", type=int, default=1000,
                   help="Number of diffusion timesteps.")
    p.add_argument("--parameterization", choices=["eps", "v"], default="eps",
                   help="Diffusion target parameterization. 'eps' predicts "
                        "noise directly (standard DDPM). 'v' predicts the "
                        "v-target from Salimans & Ho 2022, which is more "
                        "numerically stable across timesteps — recommended "
                        "for heterogeneous data and small-data regimes.")
    p.add_argument("--channels", type=int, nargs="+",
                   default=[64, 128, 256, 256])
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", action="store_true",
                   help="Use bf16 autocast for forward pass.")
    args = p.parse_args()

    # Validate argument combinations
    if args.latents_files is None and args.latents_dir is None:
        p.error("Must provide either --latents-files (v2) or "
                "--latents-dir + --labels-csv (v1).")
    if args.latents_files is not None and args.latents_dir is not None:
        p.error("Provide either --latents-files OR --latents-dir, not both.")
    if args.latents_dir is not None and args.labels_csv is None:
        p.error("--latents-dir requires --labels-csv.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Data ----------
    if args.latents_files is not None:
        print(f"Loading from v2 schema files:")
        latents, angles = load_data_from_files(
            args.latents_files,
            require_angles=not args.unconditional,
        )
    else:
        latents, angles = load_data(args.latents_dir, args.labels_csv)

    cobb_mean = angles.mean(axis=0).astype(np.float32)
    cobb_std  = angles.std(axis=0).astype(np.float32)
    # Guard against std=0 in unconditional mode where angles are all zeros
    cobb_std = np.where(cobb_std < 1e-6, 1.0, cobb_std).astype(np.float32)
    print(f"Cobb angle mean: {cobb_mean.round(2)}")
    print(f"Cobb angle std:  {cobb_std.round(2)}")

    # Latent standardisation — scalar mean and std across the full training set.
    # Cast to fp32 before computing to avoid fp16 overflow on large pools.
    latents_fp32 = latents.astype(np.float32)
    latent_mean = float(latents_fp32.mean())
    latent_std  = float(latents_fp32.std())
    print(f"Latent stats: mean={latent_mean:.4f} std={latent_std:.4f}")

    # If resuming, prefer the latent stats from the checkpoint to keep the
    # train-time distribution consistent with what the model already knows.
    resume_ckpt = None
    if args.resume_from is not None:
        print(f"\nResuming from {args.resume_from}")
        resume_ckpt = torch.load(args.resume_from, map_location="cpu",
                                  weights_only=False)
        if "latent_mean" in resume_ckpt and "latent_std" in resume_ckpt:
            latent_mean = float(resume_ckpt["latent_mean"])
            latent_std  = float(resume_ckpt["latent_std"])
            print(f"  Using checkpoint's latent stats: "
                  f"mean={latent_mean:.4f} std={latent_std:.4f}")

    latents = (latents_fp32 - latent_mean) / latent_std
    print(f"After standardisation: mean={latents.mean():.4f} "
          f"std={latents.std():.4f}")

    dataset = CobbLatentDataset(latents, angles, cobb_mean, cobb_std,
                                hflip_p=0.5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f"Dataloader: {len(loader)} batches per epoch")

    # ---------- Model + diffusion process ----------
    model = UNet(in_channels=latents.shape[1],
                 channels=tuple(args.channels),
                 embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNet, {n_params/1e6:.2f}M params")

    diffusion = GaussianDiffusion(T=args.T, device=device,
                                   parameterization=args.parameterization)
    ema = EMA(model, decay=args.ema_decay)
    print(f"Diffusion parameterization: {args.parameterization}")

    # If resuming, load model + EMA weights from checkpoint. Optimizer and
    # LR schedule are deliberately NOT loaded — we want a fresh learning-rate
    # warmup for the fine-tuning phase.
    if resume_ckpt is not None:
        # Verify architecture compatibility
        ckpt_hp = resume_ckpt["hyperparams"]
        if (tuple(ckpt_hp["channels"]) != tuple(args.channels) or
            ckpt_hp["embed_dim"] != args.embed_dim or
            ckpt_hp["in_channels"] != latents.shape[1]):
            raise ValueError(
                f"Architecture mismatch between checkpoint and current "
                f"config. Ckpt: channels={ckpt_hp['channels']}, "
                f"embed_dim={ckpt_hp['embed_dim']}, "
                f"in_channels={ckpt_hp['in_channels']}. "
                f"Current: channels={args.channels}, "
                f"embed_dim={args.embed_dim}, in_channels={latents.shape[1]}.")
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt["ema"])
        print(f"  Loaded model + EMA weights from checkpoint "
              f"(was at epoch {resume_ckpt.get('epoch', '?')})")
        if resume_ckpt.get("was_unconditional", False):
            print(f"  Checkpoint was trained unconditionally — Cobb "
                  f"embedding starts from initialisation for this run.")
        del resume_ckpt   # Free memory

    # ---------- Optimisation ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=0.0)

    total_steps = len(loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0   # constant after warmup; cosine is overkill at this scale
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    autocast_dtype = torch.bfloat16 if args.mixed_precision else torch.float32

    # ---------- Fixed sampling angles for periodic visualisation ----------
    # In unconditional mode, the conditioning is ignored at sample time so
    # we just use zeros. In conditional mode, two anchor points (mild +
    # severe scoliosis) let us visually check whether conditioning works.
    if args.unconditional:
        sample_angles_raw = np.array([[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]], dtype=np.float32)
        # CFG must be 1.0 in unconditional mode (no guidance possible)
        effective_sample_cfg = 1.0
    else:
        sample_angles_raw = np.array([
            [10.0,  5.0,  5.0],     # mild
            [40.0, 30.0, 20.0],     # severe
        ], dtype=np.float32)
        effective_sample_cfg = args.sample_cfg_scale
    sample_angles_std = (sample_angles_raw - cobb_mean) / cobb_std
    sample_angles_tensor = torch.from_numpy(sample_angles_std).to(device)
    latent_shape = (2, *latents.shape[1:])

    # In unconditional mode, every sample is treated as "no conditioning"
    # (drop probability = 1). The Cobb embedding MLP receives gradient zero
    # (its output is masked away) and stays at initialisation, which is the
    # desired behaviour for the unconditional pretrain phase.
    effective_cond_drop_prob = 1.0 if args.unconditional else args.cond_drop_prob
    if args.unconditional:
        print(f"Mode: UNCONDITIONAL — always using null embedding "
              f"(cond_drop_prob forced to 1.0)")
    else:
        print(f"Mode: CONDITIONAL — cond_drop_prob={effective_cond_drop_prob}")

    # ---------- Training loop ----------
    print(f"\nTraining for {args.epochs} epochs...")
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for z, c in loader:
            z = z.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=args.mixed_precision):
                loss = diffusion.training_loss(model, z, c,
                                               cond_drop_prob=effective_cond_drop_prob)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_losses.append(loss.item())
            global_step += 1

        avg_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "loss": avg_loss,
                        "lr": optimizer.param_groups[0]["lr"]})

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [epoch {epoch:3d}] loss={avg_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # ---------- Periodic sample generation ----------
        if args.sample_every > 0 and (epoch % args.sample_every == 0
                                       or epoch == args.epochs):
            samples = generate_samples(
                model, diffusion, ema, sample_angles_tensor,
                latent_shape, effective_sample_cfg, device)
            # Un-standardise back to MedVAE native distribution so the saved
            # latents can be decoded directly by decode_samples.py
            samples_native = samples * latent_std + latent_mean
            sample_path = args.out_dir / f"samples_epoch_{epoch:04d}.npz"
            np.savez(sample_path,
                     latents=samples_native.numpy().astype(np.float32),
                     target_angles=sample_angles_raw,
                     epoch=epoch)
            print(f"    Saved sample latents to {sample_path}  "
                  f"(standardised stats: mean={samples.mean():.3f} "
                  f"std={samples.std():.3f})")

    # ---------- Save final checkpoint ----------
    ckpt = {
        "epoch":             args.epochs,
        "model":             model.state_dict(),
        "ema":               ema.state_dict(),
        "optimizer":         optimizer.state_dict(),
        "cobb_mean":         cobb_mean,
        "cobb_std":          cobb_std,
        "latent_mean":       latent_mean,
        "latent_std":        latent_std,
        "was_unconditional": bool(args.unconditional),
        "hyperparams":  {
            "channels": args.channels,
            "embed_dim": args.embed_dim,
            "T": args.T,
            "parameterization": args.parameterization,
            "in_channels": latents.shape[1],
            "latent_shape": list(latents.shape[1:]),
        },
        "history":      history,
    }
    ckpt_path = args.out_dir / "final.pt"
    torch.save(ckpt, ckpt_path)
    print(f"\nWrote {ckpt_path}")

    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)

    def _json_safe(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [_json_safe(x) for x in v]
        return v

    with open(args.out_dir / "config.json", "w") as f:
        json.dump({
            "n_params":    n_params,
            "cobb_mean":   cobb_mean.tolist(),
            "cobb_std":    cobb_std.tolist(),
            "latent_mean": latent_mean,
            "latent_std":  latent_std,
            "hyperparams": ckpt["hyperparams"],
            "args":        {k: _json_safe(v) for k, v in vars(args).items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()
