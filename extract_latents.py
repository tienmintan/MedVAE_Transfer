"""
extract_latents.py

Encode every image in boostnet_with_angles.csv to its MedVAE latent and save
all (latent, label) pairs to a single .npz file for fast loading during
regressor training.

Run this TWICE — once for the pretrained MedVAE, once for your finetuned
checkpoint. The downstream regressor script then trains on each in turn.

Usage:
  # Pretrained baseline:
  python extract_latents.py \\
      --csv      $PROJECT_DIR/medvae/data/boostnet_with_angles.csv \\
      --img-dir  $PROJECT_DIR/medvae/data/boostnet \\
      --out      ./latents_pretrained.npz

  # Finetuned:
  python extract_latents.py \\
      --csv      $PROJECT_DIR/medvae/data/boostnet_with_angles.csv \\
      --img-dir  $PROJECT_DIR/medvae/data/boostnet \\
      --finetuned-ckpt /path/to/best.pt/pytorch_model.bin \\
      --out      ./latents_finetuned.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune_aspect


def load_finetuned_weights(mvae: MVAE, ckpt_path: Path) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = mvae.model.load_state_dict(sd, strict=False)
    print(f"Loaded {ckpt_path}")
    print(f"  missing keys:    {len(missing)} | unexpected keys: {len(unexpected)}")


@torch.no_grad()
def encode(mvae: MVAE, img: torch.Tensor) -> torch.Tensor:
    """Return the deterministic latent (posterior mode), not a sample."""
    posterior = mvae.model.encode(img)
    return posterior.mode()  # (1, latent_C, latent_H, latent_W)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--finetuned-ckpt", type=Path, default=None,
                   help="Optional. If omitted, uses pretrained MedVAE.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-name", default="medvae_4_1_2d")
    p.add_argument("--modality", default="xray")
    args = p.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["angle_proximal_thoracic"])
    df = df.reset_index(drop=True)
    print(f"Encoding {len(df)} images...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mvae = MVAE(model_name=args.model_name, modality=args.modality).to(device)
    if args.finetuned_ckpt:
        load_finetuned_weights(mvae, args.finetuned_ckpt)
    mvae.to(device).eval().requires_grad_(False)

    latents, splits, uuids, angles = [], [], [], []
    for i, row in df.iterrows():
        img_path = args.img_dir / f"{row['image_uuid']}.jpg"
        img = load_2d_finetune_aspect(str(img_path), merge_channels=True)
        img = img.unsqueeze(0).to(device)               # (1, 1, 384, 384)
        z = encode(mvae, img).squeeze(0).cpu().numpy()  # (C, H, W)
        latents.append(z)
        splits.append(row["split"])
        uuids.append(row["image_uuid"])
        angles.append([row["angle_proximal_thoracic"],
                       row["angle_main_thoracic"],
                       row["angle_thoracolumbar"]])
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"  [{i+1}/{len(df)}]")

    latents = np.stack(latents).astype(np.float32)
    angles = np.array(angles, dtype=np.float32)
    print(f"\nLatents shape: {latents.shape}  "
          f"(N, C, H, W) — flat dim per image = {np.prod(latents.shape[1:])}")
    print(f"Angles shape:  {angles.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             latents=latents,
             angles=angles,
             splits=np.array(splits),
             uuids=np.array(uuids))
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
