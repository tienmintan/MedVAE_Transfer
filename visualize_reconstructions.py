"""
visualize_reconstructions.py

Quick qualitative check: pretrained MedVAE vs finetuned MedVAE on a handful of
test images. Saves a single PNG with rows = images, columns = [original,
pretrained recon, finetuned recon, |orig - finetuned|].

All grayscale image panels (cols 0–2) share the SAME intensity windowing —
vmin/vmax computed globally across every pixel we display. This matches the
medical-image viewing convention (window/level): the display bounds are
fixed deliberately so visible differences reflect real differences in the
data rather than per-panel auto-scaling artefacts. A colorbar on the right
shows the active scale. The |diff| column (col 3, 'hot' colormap) uses its
own scale, also computed globally across all diff panels, with its bounds
stated in the caption.

IMPORTANT: this script intentionally uses load_2d_finetune (resize to 384x384)
rather than mvae.apply_transform() (which uses load_2d, NO resize). The model
was trained at 384x384, and feeding native-resolution BoostNet images
(~880x2480) into the encoder explodes the bottleneck self-attention to ~74 GB.

Usage:
  python visualize_reconstructions.py \\
      --csv      $PROJECT_DIR/medvae/data/boostnet.csv \\
      --img-dir  $PROJECT_DIR/medvae/data/boostnet \\
      --finetuned-ckpt $PROJECT_DIR/medvae/logs/boostnet_finetuning/runs/<TS>/checkpoints/best.pt/pytorch_model.bin \\
      --out      ./recon_comparison.png \\
      --n        5 \\
      --window-mode percentile \\
      --seed     0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune


def load_finetuned_weights(mvae: MVAE, ckpt_path: Path) -> None:
    """Load finetuned AE weights into the underlying AutoencoderKL_2D.

    Accepts either:
      - a path to the .bin file directly (e.g. .../best.pt/pytorch_model.bin), or
      - a path to the accelerate-style checkpoint directory (e.g. .../best.pt/),
        in which case we look for pytorch_model.bin inside it.
    The directory form matches the convention used by the tile-and-stitch
    evaluation scripts, so the same env var can be used across scripts.
    """
    if ckpt_path.is_dir():
        bin_path = ckpt_path / "pytorch_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} is a directory but no pytorch_model.bin inside. "
                f"Files present: {sorted(p.name for p in ckpt_path.iterdir())}")
        ckpt_path = bin_path
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = mvae.model.load_state_dict(sd, strict=False)
    print(f"Loaded {ckpt_path}")
    print(f"  missing keys:    {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")


def transform_image(path: str) -> torch.Tensor:
    """Apply the SAME transform used during finetuning (resize to 384x384)."""
    img = load_2d_finetune(path, merge_channels=True)   # (1, 384, 384) in [-1, 1]
    return img.unsqueeze(0)                              # (1, 1, 384, 384)


@torch.no_grad()
def reconstruct(mvae: MVAE, img: torch.Tensor) -> torch.Tensor:
    mvae.eval()
    dec, _ = mvae(img, decode=True)
    return dec


def to_display(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor → [0, 1] numpy array, clipped, on CPU."""
    return ((x.detach().float().cpu().clamp(-1, 1) + 1) / 2).numpy()


def compute_bounds(arrays, mode: str, low_pct: float, high_pct: float):
    """Compute (vmin, vmax) for windowing across all displayed pixels."""
    if mode == "fixed":
        return 0.0, 1.0
    flat = np.concatenate([a.ravel() for a in arrays])
    if mode == "minmax":
        return float(flat.min()), float(flat.max())
    elif mode == "percentile":
        return (float(np.percentile(flat, low_pct)),
                float(np.percentile(flat, high_pct)))
    else:
        raise ValueError(f"Unknown window-mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--finetuned-ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", default="medvae_4_1_2d")
    p.add_argument("--modality", default="xray")
    p.add_argument("--split", default="test")
    p.add_argument("--window-mode", choices=["percentile", "minmax", "fixed"],
                   default="percentile",
                   help="How to pick the global vmin/vmax for windowing the "
                        "grayscale image columns. 'percentile' is robust to "
                        "bright outliers; 'minmax' uses raw global bounds; "
                        "'fixed' uses [0, 1].")
    p.add_argument("--window-low",  type=float, default=1.0,
                   help="Lower percentile for 'percentile' mode (default 1).")
    p.add_argument("--window-high", type=float, default=99.0,
                   help="Upper percentile for 'percentile' mode (default 99).")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"No rows with split={args.split} in {args.csv}")
    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    paths = [args.img_dir / f"{u}.jpg" for u in sample["image_uuid"]]
    print(f"Sampling {n} images from split='{args.split}':")
    for p_ in paths:
        print(f"  {p_.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading pretrained MedVAE...")
    pretrained = MVAE(model_name=args.model_name, modality=args.modality).to(device)
    pretrained.requires_grad_(False)

    print("Loading finetuned MedVAE...")
    finetuned = MVAE(model_name=args.model_name, modality=args.modality).to(device)
    load_finetuned_weights(finetuned, args.finetuned_ckpt)
    finetuned.to(device)
    finetuned.requires_grad_(False)

    # =========================================================================
    # PASS 1: collect everything we'll display
    # =========================================================================
    print("\nRunning reconstructions...")
    records = []
    for i, fpath in enumerate(paths):
        img = transform_image(str(fpath)).to(device)
        recon_pre = reconstruct(pretrained, img)
        recon_ft  = reconstruct(finetuned, img)

        orig_np = to_display(img.squeeze())
        pre_np  = to_display(recon_pre)
        ft_np   = to_display(recon_ft)
        diff_np = np.abs(orig_np - ft_np)

        l1_pre = float(np.abs(orig_np - pre_np).mean())
        l1_ft  = float(np.abs(orig_np - ft_np).mean())
        print(f"  [{i+1}/{n}] {fpath.name}  L1 pretrained={l1_pre:.4f}  "
              f"finetuned={l1_ft:.4f}  Δ={l1_pre - l1_ft:+.4f}")

        records.append({
            "name":    fpath.name,
            "orig":    orig_np,  "pre": pre_np,  "ft": ft_np,
            "diff":    diff_np,
            "l1_pre":  l1_pre,   "l1_ft": l1_ft,
        })

    # =========================================================================
    # PASS 2: global windowing bounds
    # =========================================================================
    grayscale_arrays = [r[k] for r in records for k in ("orig", "pre", "ft")]
    vmin, vmax = compute_bounds(
        grayscale_arrays, args.window_mode,
        args.window_low, args.window_high)
    print(f"\nGrayscale window ({args.window_mode}): "
          f"[{vmin:.4f}, {vmax:.4f}]")

    # Diff column has its own scale — use the same percentile policy on the
    # diff distribution for consistency. (Diff values are non-negative, so we
    # pin the lower bound at 0 and only compute the upper.)
    diff_arrays = [r["diff"] for r in records]
    if args.window_mode == "fixed":
        diff_vmax = 0.5
    else:
        diff_flat = np.concatenate([a.ravel() for a in diff_arrays])
        if args.window_mode == "minmax":
            diff_vmax = float(diff_flat.max())
        else:
            diff_vmax = float(np.percentile(diff_flat, args.window_high))
    print(f"Diff window: [0, {diff_vmax:.4f}]")

    # =========================================================================
    # PASS 3: render
    # =========================================================================
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3, n * 3))
    if n == 1:
        axes = axes[None, :]
    for c, t in enumerate(["Original", "Pretrained", "Finetuned",
                           "|Original − Finetuned|"]):
        axes[0, c].set_title(t, fontsize=11)

    last_gray = None
    last_diff = None
    for i, r in enumerate(records):
        for c, key in enumerate(["orig", "pre", "ft"]):
            last_gray = axes[i, c].imshow(r[key], cmap="gray",
                                          vmin=vmin, vmax=vmax)
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])
        last_diff = axes[i, 3].imshow(r["diff"], cmap="hot",
                                      vmin=0, vmax=diff_vmax)
        axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])
        axes[i, 0].set_ylabel(f"L1: {r['l1_pre']:.3f} → {r['l1_ft']:.3f}",
                              fontsize=9)

    fig.tight_layout()
    # Make room for colorbars on the right
    fig.subplots_adjust(right=0.88)
    gray_cbar_ax = fig.add_axes([0.895, 0.18, 0.012, 0.64])
    diff_cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    gcb = fig.colorbar(last_gray, cax=gray_cbar_ax)
    gcb.set_label("Grayscale intensity", rotation=270, labelpad=12, fontsize=9)
    dcb = fig.colorbar(last_diff, cax=diff_cbar_ax)
    dcb.set_label("|Original − Finetuned|", rotation=270, labelpad=12, fontsize=9)

    # Caption
    if args.window_mode == "percentile":
        caption = (f"Display windowing: grayscale cols use "
                   f"{args.window_low:.0f}th–{args.window_high:.0f}th "
                   f"percentile of all panel pixels "
                   f"(vmin={vmin:.3f}, vmax={vmax:.3f}); diff col uses "
                   f"[0, {diff_vmax:.3f}]. Same scale within each colormap.")
    elif args.window_mode == "minmax":
        caption = (f"Display windowing: grayscale cols use global min/max "
                   f"(vmin={vmin:.3f}, vmax={vmax:.3f}); diff col uses "
                   f"[0, {diff_vmax:.3f}]. Same scale within each colormap.")
    else:
        caption = (f"Display windowing: grayscale cols fixed [0, 1]; diff col "
                   f"fixed [0, {diff_vmax:.3f}].")
    fig.text(0.5, 0.02, caption, ha="center", va="bottom",
             fontsize=9, style="italic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
