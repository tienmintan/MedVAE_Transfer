"""
visualize_all_models.py

Combined comparison figure showing the full project's model lineup in one image.

Top section (per BoostNet test image, n=3):
    | Original | MedVAE 4× pretrained | MedVAE 4× patch (tile-and-stitch) |
      MedVAE 8× pretrained | SD-VAE 512×512 |

Bottom section (n=3 synthetic samples per generative model):
    Row 1: Unconditional diffusion samples (v-prediction, VinDr+BoostNet pretrain)
    Row 2: Conditional diffusion samples (fine-tuned on BoostNet)

Display conventions:
  - All grayscale panels share percentile-based global windowing (default 1-99%)
    so cross-panel comparison reflects real differences, not auto-scaling.
  - All panels are square-aspect for clean column alignment; the patch
    tile-and-stitch output (native ~1:2.8 aspect) is aspect-padded to square
    at display time without modifying the underlying reconstruction.
  - A single colorbar on the right shows the active intensity scale.

Memory management:
  Models are loaded sequentially: MedVAE 4× pretrained → reconstruct 3 → free,
  then MedVAE 4× patch → reconstruct 3 → free, etc. SD-VAE is loaded once and
  reused for both BoostNet reconstruction AND diffusion sample decoding.

Usage:
    python visualize_all_models.py \\
        --csv          ./medvae/data/boostnet.csv \\
        --img-dir      ./medvae/data/boostnet \\
        --patch-ckpt   $PATCH_CKPT \\
        --uncond-npz   ./outputs/synthetic/unconditional_pretrain_v.npz \\
        --cond-npz     ./outputs/synthetic/synthetic_finetune_v_500.npz \\
        --out          ./outputs/figures/all_models_comparison.png \\
        --n-images     3 \\
        --seed         42
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL as SDAutoencoderKL

from medvae.models.autoencoder_kl import AutoencoderKL as MedVAEAutoencoderKL


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def load_and_aspect_pad(img_path: Path, target_size: int) -> torch.Tensor:
    """Load grayscale image, aspect-pad to square with black borders,
    resize to target_size x target_size, normalise to [-1, 1].

    Returns: (1, target_size, target_size) tensor in [-1, 1].
    """
    pil = Image.open(img_path).convert("L")
    W, H = pil.size
    # Pad to square with zeros (black) on the short side
    if H > W:
        pad = (H - W) // 2
        padded = Image.new("L", (H, H), 0)
        padded.paste(pil, (pad, 0))
    else:
        pad = (W - H) // 2
        padded = Image.new("L", (W, W), 0)
        padded.paste(pil, (0, pad))
    resized = padded.resize((target_size, target_size), Image.BICUBIC)
    arr = np.array(resized).astype(np.float32) / 255.0
    # Per-image min-max to [0, 1] then [-1, 1]
    amin, amax = arr.min(), arr.max()
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)


def aspect_pad_to_square(img: np.ndarray) -> np.ndarray:
    """Aspect-pad a 2D numpy image to square with zeros on the short side.
    Used at display time for tile-and-stitch outputs which are native aspect.
    """
    H, W = img.shape
    if H == W:
        return img
    side = max(H, W)
    out = np.zeros((side, side), dtype=img.dtype)
    if H > W:
        pad = (H - W) // 2
        out[:, pad:pad + W] = img
    else:
        pad = (W - H) // 2
        out[pad:pad + H, :] = img
    return out


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_medvae_pretrained(variant: str, device: torch.device) -> MedVAEAutoencoderKL:
    """Load a pretrained MedVAE variant from HuggingFace.

    variant: '4x1' for 4×/1ch, '8x1' for 8×/1ch.
    """
    cfg_name = f"model_weights/medvae_{variant}.yaml"
    if variant == "4x1":
        ckpt_name = "model_weights/vae_4x_1c_2D.ckpt"
    elif variant == "8x1":
        ckpt_name = "model_weights/vae_8x_1c_2D.ckpt"
    else:
        raise ValueError(f"Unknown MedVAE variant: {variant}")
    cfg_path = hf_hub_download(repo_id="stanfordmimi/MedVAE", filename=cfg_name)
    ckpt_path = hf_hub_download(repo_id="stanfordmimi/MedVAE", filename=ckpt_name)
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    model = MedVAEAutoencoderKL(**cfg, ckpt_path=ckpt_path).to(device).eval()
    return model


def load_medvae_patch_finetuned(ckpt_dir: Path, device: torch.device) -> MedVAEAutoencoderKL:
    """Load the patch-finetuned MedVAE 4×/1ch model from a checkpoint directory.

    Accelerate's `best.pt` is a directory of sharded weights; we load
    pytorch_model.bin (just the AE) with strict=False.
    """
    # Architecture from pretrained 4×/1ch
    cfg_path = hf_hub_download(repo_id="stanfordmimi/MedVAE",
                                filename="model_weights/medvae_4x1.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    # Initialise architecture without weights (no ckpt_path)
    model = MedVAEAutoencoderKL(**cfg).to(device).eval()

    # Load finetuned weights
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.is_dir():
        weight_file = ckpt_path / "pytorch_model.bin"
    else:
        weight_file = ckpt_path
    if not weight_file.exists():
        raise FileNotFoundError(f"Cannot find weights at {weight_file}")

    sd = torch.load(weight_file, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys when loading patch ckpt")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys when loading patch ckpt")
    return model


def load_sdvae(device: torch.device, dtype: torch.dtype = torch.float16) -> SDAutoencoderKL:
    vae = SDAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                          torch_dtype=dtype).to(device)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
# Reconstruction functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_medvae(model, img_neg1to1: torch.Tensor, device) -> np.ndarray:
    """Encode + decode through MedVAE.

    img_neg1to1: (1, H, W) in [-1, 1].
    Returns: (H, W) numpy array in [0, 1].
    """
    inp = img_neg1to1.unsqueeze(0).to(device).float()
    posterior = model.encode(inp)
    z = posterior.sample()
    out = model.decode(z)
    out = out.squeeze(0).squeeze(0).float().cpu().clamp(-1, 1).numpy()
    return (out + 1.0) / 2.0


@torch.no_grad()
def reconstruct_medvae_patch_tilestitch(model, img_path: Path, device,
                                          tile_size: int = 384,
                                          stride: int = 192) -> np.ndarray:
    """Tile-and-stitch reconstruction at native resolution.

    Returns: (H, W) numpy array in [0, 1] at the original image's resolution.
    """
    pil = Image.open(img_path).convert("L")
    orig_arr = np.array(pil).astype(np.float32) / 255.0
    H, W = orig_arr.shape

    # Reflect-pad to nearest stride multiple
    pad_h = ((-H) % stride) + tile_size
    pad_w = ((-W) % stride) + tile_size
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    padded = np.pad(orig_arr,
                     ((pad_h_half, pad_h - pad_h_half),
                      (pad_w_half, pad_w - pad_w_half)),
                     mode="reflect")
    Hp, Wp = padded.shape

    # Cosine window for blending
    y = np.arange(tile_size); x = np.arange(tile_size)
    win_y = 0.5 * (1 - np.cos(2 * np.pi * y / (tile_size - 1)))
    win_x = 0.5 * (1 - np.cos(2 * np.pi * x / (tile_size - 1)))
    window = (win_y[:, None] * win_x[None, :]).astype(np.float32)

    out = np.zeros_like(padded)
    weight = np.zeros_like(padded)

    for y0 in list(range(0, Hp - tile_size + 1, stride)) + [Hp - tile_size]:
        for x0 in list(range(0, Wp - tile_size + 1, stride)) + [Wp - tile_size]:
            tile = padded[y0:y0 + tile_size, x0:x0 + tile_size]
            tmin, tmax = tile.min(), tile.max()
            if tmax > tmin:
                norm = (tile - tmin) / (tmax - tmin)
            else:
                norm = np.zeros_like(tile)
            norm = (norm - 0.5) / 0.5
            inp = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0).to(device)
            posterior = model.encode(inp)
            z = posterior.sample()
            recon = model.decode(z)
            recon = recon.squeeze().float().cpu().clamp(-1, 1).numpy()
            recon = (recon + 1.0) / 2.0
            # De-normalise back to tile's original range
            recon = recon * (tmax - tmin) + tmin
            out[y0:y0 + tile_size, x0:x0 + tile_size] += recon * window
            weight[y0:y0 + tile_size, x0:x0 + tile_size] += window

    stitched = out / np.clip(weight, 1e-6, None)
    # Crop padding back off
    cropped = stitched[pad_h_half:pad_h_half + H, pad_w_half:pad_w_half + W]
    return cropped


@torch.no_grad()
def reconstruct_sdvae(vae, img_path: Path, device, target_size: int = 512) -> np.ndarray:
    """Aspect-pad + resize input to target_size, encode/decode through SD-VAE.

    Returns: (target_size, target_size) numpy array in [0, 1].
    """
    img_neg1to1 = load_and_aspect_pad(img_path, target_size)   # (1, T, T)
    img_3ch = img_neg1to1.unsqueeze(0).repeat(1, 3, 1, 1).to(device=device,
                                                              dtype=vae.dtype)
    z = vae.encode(img_3ch).latent_dist.mode()
    out = vae.decode(z).sample            # (1, 3, T, T) in [-1, 1]
    out_gray = out.mean(dim=1).squeeze(0).float().cpu().clamp(-1, 1).numpy()
    return (out_gray + 1.0) / 2.0


@torch.no_grad()
def decode_diffusion_samples(latents: np.ndarray, vae, device,
                              n_show: int = 3) -> list:
    """Decode synthetic SD-VAE latents through SD-VAE.

    latents: (N, 4, H_lat, W_lat) numpy array of un-standardised latents
             (matches what sample_diffusion.py writes natively).
    Returns: list of n_show (H, W) numpy arrays in [0, 1].
    """
    n = min(n_show, len(latents))
    outputs = []
    for i in range(n):
        z = torch.from_numpy(latents[i:i+1]).to(device=device, dtype=vae.dtype)
        decoded = vae.decode(z).sample                # (1, 3, H, W) in [-1, 1]
        gray = decoded.mean(dim=1).squeeze(0).float().cpu().clamp(-1, 1).numpy()
        outputs.append((gray + 1.0) / 2.0)
    return outputs


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def compute_global_bounds(arrays, mode: str, low_pct: float, high_pct: float):
    if mode == "fixed":
        return 0.0, 1.0
    flat = np.concatenate([a.ravel() for a in arrays])
    if mode == "minmax":
        return float(flat.min()), float(flat.max())
    elif mode == "percentile":
        return (float(np.percentile(flat, low_pct)),
                float(np.percentile(flat, high_pct)))
    else:
        raise ValueError(f"Unknown window mode: {mode}")


# ---------------------------------------------------------------------------
# Plot assembly
# ---------------------------------------------------------------------------

def build_figure(records, gen_records, vmin, vmax, out_path: Path,
                  window_mode: str, window_low: float, window_high: float):
    """Assemble the two-section comparison figure.

    records: list of dicts, one per test image. Each has keys:
      "uuid", "Original", and one key per reconstruction model column name.
    gen_records: dict mapping generative-model-name → list of (H, W) arrays.

    Layout: two separate subplot grids stacked vertically with a horizontal
    divider line between. Each grid uses an extra leftmost "label column"
    sized to hold the longest row label, so labels never get truncated.
    Single shared colorbar on the right.
    """
    # Determine which columns exist in the reconstruction section
    recon_cols = ["Original", "MedVAE 4× pretrained",
                   "MedVAE 4× patch (tile-and-stitch)",
                   "MedVAE 8× pretrained", "SD-VAE 512×512"]
    present_recon_cols = [c for c in recon_cols
                          if any(r.get(c) is not None for r in records)]
    n_recon_rows = len(records)
    n_recon_cols = len(present_recon_cols)

    gen_model_names = [k for k, v in gen_records.items() if v]
    n_gen_rows = len(gen_model_names)
    n_gen_cols = max((len(v) for v in gen_records.values()), default=0)

    # Figure sizing — each panel is ~3 inches square, plus dedicated label
    # column (1.6") and colorbar gutter on the right (1.0").
    panel = 3.0
    label_w = 1.6
    cbar_gutter = 1.0
    n_total_cols = max(n_recon_cols, n_gen_cols)
    fig_w = label_w + panel * n_total_cols + cbar_gutter

    # Vertical: recon block + divider strip + gen block + caption strip
    recon_h = panel * n_recon_rows
    gen_h   = panel * n_gen_rows
    divider_h = 0.6 if n_gen_rows > 0 else 0
    fig_h = 0.7 + recon_h + divider_h + gen_h + 0.7   # 0.7 top/bottom margins

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    # Master layout: 3 rows (recon block, divider, gen block), 2 cols
    # (main panel area, colorbar gutter). The label column lives INSIDE
    # the main panel area as its leftmost gridspec column.
    height_ratios = []
    if n_recon_rows > 0:  height_ratios.append(recon_h)
    if n_gen_rows > 0:    height_ratios.extend([divider_h, gen_h])
    master = fig.add_gridspec(
        nrows=len(height_ratios), ncols=2,
        width_ratios=[panel * n_total_cols + label_w, cbar_gutter],
        height_ratios=height_ratios,
        left=0.02, right=0.98, top=0.93, bottom=0.06,
        hspace=0.05, wspace=0.02,
    )

    last_handle = None

    # =============== Reconstruction block ===============
    if n_recon_rows > 0:
        recon_block = master[0, 0].subgridspec(
            nrows=n_recon_rows, ncols=n_recon_cols + 1,
            width_ratios=[label_w] + [panel] * n_recon_cols,
            wspace=0.02, hspace=0.03,
        )
        for i, rec in enumerate(records):
            # Label cell
            lab_ax = fig.add_subplot(recon_block[i, 0])
            lab_ax.text(0.95, 0.5,
                         rec.get("uuid", "")[:25],
                         ha="right", va="center", fontsize=9,
                         family="monospace", transform=lab_ax.transAxes)
            lab_ax.axis("off")
            # Image cells
            for j, col_name in enumerate(present_recon_cols):
                ax = fig.add_subplot(recon_block[i, j + 1])
                img = rec.get(col_name)
                if img is None:
                    ax.text(0.5, 0.5, "(missing)",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=10, color="gray")
                    ax.axis("off")
                else:
                    last_handle = ax.imshow(
                        img, cmap="gray", vmin=vmin, vmax=vmax,
                        aspect="equal")
                    ax.axis("off")
                if i == 0:
                    ax.set_title(col_name, fontsize=11, pad=6)

    # =============== Section divider + bottom header ===============
    if n_gen_rows > 0:
        div_ax = fig.add_subplot(master[1, 0])
        div_ax.set_xlim(0, 1); div_ax.set_ylim(0, 1)
        # Horizontal rule
        div_ax.axhline(0.65, color="black", lw=0.8, xmin=0.02, xmax=0.98)
        # Header text immediately below the rule
        div_ax.text(0.5, 0.20,
                     "Generations: synthetic samples from "
                     "conditional latent diffusion",
                     ha="center", va="center", fontsize=13, weight="bold",
                     transform=div_ax.transAxes)
        div_ax.axis("off")

    # =============== Generation block ===============
    if n_gen_rows > 0:
        gen_block = master[2, 0].subgridspec(
            nrows=n_gen_rows, ncols=n_gen_cols + 1,
            width_ratios=[label_w] + [panel] * n_gen_cols,
            wspace=0.02, hspace=0.03,
        )
        for i, model_name in enumerate(gen_model_names):
            samples = gen_records[model_name]
            # Label cell
            lab_ax = fig.add_subplot(gen_block[i, 0])
            lab_ax.text(0.95, 0.5, model_name,
                         ha="right", va="center", fontsize=9,
                         weight="bold", transform=lab_ax.transAxes)
            lab_ax.axis("off")
            # Image cells
            for j in range(n_gen_cols):
                ax = fig.add_subplot(gen_block[i, j + 1])
                if j < len(samples):
                    last_handle = ax.imshow(
                        samples[j], cmap="gray", vmin=vmin, vmax=vmax,
                        aspect="equal")
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(f"Sample {j + 1}", fontsize=11, pad=6)
                else:
                    ax.axis("off")

    # =============== Colorbar ===============
    if last_handle is not None:
        cbar_master_row = len(height_ratios) - 1   # span the gen block height
        # If there's only the recon block, colorbar spans master[0, 1].
        # Otherwise span all rows of column 1.
        if n_gen_rows > 0:
            cbar_ax = fig.add_subplot(master[:, 1])
        else:
            cbar_ax = fig.add_subplot(master[0, 1])
        # Shrink the colorbar axes a bit so it doesn't stretch the full height
        cbar_pos = cbar_ax.get_position()
        new_x = cbar_pos.x0 + 0.01
        new_w = 0.012
        new_y = cbar_pos.y0 + 0.05 * cbar_pos.height
        new_h = 0.90 * cbar_pos.height
        cbar_ax.set_position([new_x, new_y, new_w, new_h])
        cbar = fig.colorbar(last_handle, cax=cbar_ax)
        cbar.set_label("Normalised pixel intensity", rotation=270,
                        labelpad=15, fontsize=10)

    # =============== Suptitle (top section header) ===============
    fig.suptitle(
        "Reconstructions: pixel-level fidelity comparison "
        "across MedVAE and SD-VAE",
        fontsize=14, weight="bold", y=0.985,
    )

    # =============== Bottom caption ===============
    if window_mode == "percentile":
        caption = (f"Display windowing: {window_low:.0f}th–{window_high:.0f}th "
                    f"percentile across all panels (vmin={vmin:.3f}, "
                    f"vmax={vmax:.3f}). Same scale for every panel.")
    elif window_mode == "minmax":
        caption = (f"Display windowing: global min/max across all panels "
                    f"(vmin={vmin:.3f}, vmax={vmax:.3f}).")
    else:
        caption = "Display windowing: fixed [0, 1]."
    fig.text(0.5, 0.015, caption, ha="center", va="bottom",
              fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path,
                    help="BoostNet labels CSV (must have a 'split' column).")
    p.add_argument("--img-dir", required=True, type=Path,
                    help="Directory containing BoostNet image files.")
    p.add_argument("--patch-ckpt", type=str, default=None,
                    help="Path to patch-finetuned MedVAE checkpoint. "
                         "Omit to skip the patch column.")
    p.add_argument("--uncond-npz", type=Path, default=None,
                    help="Path to unconditional diffusion samples .npz. "
                         "Omit to skip the unconditional row.")
    p.add_argument("--cond-npz", type=Path, default=None,
                    help="Path to conditional diffusion samples .npz. "
                         "Omit to skip the conditional row.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-mode", choices=["percentile", "minmax", "fixed"],
                    default="percentile")
    p.add_argument("--window-low",  type=float, default=1.0)
    p.add_argument("--window-high", type=float, default=99.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------ #
    # Pick test images
    # ------------------------------------------------------------------ #
    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    selected_idx = rng.choice(len(test_df), size=args.n_images, replace=False)
    selected = test_df.iloc[selected_idx].reset_index(drop=True)
    img_paths = [args.img_dir / f"{u}.jpg" for u in selected["image_uuid"]]
    print(f"Selected {args.n_images} test images:")
    for p_ in img_paths:
        print(f"  {p_.name}")

    # Pre-load originals (aspect-padded to 384×384 for display consistency
    # with the MedVAE 4× and 8× outputs)
    records = []
    for img_path, uuid in zip(img_paths, selected["image_uuid"]):
        orig_neg1to1 = load_and_aspect_pad(img_path, target_size=384)
        orig_disp = (orig_neg1to1.squeeze(0).numpy() + 1.0) / 2.0
        records.append({"uuid": uuid, "Original": orig_disp})

    # ------------------------------------------------------------------ #
    # MedVAE 4× pretrained
    # ------------------------------------------------------------------ #
    print("\n[1/5] MedVAE 4× pretrained...")
    try:
        m_pre_4x = load_medvae_pretrained("4x1", device)
        for rec, img_path in zip(records, img_paths):
            inp = load_and_aspect_pad(img_path, target_size=384)
            rec["MedVAE 4× pretrained"] = reconstruct_medvae(m_pre_4x, inp, device)
        del m_pre_4x
        torch.cuda.empty_cache()
    except Exception as e:
        warnings.warn(f"MedVAE 4× pretrained failed: {e}")
        for rec in records:
            rec["MedVAE 4× pretrained"] = None

    # ------------------------------------------------------------------ #
    # MedVAE 4× patch (tile-and-stitch)
    # ------------------------------------------------------------------ #
    if args.patch_ckpt:
        print(f"\n[2/5] MedVAE 4× patch (tile-and-stitch) from {args.patch_ckpt}...")
        try:
            m_patch = load_medvae_patch_finetuned(args.patch_ckpt, device)
            for rec, img_path in zip(records, img_paths):
                recon = reconstruct_medvae_patch_tilestitch(m_patch, img_path, device)
                rec["MedVAE 4× patch (tile-and-stitch)"] = aspect_pad_to_square(recon)
            del m_patch
            torch.cuda.empty_cache()
        except Exception as e:
            warnings.warn(f"MedVAE 4× patch failed: {e}")
            for rec in records:
                rec["MedVAE 4× patch (tile-and-stitch)"] = None
    else:
        print("\n[2/5] Skipping MedVAE 4× patch (no --patch-ckpt provided)")
        for rec in records:
            rec["MedVAE 4× patch (tile-and-stitch)"] = None

    # ------------------------------------------------------------------ #
    # MedVAE 8× pretrained
    # ------------------------------------------------------------------ #
    print("\n[3/5] MedVAE 8× pretrained...")
    try:
        m_pre_8x = load_medvae_pretrained("8x1", device)
        for rec, img_path in zip(records, img_paths):
            inp = load_and_aspect_pad(img_path, target_size=384)
            rec["MedVAE 8× pretrained"] = reconstruct_medvae(m_pre_8x, inp, device)
        del m_pre_8x
        torch.cuda.empty_cache()
    except Exception as e:
        warnings.warn(f"MedVAE 8× pretrained failed: {e}")
        for rec in records:
            rec["MedVAE 8× pretrained"] = None

    # ------------------------------------------------------------------ #
    # SD-VAE — used for both BoostNet recon AND diffusion sample decode
    # ------------------------------------------------------------------ #
    print("\n[4/5] SD-VAE 512×512 (also used for diffusion decoding)...")
    sdvae = load_sdvae(device)
    try:
        for rec, img_path in zip(records, img_paths):
            rec["SD-VAE 512×512"] = reconstruct_sdvae(sdvae, img_path, device, 512)
    except Exception as e:
        warnings.warn(f"SD-VAE reconstruction failed: {e}")
        for rec in records:
            rec["SD-VAE 512×512"] = None

    # ------------------------------------------------------------------ #
    # Diffusion samples (decoded through the same SD-VAE)
    # ------------------------------------------------------------------ #
    print("\n[5/5] Decoding diffusion samples...")
    gen_records = {}

    if args.uncond_npz and args.uncond_npz.exists():
        try:
            data = np.load(args.uncond_npz, allow_pickle=True)
            latents = data["latents"]
            print(f"  Loaded {len(latents)} unconditional samples "
                  f"(shape {latents.shape}) from {args.uncond_npz.name}")
            decoded = decode_diffusion_samples(latents, sdvae, device,
                                                 n_show=args.n_images)
            gen_records["Unconditional diffusion\n(v-pred, VinDr+BoostNet pretrain)"] = decoded
        except Exception as e:
            warnings.warn(f"Decoding unconditional samples failed: {e}")
            gen_records["Unconditional diffusion\n(v-pred, VinDr+BoostNet pretrain)"] = []
    else:
        print("  Skipping unconditional row (no --uncond-npz)")

    if args.cond_npz and args.cond_npz.exists():
        try:
            data = np.load(args.cond_npz, allow_pickle=True)
            latents = data["latents"]
            print(f"  Loaded {len(latents)} conditional samples "
                  f"(shape {latents.shape}) from {args.cond_npz.name}")
            decoded = decode_diffusion_samples(latents, sdvae, device,
                                                 n_show=args.n_images)
            gen_records["Conditional diffusion\n(BoostNet fine-tune)"] = decoded
        except Exception as e:
            warnings.warn(f"Decoding conditional samples failed: {e}")
            gen_records["Conditional diffusion\n(BoostNet fine-tune)"] = []
    else:
        print("  Skipping conditional row (no --cond-npz)")

    del sdvae
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Compute global windowing bounds, render figure
    # ------------------------------------------------------------------ #
    all_arrays = []
    for rec in records:
        for v in rec.values():
            if isinstance(v, np.ndarray):
                all_arrays.append(v)
    for samples in gen_records.values():
        all_arrays.extend(samples)

    vmin, vmax = compute_global_bounds(all_arrays, args.window_mode,
                                         args.window_low, args.window_high)
    print(f"\nGlobal intensity window ({args.window_mode}): "
          f"[{vmin:.3f}, {vmax:.3f}]")

    build_figure(records, gen_records, vmin, vmax, args.out,
                  args.window_mode, args.window_low, args.window_high)


if __name__ == "__main__":
    main()
