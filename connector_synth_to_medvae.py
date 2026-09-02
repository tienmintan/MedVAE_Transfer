"""
connector_synth_to_medvae.py

Convert synthetic SD-VAE latents from the diffusion pipeline (output of
sample_diffusion.py at SD-VAE format, shape (N, 4, 64, 64)) into MedVAE 4x/1ch
latents (shape (N, 1, 96, 96)) compatible with train_cobb_regressor.py.

Pipeline per synthetic latent:
  1. SD-VAE decode: (1, 4, 64, 64) → (1, 3, 512, 512) in [-1, 1]
  2. Convert RGB → grayscale by channel-mean
  3. Resize 512x512 → 384x384 (matches BoostNet preprocessing target)
  4. MedVAE 4x/1ch encode: (1, 1, 384, 384) → (1, 1, 96, 96)

Concatenate with real BoostNet latents from latents_pretrained_4x.npz and
write a combined .npz in the schema train_cobb_regressor.py expects:
  - latents  (N+M, 1, 96, 96)
  - angles   (N+M, 3)
  - splits   (N+M,) ["train"/"val"/"test"], synthetic all tagged "train"
  - uuids    (N+M,), synthetic prefixed with "synth_diff_"

Includes a sanity-check visualization decoding 4 synthetic MedVAE latents
back through MedVAE's pixel decoder, so we can verify the connector is
producing in-distribution MedVAE latents (recognizable spine X-rays).

Caveat: synthetic images may have different aspect-padding proportions
than real BoostNet (synthetic content fills more of the 384x384 frame).
This is a real distribution shift that the downstream regressor will
need to handle. Documented here; assessed by the regression result.

Usage:
    python connector_synth_to_medvae.py \\
        --synth-latents ./synthetic_finetune_v_500.npz \\
        --real-latents  ./latents_pretrained_4x.npz \\
        --out           ./latents_pretrained_4x_synthdiff500.npz \\
        --sanity-out    ./synth_connector_sanity.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import AutoencoderKL as SDAutoencoderKL
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from medvae.models.autoencoder_kl import AutoencoderKL as MedVAEAutoencoderKL


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sdvae(device, dtype):
    print("Loading Stable Diffusion VAE...")
    vae = SDAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                          torch_dtype=dtype).to(device)
    vae.eval()
    return vae


def load_medvae(device, dtype):
    print("Loading MedVAE 4x/1ch...")
    cfg_path = hf_hub_download(repo_id="stanfordmimi/MedVAE",
                                filename="model_weights/medvae_4x1.yaml")
    ckpt_path = hf_hub_download(repo_id="stanfordmimi/MedVAE",
                                 filename="model_weights/vae_4x_1c_2D.ckpt")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    model = MedVAEAutoencoderKL(**cfg, ckpt_path=ckpt_path).to(device).eval()
    # MedVAE's loader doesn't honour torch_dtype; keep it in fp32 to avoid
    # mixed-precision surprises during encode.
    return model


# ---------------------------------------------------------------------------
# Single-sample round-trip
# ---------------------------------------------------------------------------

@torch.no_grad()
def synth_sdvae_latent_to_medvae_latent(
        sd_latent: torch.Tensor,
        sdvae, medvae, device,
        sdvae_dtype=torch.float16):
    """Convert (1, 4, 64, 64) SD-VAE latent → (1, 1, 96, 96) MedVAE latent.

    Done per-sample to avoid OOM on the A10 (attention layers in both VAEs
    are at their bottleneck spatial extent and don't batch cheaply).
    """
    # ---- SD-VAE decode ----
    z_sd = sd_latent.to(device=device, dtype=sdvae_dtype)
    img_3ch = sdvae.decode(z_sd).sample              # (1, 3, 512, 512), [-1, 1]
    img_1ch = img_3ch.mean(dim=1, keepdim=True)      # (1, 1, 512, 512)
    img_1ch = img_1ch.clamp(-1.0, 1.0).float()

    # ---- Resize 512 → 384 ----
    img_384 = F.interpolate(img_1ch, size=(384, 384),
                             mode="bilinear", align_corners=False)

    # ---- MedVAE encode ----
    posterior = medvae.encode(img_384)
    z_med = posterior.sample()                       # (1, 1, 96, 96)
    return z_med.float(), img_384.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Sanity-check visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_sanity_visualization(
        sd_latents: np.ndarray, intermediate_imgs: list,
        medvae_latents: np.ndarray, medvae, device,
        out_path: Path, n_show: int = 4):
    """Side-by-side visualization for a few samples:
        col 1: 384x384 intermediate (after SD-VAE decode + resize)
        col 2: MedVAE-decoded image from the produced MedVAE latent

    Lets us verify the connector preserves anatomical structure end-to-end.
    """
    n = min(n_show, len(medvae_latents))

    decoded = []
    for i in range(n):
        z = torch.from_numpy(medvae_latents[i:i+1]).to(device=device,
                                                        dtype=torch.float32)
        img = medvae.decode(z)                        # (1, 1, 384, 384), ~[-1, 1]
        img = (img.float() + 1.0) / 2.0
        img = img.clamp(0, 1).squeeze().cpu().numpy()
        decoded.append(img)

    fig, axes = plt.subplots(n, 2, figsize=(7, 4 * n),
                              gridspec_kw={"hspace": 0.1, "wspace": 0.05})
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        # Convert intermediate from [-1, 1] to [0, 1] for display
        inter = (intermediate_imgs[i] + 1.0) / 2.0
        inter = np.clip(inter, 0, 1)
        axes[i, 0].imshow(inter, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(decoded[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")

    axes[0, 0].set_title("SD-VAE decode → 384x384", fontsize=10)
    axes[0, 1].set_title("MedVAE re-decode (from connector output)",
                          fontsize=10)
    fig.text(0.5, 0.02,
              "Connector pipeline sanity check. The right column shows what "
              "the regressor will 'see' after our connector — if both columns "
              "show recognizable spines, the connector preserves structure.",
              ha="center", fontsize=9, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved sanity visualization to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth-latents", required=True, type=Path,
                    help="NPZ from sample_diffusion.py — SD-VAE format "
                         "(latents shape (N, 4, 64, 64), with angles).")
    p.add_argument("--real-latents",  required=True, type=Path,
                    help="Real BoostNet MedVAE latents from extract_latents.py "
                         "(e.g. latents_pretrained_4x.npz).")
    p.add_argument("--out",           required=True, type=Path)
    p.add_argument("--sanity-out",    type=Path, default=None,
                    help="Optional path to save a sanity-check visualization "
                         "of 4 round-tripped samples.")
    p.add_argument("--n-synth",       type=int, default=None,
                    help="Limit number of synthetic samples to process "
                         "(default: all).")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--seed",          type=int, default=42,
                    help="For MedVAE posterior.sample().")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ---- Load data ----
    synth = np.load(args.synth_latents, allow_pickle=True)
    synth_lat = synth["latents"]
    synth_ang = synth["angles"]
    n_synth = min(args.n_synth or len(synth_lat), len(synth_lat))
    synth_lat = synth_lat[:n_synth]
    synth_ang = synth_ang[:n_synth]
    print(f"Loaded {n_synth} synthetic latents (SD-VAE format) "
          f"from {args.synth_latents}")
    print(f"  Shape: {synth_lat.shape}")
    print(f"  Stats: mean={synth_lat.mean():.3f} std={synth_lat.std():.3f}")

    real = np.load(args.real_latents, allow_pickle=True)
    real_lat = real["latents"]
    real_ang = real["angles"]
    real_splits = real["splits"].astype(str)
    real_uuids = real["uuids"].astype(str)
    print(f"\nLoaded {len(real_lat)} real latents (MedVAE format) "
          f"from {args.real_latents}")
    print(f"  Shape: {real_lat.shape}")
    print(f"  Split counts: " + ", ".join(
        f"{s}={(real_splits == s).sum()}" for s in sorted(set(real_splits))))

    # ---- Load both VAEs ----
    sdvae = load_sdvae(device, dtype=torch.float16)
    medvae = load_medvae(device, dtype=torch.float32)

    # ---- Round-trip each synthetic latent ----
    print(f"\nRound-trip pipeline (SD-VAE decode → resize → MedVAE encode)...")
    medvae_synth_latents = np.empty((n_synth, 1, 96, 96), dtype=np.float32)
    saved_intermediates = []

    # Pick a few indices for the sanity viz
    n_sanity = 4 if args.sanity_out is not None else 0
    sanity_indices = (set(np.linspace(0, n_synth - 1, n_sanity).astype(int))
                       if n_sanity > 0 else set())

    for i in tqdm(range(n_synth)):
        sd_z = torch.from_numpy(synth_lat[i:i+1].astype(np.float32))
        med_z, inter = synth_sdvae_latent_to_medvae_latent(
            sd_z, sdvae, medvae, device)
        medvae_synth_latents[i] = med_z.cpu().numpy().squeeze(0)
        if i in sanity_indices:
            saved_intermediates.append(inter)
        torch.cuda.empty_cache()

    print(f"\nSynthetic MedVAE latents produced:")
    print(f"  Shape: {medvae_synth_latents.shape}")
    print(f"  Stats: mean={medvae_synth_latents.mean():.3f} "
          f"std={medvae_synth_latents.std():.3f}")
    real_train = real_lat[real_splits == "train"]
    print(f"  Compared to real (train split): "
          f"mean={real_train.mean():.3f} std={real_train.std():.3f}")

    # ---- Sanity visualization ----
    if args.sanity_out is not None:
        sanity_lat = medvae_synth_latents[np.array(sorted(sanity_indices))]
        make_sanity_visualization(synth_lat, saved_intermediates,
                                    sanity_lat, medvae, device,
                                    args.sanity_out)

    # ---- Combine with real, write final ----
    synth_uuids = np.array([f"synth_diff_{i:06d}" for i in range(n_synth)])
    synth_splits = np.array(["train"] * n_synth)

    combined_lat = np.concatenate([real_lat, medvae_synth_latents], axis=0)
    combined_ang = np.concatenate([real_ang, synth_ang.astype(np.float32)],
                                    axis=0)
    combined_splits = np.concatenate([real_splits, synth_splits])
    combined_uuids = np.concatenate([real_uuids, synth_uuids])

    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_lat)}")
    n_real_train = (real_splits == "train").sum()
    print(f"  Train: {(combined_splits == 'train').sum()} "
          f"({n_real_train} real + {n_synth} synthetic)")
    print(f"  Val:   {(combined_splits == 'val').sum()} (unchanged)")
    print(f"  Test:  {(combined_splits == 'test').sum()} (unchanged)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        latents=combined_lat.astype(np.float32),
        angles=combined_ang.astype(np.float32),
        splits=combined_splits,
        uuids=combined_uuids,
    )
    print(f"\nWrote {args.out}")
    print(f"\nNext: train Cobb regressor with this augmented file via")
    print(f"  python train_cobb_regressor.py \\")
    print(f"      --latents          {args.out} \\")
    print(f"      --out              ./cobb_synth_diff_500.npz \\")
    print(f"      --baseline-results ./cobb_pretrained_4x.npz \\")
    print(f"      --seed             42")


if __name__ == "__main__":
    main()
