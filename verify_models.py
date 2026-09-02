"""
verify_models.py — confirm the four native MedVAE 2D checkpoints load, and
report what latent each produces. Chiefly to confirm `medvae_8_4_2d` (inferred
from the paper's f=64/C=4 row) actually exists before we extract it.

Total compression ratio for a grayscale input = f / C, where f is the 2D-area
downsizing factor (16 for the '4x' models, 64 for the '8x' models) and C is the
latent channel count.

Run:
  python verify_models.py --img ./medvae/data/boostnet/<any_uuid>.jpg
"""
import argparse
from pathlib import Path

import torch
from medvae import MVAE
from medvae.utils.loaders import load_2d_finetune_aspect

CONFIGS = ["medvae_4_3_2d", "medvae_4_1_2d", "medvae_8_4_2d", "medvae_8_1_2d"]
F_AREA = {"4": 16, "8": 64}   # area downsizing factor by per-dim spatial factor


@torch.no_grad()
def encode(mvae, x):
    out = mvae.model.encode(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        z = out
    elif hasattr(out, "mode"):
        z = out.mode()
    elif hasattr(out, "mean"):
        z = out.mean
    else:
        z = out.sample()
    return z


def match_in(x, n):
    return x if x.shape[1] == n else x.repeat(1, n, 1, 1)


@torch.no_grad()
def probe_in_channels(mvae, x1, device):
    for c in (1, 3, 4):
        try:
            encode(mvae, match_in(x1, c).to(device))
            return c
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True, help="any spine image, to probe latent shape")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_2d_finetune_aspect(str(args.img), merge_channels=True).unsqueeze(0).to(device)

    print(f"{'config':<18}{'loads':<8}{'in→latent (C,H,W)':<22}{'total compression':<18}")
    print("-" * 66)
    for cfg in CONFIGS:
        try:
            mvae = MVAE(model_name=cfg, modality="xray").to(device).eval()
            n_in = probe_in_channels(mvae, x, device)
            if n_in is None:
                print(f"{cfg:<18}{'FAIL':<8}could not determine input channels")
                continue
            z = encode(mvae, match_in(x, n_in)).squeeze(0)     # (C,H,W)
            C, H, W = z.shape
            spatial = cfg.split("_")[1]                        # '4' or '8'
            ratio = F_AREA.get(spatial, float("nan")) / C
            print(f"{cfg:<18}{'OK':<8}{f'{n_in}ch → {tuple(z.shape)}':<22}"
                  f"{ratio:>6.1f}x  (f={F_AREA.get(spatial)}, C={C})")
            del mvae
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{cfg:<18}{'FAIL':<8}{str(e)[:60]}")


if __name__ == "__main__":
    main()
