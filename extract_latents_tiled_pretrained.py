"""
extract_latents_tiled_pretrained.py: encode BoostNet images into tile-and-stitch latent grids with the pretrained MedVAE 4x/1ch encoder.

Ablation companion to extract_latents_tiled.py, which uses the
patch-finetuned encoder. The patch-finetuned tile-and-stitch configuration
(matched_tiled) combines two changes: native-resolution access via
tile-and-stitch, and patch-based encoder finetuning. This script provides
the pretrained encoder accessed via the same tile-and-stitch protocol, under
the same matched regressor, so that comparing it with matched_tiled
isolates the effect of patch finetuning.

Interpretation:
  pretrained_tiled MAE similar to matched_tiled -> tile-and-stitch access
      accounts for the difference; patch finetuning adds little.
  pretrained_tiled no better than the aspect-padded baseline -> patch
      finetuning is essential; native-resolution access alone does not help.
  pretrained_tiled in between -> both contribute.

Pipeline is identical to extract_latents_tiled.py in every respect
(resize, tile geometry, per-tile normalisation, cosine blending in latent
space) except that the encoder weights come from the HuggingFace
pretrained release rather than a local patch-finetuned checkpoint.

Preprocessing: per-tile min-max normalisation is kept even though the
pretrained encoder was trained on per-image-normalised inputs. Matching the
preprocessing of matched_tiled means the comparison between the two
conditions isolates the encoder weights. If the pretrained encoder
underperforms under this protocol, part of the effect of patch finetuning
is then adaptation of the encoder to per-tile-normalised inputs.

Inputs:  --csv (image_uuid, split); JPEG images in --img-dir; pretrained
         weights for --model-name (loaded via extract_latents_aspect.py).
Outputs: <out-dir>/latents_{train,val,test}.npz (compressed).

Usage:
    python extract_latents_tiled_pretrained.py \\
        --csv     ./medvae/data/boostnet.csv \\
        --img-dir ./medvae/data/boostnet \\
        --out-dir ./latents_pretrained_tiled
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_reconstruction_tiled import (
    make_cosine_window, tile_positions,
)
from extract_latents_aspect import load_medvae_variant, MODEL_REGISTRY


H_FIX, W_FIX  = 2304, 768
TILE_SIZE     = 384
STRIDE        = 192
DOWNSAMPLE    = 4               # MedVAE 4x model
LATENT_H, LATENT_W = H_FIX // DOWNSAMPLE, W_FIX // DOWNSAMPLE     # 576, 192
LATENT_TILE   = TILE_SIZE // DOWNSAMPLE                           # 96
LATENT_STRIDE = STRIDE // DOWNSAMPLE                              # 48


@torch.no_grad()
def encode_image_tiled(
    model,
    image: torch.Tensor,             # (1, H_FIX, W_FIX) in [0, 1]
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Identical to encode_image_tiled() in extract_latents_tiled.py.

    Slides 384x384 tiles, per-tile min-max normalises, encodes, then blends
    latents with a cosine window. Returns (Cz, 576, 192) on CPU as float32.
    """
    _, H, W = image.shape
    assert (H, W) == (H_FIX, W_FIX), \
        f"encode_image_tiled expects {(H_FIX, W_FIX)}, got {(H, W)}"

    output  = None
    weights = None
    window  = make_cosine_window(LATENT_TILE, device)

    ys = tile_positions(H, TILE_SIZE, STRIDE)
    xs = tile_positions(W, TILE_SIZE, STRIDE)

    for y in ys:
        for x in xs:
            tile = image[:, y:y + TILE_SIZE, x:x + TILE_SIZE]

            tmin, tmax = tile.min(), tile.max()
            if tmax > tmin:
                tile_n = (tile - tmin) / (tmax - tmin)
            else:
                tile_n = torch.zeros_like(tile)
            tile_n = (tile_n - 0.5) / 0.5      # [0, 1] -> [-1, 1]

            inp = tile_n.unsqueeze(0).to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype):
                posterior = model.encode(inp)
                z = posterior.mode().float()
            z = z.squeeze(0).to(device)        # (Cz, 96, 96)

            if output is None:
                Cz = z.shape[0]
                output  = torch.zeros((Cz, LATENT_H, LATENT_W), device=device)
                weights = torch.zeros((1,  LATENT_H, LATENT_W), device=device)

            ly, lx = y // DOWNSAMPLE, x // DOWNSAMPLE
            output [:, ly:ly + LATENT_TILE, lx:lx + LATENT_TILE] += z * window
            weights[:, ly:ly + LATENT_TILE, lx:lx + LATENT_TILE] += window

    return (output / weights.clamp(min=1e-6)).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True, type=Path)
    p.add_argument("--img-dir",    required=True, type=Path)
    p.add_argument("--model-name", default="medvae_4_1_2d",
                    choices=list(MODEL_REGISTRY),
                    help="Which pretrained MedVAE 2D variant. Default "
                         "matches matched_tiled (patch-finetuned 4x/1ch).")
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    spec = MODEL_REGISTRY[args.model_name]
    if spec["downsample"] != DOWNSAMPLE:
        raise ValueError(
            f"Model {args.model_name} has spatial downsample "
            f"{spec['downsample']}, but this tile-and-stitch geometry "
            f"assumes {DOWNSAMPLE}. Use a 4x variant."
        )

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained MedVAE variant: {args.model_name}")
    model = load_medvae_variant(args.model_name, device, dtype)

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            print(f"No {split} rows, skipping.")
            continue

        # Discover Cz from the first image
        first_path = args.img_dir / f"{sub.iloc[0]['image_uuid']}.jpg"
        pil = Image.open(first_path).convert("L").resize(
            (W_FIX, H_FIX), Image.BICUBIC)
        first = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
        first_z = encode_image_tiled(model, first.to(device), device, dtype)
        Cz = first_z.shape[0]
        print(f"  {split}: discovered latent shape "
              f"Cz={Cz} H={LATENT_H} W={LATENT_W}")

        latents = np.empty((len(sub), Cz, LATENT_H, LATENT_W), dtype=np.float32)
        uuids = []
        latents[0] = first_z.numpy()
        uuids.append(sub.iloc[0]["image_uuid"])

        for i, row in tqdm(sub.iloc[1:].iterrows(), total=len(sub) - 1,
                           desc=f"  {split}"):
            img_path = args.img_dir / f"{row['image_uuid']}.jpg"
            pil = Image.open(img_path).convert("L").resize(
                (W_FIX, H_FIX), Image.BICUBIC)
            arr = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255.0
            z = encode_image_tiled(model, arr.to(device), device, dtype)
            latents[i] = z.numpy()
            uuids.append(row["image_uuid"])

        out_path = args.out_dir / f"latents_{split}.npz"
        np.savez_compressed(out_path,
                            latents=latents,
                            uuids=np.array(uuids))
        print(f"  Wrote {out_path}  shape={latents.shape}  "
              f"({latents.nbytes / 1e9:.2f} GB uncompressed)")


if __name__ == "__main__":
    main()
