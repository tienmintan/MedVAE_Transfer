"""
recon_check.py: read-only inventory of reconstruction tooling, compression configurations and landmark labels.

Writes nothing and trains nothing; it reports what exists on the workstation,
as input to the frequency/structure-loss analysis (freq_loss_*.py).

It checks:
  1. Reconstruction capability: can MedVAE both encode and decode?
  2. Compression configurations: which model variants and latent files exist?
  3. Landmark labels: location, format and coordinate convention.

Inputs:  files under ROOT (~/MedVAE/MedVAE) and the HuggingFace cache
         (paths set in the module).
Outputs: none; the report is printed to stdout.

Usage:  python recon_check.py
"""

import os
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "MedVAE" / "MedVAE"
DATA = ROOT / "medvae" / "data"


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# 1. Reconstruction capability
hr("1. Reconstruction capability (encode -> decode -> image)")

# Look for any script that calls a decoder
recon_keywords = ["decode", "reconstruct", "recon", "evaluate_reconstruction"]
py_files = glob.glob(str(ROOT / "*.py"))
hits = []
for f in py_files:
    base = os.path.basename(f).lower()
    try:
        txt = open(f, "r", errors="ignore").read().lower()
    except Exception:
        continue
    if any(k in base for k in recon_keywords) or ".decode(" in txt or "decode_samples" in txt:
        # record which signals fired
        sig = []
        if any(k in base for k in recon_keywords):
            sig.append("filename")
        if ".decode(" in txt:
            sig.append(".decode() call")
        if "decode_samples" in txt:
            sig.append("decode_samples")
        hits.append((os.path.basename(f), sig))

if hits:
    print("Candidate reconstruction/decode scripts found:")
    for name, sig in sorted(hits):
        print(f"  {name:45s}  [{', '.join(sig)}]")
else:
    print("No obvious decode/reconstruction script found.")

# Does the MedVAE model object expose a decode method?
print("\nProbing the MedVAE model API for a decode path:")
try:
    from medvae import MVAE
    m = MVAE(model_name="medvae_4_1_2d", modality="xray")
    inner = m.model
    has = {meth: hasattr(inner, meth) for meth in
           ["encode", "decode", "forward", "reconstruct"]}
    print(f"  MVAE.model methods present: {has}")
    # AutoencoderKL typically has .decode(z) -> image
    if hasattr(inner, "decode"):
        print("  -> .decode(z) is available: full reconstruction is possible.")
    else:
        print("  -> No .decode found on this object; the decoder has to be located elsewhere.")
except Exception as e:
    print(f"  Could not import/instantiate MVAE here ({e}).")
    print("  (Not fatal; the probe needs to run inside the training environment.)")


# 2. Compression configurations available
hr("2. Compression configurations (weights + extracted latents)")

# 2a. Which HF MedVAE variants are cached locally?
print("HuggingFace MedVAE weight files in cache:")
hf_cache = Path.home() / ".cache" / "huggingface"
ckpts = list(hf_cache.rglob("*vae_*c_2D.ckpt")) + list(hf_cache.rglob("*MedVAE*/**/*.ckpt"))
seen = set()
for c in ckpts:
    if c.name in seen:
        continue
    seen.add(c.name)
    print(f"  {c.name}")
if not seen:
    print("  (none found in HF cache; they download on first use)")

# 2b. Which latent .npz files exist, grouped by apparent config
print("\nExtracted latent files (.npz) and their shapes:")
npz_files = sorted(glob.glob(str(ROOT / "**" / "*.npz"), recursive=True))
interesting = [f for f in npz_files if any(
    k in f.lower() for k in ["latent", "aspect", "tiled", "whole", "pretrained_4x",
                              "pretrained_8x", "4_1", "4_3", "8_1", "8_4"])]
for f in interesting:
    rel = os.path.relpath(f, ROOT)
    try:
        with np.load(f, allow_pickle=True) as d:
            if "latents" in d:
                shp = d["latents"].shape
                print(f"  {rel:55s}  latents {shp}")
            else:
                print(f"  {rel:55s}  keys={list(d.keys())}")
    except Exception as e:
        print(f"  {rel:55s}  [unreadable: {e}]")

print("\nChannel-variant check (are 3ch / 4ch latents present?):")
for tag in ["4_3", "8_4", "3ch", "4ch", "4_3_2d", "8_4_2d"]:
    matches = [f for f in npz_files if tag in os.path.basename(f).lower()]
    if matches:
        print(f"  '{tag}': {len(matches)} file(s) -> {[os.path.basename(m) for m in matches[:3]]}")
print("  (If nothing is listed above, only spatial configurations are present: 4x/1ch + 8x/1ch.)")


# 3. Landmark labels
hr("3. Landmark labels (format + coordinate convention)")

# Search common locations and extensions
landmark_candidates = []
for pat in ["*landmark*", "*.mat", "*coord*", "*point*"]:
    landmark_candidates += glob.glob(str(DATA / "**" / pat), recursive=True)
landmark_candidates = sorted(set(landmark_candidates))

if landmark_candidates:
    print("Candidate landmark files:")
    for f in landmark_candidates[:30]:
        print(f"  {os.path.relpath(f, ROOT)}  ({os.path.getsize(f)} bytes)")
else:
    print("No landmark/.mat/coord files found under medvae/data/.")
    print("AASCE/BoostNet landmarks are usually .mat files; they may be")
    print("in a separate download dir. Searching home dir broadly:")
    broad = glob.glob(str(Path.home() / "**" / "*landmark*"), recursive=True)[:20]
    broad += glob.glob(str(Path.home() / "**" / "*.mat"), recursive=True)[:20]
    for f in sorted(set(broad))[:20]:
        print(f"  {f}")

# Inspect the structure of the first .mat or landmark CSV found
def inspect_mat(path):
    try:
        import scipy.io as sio
        m = sio.loadmat(path)
        print(f"\n  .mat structure of {os.path.basename(path)}:")
        for k, v in m.items():
            if k.startswith("__"):
                continue
            if hasattr(v, "shape"):
                print(f"    {k}: shape {v.shape}, dtype {v.dtype}")
                # show value range to infer normalised vs pixel coords
                try:
                    arr = np.asarray(v, dtype=float)
                    print(f"        min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f} "
                          f"-> {'likely normalised [0,1]' if np.nanmax(arr) <= 1.5 else 'likely pixel coords'}")
                except Exception:
                    pass
            else:
                print(f"    {k}: {type(v)}")
    except ImportError:
        print("  (scipy not available to inspect .mat; pip install scipy)")
    except Exception as e:
        print(f"  Could not read {path}: {e}")

mats = [f for f in landmark_candidates if f.endswith(".mat")]
if mats:
    inspect_mat(mats[0])

csvs = [f for f in landmark_candidates if f.endswith(".csv")]
for c in csvs[:1]:
    print(f"\n  CSV head of {os.path.basename(c)}:")
    try:
        for i, line in enumerate(open(c)):
            if i > 2:
                break
            print(f"    {line.rstrip()[:160]}")
    except Exception as e:
        print(f"    [unreadable: {e}]")

# Also check the known bbox csv for coordinate convention reference
bbox = DATA / "boostnet.csv"
if bbox.exists():
    print(f"\n  Reference: boostnet.csv head (known bbox file):")
    for i, line in enumerate(open(bbox)):
        if i > 2:
            break
        print(f"    {line.rstrip()[:160]}")

hr("Done")
print("Checks covered: (1) reconstruction capability, (2) which configurations span")
print("the compression range, (3) the landmark format.")
