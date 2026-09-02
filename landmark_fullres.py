"""
landmark_fullres.py — the FULL-RESOLUTION detector control.

Question: the latent detector localises at ~68 px (vertebra-scale). Is that
coarseness a fact about the COMPRESSION, or just a weak detector?

Control: train a landmark detector on the UNCOMPRESSED image — the same
aspect-padded input MedVAE's encoder itself receives — and compare. Because the
input is the encoder's own input, the only difference is compression.

  * detector sharpens markedly on full-res  -> the ceiling is the LATENT.
    This underwrites the whole "coarse because compressed" mechanism claim.
  * detector still ~68 px on full-res       -> our detector is simply weak, and
    the mechanism story needs revisiting.

Fair-effort caution: full-res detection is a bigger modelling problem than a
tiny latent head, so we give it a real backbone (ResNet18, optionally
ImageNet-pretrained) and optional augmentation. Crippling the full-res arm
would rig the control toward the answer we want.

Predictions are saved in the shared format (uuids, pred_landmarks, gt_landmarks;
(N,136) ordering A normalised [0,1]) so landmark_pixel_audit.py audits it
identically (pixels, % of spine extent, per-region, overlays).

Run:
  python landmark_fullres.py \
      --csv ./medvae/data/boostnet_with_angles.csv \
      --img-dir ./medvae/data/boostnet \
      --landmarks-dir ~/MedVAE/data/boostnet_labeldata/labels \
      --train-lookup ./landmark_row_lookup_train_fixed.csv \
      --test-lookup ./landmark_row_lookup_test_fixed.csv \
      --out-dir ./outputs/fullres --seeds 0 1 2

  python landmark_fullres.py --smoke
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

N_LM = 68
N_VERT = 17
SPLIT_TO_LMK_SUBDIR = {"train": "training", "val": "training", "test": "test"}
# reference: the latent detector's normalised error (4x / 8x)
LATENT_REF = {"4x": 0.0420, "8x": 0.0418}


# ==========================================================================
# data
# ==========================================================================
def aspect_pad_square(img, size):
    """PIL grayscale -> (1,size,size) float tensor in [-1,1], aspect-padded.
    Mirrors the encoder's aspect-padded square preprocessing so the control
    sees the same framing (and the same [0,1] landmark convention holds)."""
    import PIL.Image as Image
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, (0, 0))                      # top-left paste (content bbox at origin)
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    return (x * 2 - 1).unsqueeze(0), (nw, nh)


def landmarks_to_canvas(lm136, nw, nh, size):
    """Landmarks are normalised to the ORIGINAL image. After aspect-pad into a
    top-left (nw,nh) region of a size×size canvas, re-normalise to the canvas."""
    x = lm136[:N_LM] * nw / size
    y = lm136[N_LM:] * nh / size
    return np.concatenate([x, y]).astype(np.float32)


def load_split_images(df, split, img_dir, uuid_to_row, lmk_all, size):
    """Targets stay in ORIGINAL-IMAGE normalised coords — the same convention as
    the latent detector (train_frontier.py) — so the two arms are directly
    comparable and landmark_pixel_audit.py can convert both to pixels. The model
    infers the aspect from the visible padding, exactly as the latent model does."""
    xs, ys, uu = [], [], []
    import PIL.Image as Image
    sub = df[df["split"] == split].reset_index(drop=True)
    for _, r in sub.iterrows():
        u = str(r["image_uuid"])
        if u not in uuid_to_row:
            continue
        p = Path(img_dir) / f"{u}.jpg"
        if not p.exists():
            continue
        with Image.open(p) as im:
            im = im.convert("L")
            t, (nw, nh) = aspect_pad_square(im, size)
        xs.append(t)
        ys.append(lmk_all[uuid_to_row[u]].astype(np.float32))   # original-image normalised
        uu.append(u)
    return torch.stack(xs), np.stack(ys), np.array(uu)


def load_split_latents(latents_dir, config, split):
    """Matched-capacity control: same backbone, but reading the LATENT."""
    d = np.load(Path(latents_dir) / f"latents_{config}_{split}.npz", allow_pickle=True)
    return (torch.as_tensor(d["latents"], dtype=torch.float32),
            d["landmarks"].astype(np.float32), d["uuids"])


def parse_landmark_row(vals):
    xs = np.asarray(vals[:N_LM], dtype=float)
    ys = np.asarray(vals[N_LM:], dtype=float)
    return np.concatenate([xs, ys])


def load_landmarks_for_split(split, landmarks_dir):
    sub = SPLIT_TO_LMK_SUBDIR[split]
    arr = pd.read_csv(Path(landmarks_dir) / sub / "landmarks.csv", header=None).values
    return np.stack([parse_landmark_row(r) for r in arr])


def build_uuid_to_row(split, lookups):
    lk = lookups.get("train" if SPLIT_TO_LMK_SUBDIR[split] == "training" else "test")
    if lk is None:
        raise SystemExit(f"need a lookup for split '{split}'")
    ucol = "image_uuid" if "image_uuid" in lk.columns else lk.columns[0]
    if "landmark_row" in lk.columns:
        return {str(u): int(r) for u, r in zip(lk[ucol], lk["landmark_row"])}
    return {str(u): i for i, u in enumerate(lk[ucol].astype(str))}


class ImgLM(Dataset):
    def __init__(self, x, y, aug=0.0):
        self.x, self.y, self.aug = x, torch.as_tensor(y), aug

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.aug:
            x = x * (1 + self.aug * (torch.rand(1) - 0.5)) + self.aug * (torch.rand(1) - 0.5)
            x = x + self.aug * 0.3 * torch.randn_like(x)
        return x, self.y[i]


# ==========================================================================
# model
# ==========================================================================
def build_model(backbone, pretrained, in_ch=1):
    if backbone == "resnet18":
        try:
            import torchvision
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            m = torchvision.models.resnet18(weights=w)
            m.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
            m.fc = nn.Linear(m.fc.in_features, 2 * N_LM)
            return m
        except Exception as e:
            print(f"  (torchvision unavailable: {e}; falling back to scratch CNN)")
    return ScratchCNN(in_ch=in_ch)


class ScratchCNN(nn.Module):
    def __init__(self, k=2 * N_LM, in_ch=1):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.GELU())
        self.body = nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128),
                                  blk(128, 256), blk(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 9, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, k))

    def forward(self, x):
        return self.head(self.pool(self.body(x)))


def norm_err(pred, gt):
    dx = pred[:, :N_LM] - gt[:, :N_LM]
    dy = pred[:, N_LM:] - gt[:, N_LM:]
    return np.sqrt(dx ** 2 + dy ** 2).mean(1)


def boot_ci(v, seed=0, nb=10000):
    rng = np.random.default_rng(seed); v = np.asarray(v); n = len(v)
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def train_one(tr, va, *, seed, epochs, bs, lr, wd, patience, device,
              backbone, pretrained, aug, in_ch=1):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(ImgLM(*tr, aug=aug), batch_size=bs, shuffle=True)
    xva = va[0].to(device); yva = va[1]
    model = build_model(backbone, pretrained, in_ch=in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.smooth_l1_loss(model(xb), yb, beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(xva[i:i + 32]) for i in range(0, len(xva), 32)]).cpu().numpy()
        m = norm_err(pv, yva).mean()
        if m < best - 1e-6:
            best, best_state, bad = m, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, best


def predict(model, x, device):
    with torch.no_grad():
        return torch.cat([model(x[i:i + 32].to(device)) for i in range(0, len(x), 32)]).cpu().numpy()


# ==========================================================================
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    if args.input == "latent":
        for split in ("train", "val", "test"):
            data[split] = load_split_latents(args.latents_dir, args.config, split)
            print(f"[{split}] {len(data[split][0])} latents "
                  f"{tuple(data[split][0].shape[1:])} ({args.config})")
        in_ch = data["train"][0].shape[1]
        src = f"LATENT {args.config}"
    else:
        df = pd.read_csv(args.csv)
        lookups = {}
        if args.train_lookup: lookups["train"] = pd.read_csv(args.train_lookup)
        if args.test_lookup: lookups["test"] = pd.read_csv(args.test_lookup)
        for split in ("train", "val", "test"):
            u2r = build_uuid_to_row(split, lookups)
            lmk = load_landmarks_for_split(split, args.landmarks_dir)
            data[split] = load_split_images(df, split, args.img_dir, u2r, lmk, args.size)
            print(f"[{split}] {len(data[split][0])} images at {args.size}x{args.size}")
        in_ch = 1
        src = f"FULL-RES image {args.size}px"

    tr = (data["train"][0], data["train"][1])
    va = (data["val"][0], data["val"][1])
    xte, yte, ute = data["test"]

    seed_preds, seed_train = [], []
    for s in args.seeds:
        model, val_best = train_one(tr, va, seed=s, epochs=args.epochs, bs=args.batch_size,
                                    lr=args.lr, wd=args.weight_decay, patience=args.patience,
                                    device=device, backbone=args.backbone,
                                    pretrained=not args.no_pretrained, aug=args.aug,
                                    in_ch=in_ch)
        seed_preds.append(predict(model, xte, device))
        seed_train.append(predict(model, tr[0], device))
        print(f"  seed {s}: best val {val_best:.4f}")
    ens = np.mean(seed_preds, axis=0)
    errs = norm_err(ens, yte)
    train_err = norm_err(np.mean(seed_train, axis=0), tr[1]).mean()
    m, lo, hi = boot_ci(errs)
    gap = "OVERFIT" if m > 1.5 * train_err else "ok"
    print(f"\n{src} detector ({args.backbone}"
          f"{'' if args.no_pretrained else ', pretrained'}):")
    print(f"  test  norm error = {m:.4f}  CI[{lo:.4f},{hi:.4f}]   [{gap}]")
    print(f"  train norm error = {train_err:.4f}")
    print(f"  small-head-on-latent reference: 4x {LATENT_REF['4x']:.4f} | 8x {LATENT_REF['8x']:.4f}")
    print("\n  NOTE: units are ORIGINAL-IMAGE normalised, matching the latent arm.")
    if args.input == "image":
        print("  To separate 'compression is the ceiling' from 'our head was small',\n"
              "  run the SAME backbone on the latent:  --input latent "
              "--latents-dir ./latents_aspect --config medvae_4_1_2d")

    tag = "fullres" if args.input == "image" else f"latent_{args.config}"
    np.savez_compressed(args.out_dir / f"preds_{tag}.npz",
                        pred_landmarks=ens.astype(np.float32),
                        gt_landmarks=yte.astype(np.float32), uuids=ute)
    print(f"\nwrote {args.out_dir}/preds_{tag}.npz "
          f"(audit it with landmark_pixel_audit.py for px / % of spine)")


def smoke():
    torch.manual_seed(0); np.random.seed(0)
    device = "cpu"
    # synthetic: a bright blob whose position encodes the landmarks
    def synth(n, size=64):
        xs, ys = [], []
        for _ in range(n):
            img = torch.full((1, size, size), -1.0)
            cx, cy = np.random.randint(12, size - 12, 2)
            img[0, cy - 4:cy + 4, cx - 4:cx + 4] = 1.0
            xs.append(img)
            lm = np.concatenate([np.full(N_LM, cx / size), np.full(N_LM, cy / size)])
            ys.append(lm.astype(np.float32))
        return torch.stack(xs), np.stack(ys)
    tr, va, te = synth(24), synth(8), synth(8)
    model, best = train_one(tr, va, seed=0, epochs=3, bs=8, lr=1e-3, wd=1e-4,
                            patience=5, device=device, backbone="scratch",
                            pretrained=False, aug=0.0, in_ch=1)
    pv = predict(model, te[0], device)
    e = norm_err(pv, te[1]).mean()
    # padding helper still used for the image arm
    import PIL.Image as Image
    im = Image.new("L", (200, 400), 128)
    t, (nw, nh) = aspect_pad_square(im, 96)
    # multi-channel latent path (e.g. 4_3 / 8_4 configs)
    m3 = build_model("scratch", False, in_ch=3)
    y3 = m3(torch.randn(2, 3, 96, 96))
    ok = (t.shape == (1, 96, 96) and (nw, nh) == (48, 96)
          and y3.shape == (2, 136) and np.isfinite(e))
    print(f"pad: {tuple(t.shape)} content {(nw,nh)}  |  3ch latent -> {tuple(y3.shape)}")
    print(f"pred {pv.shape}, test err {e:.4f}")
    print("SMOKE:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--landmarks-dir", type=Path)
    ap.add_argument("--train-lookup", type=Path)
    ap.add_argument("--test-lookup", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/fullres"))
    ap.add_argument("--input", choices=["image", "latent"], default="image",
                    help="'image' = full-res control; 'latent' = matched-capacity control")
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--config", default="medvae_4_1_2d")
    ap.add_argument("--size", type=int, default=384,
                    help="input square size; 384 = the encoder's own input (cleanest control)")
    ap.add_argument("--backbone", choices=["resnet18", "scratch"], default="resnet18")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--aug", type=float, default=0.0, help="intensity aug strength, e.g. 0.1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.input == "latent":
        if not args.latents_dir:
            raise SystemExit("--latents-dir required for --input latent")
        run(args)
    elif args.csv and args.img_dir and args.landmarks_dir:
        run(args)
    else:
        raise SystemExit("need --csv --img-dir --landmarks-dir (or --smoke)")


if __name__ == "__main__":
    main()
