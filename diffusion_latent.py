"""
diffusion_latent.py: landmark-conditioned latent diffusion for spine radiographs.

Generates new spine anatomy with exact labels, to address the data limit that
augmentation only partly addresses (geometric augmentation is clinically
implausible; intensity-only augmentation gives a smaller gain).

Design, and the reason for each choice:

  * Generate in MedVAE latent space (1x96x96), not pixel space. 9x smaller than
    a 288x288 image, trainable on this dataset, and the same representation
    that the other analyses characterise.
    Caveat: MedVAE discards fine endplate detail, so generated images inherit
    that loss. Samples cannot contain detail the encoder cannot represent.
    This is a limitation of the approach, not a bug.

  * Condition on landmark heatmaps. The model is told where the 68 corners are,
    so the sampled landmarks are the label: no annotation, no label noise, and
    no circularity from labelling synthetic data with a trained detector.

  * Sample shapes from a PCA statistical shape model fitted to the real
    training landmarks. Sampling within +/-2 SD of the observed distribution
    gives new curve magnitudes and apex positions that stay anatomically
    plausible: new spines, not new textures on old spines.

  * Train on the augmented latents (2454, not 409). The generator needs data too.

Validation before use (`--mode validate`), in order of severity:
  1. Memorisation: nearest-neighbour distance from each sample to the training
     set, against the train-train baseline. If samples sit much closer to
     training points than training points sit to each other, the model is
     copying and the "new data" is illusory.
  2. Condition fidelity: run the real-data landmark detector on generated
     samples. If predicted landmarks match the conditioning landmarks, the
     conditioning works and the labels are reliable.
  3. Distribution: per-channel latent statistics vs real latents.

Validation criteria: the generated data is not usable if samples do not decode
to spine-like images, nearest-neighbour analysis shows memorisation, or
condition fidelity is no better than a mean-shape predictor.

Inputs:  --latents-dir (latents_{config}_{split}.npz with latents, landmarks
         and uuids) and --img-dir (images, for the aspect ratio used to map
         landmarks to canvas coordinates).
Outputs: in --out-dir, by mode: diffusion.pt (train), generated.npz (sample),
         overfit.npz (overfit), conditioning_scaling.csv/.png (scaling),
         alignment_check.png (align), synthetic_utility.csv (utility),
         label_quality.png (showlabels), generated_samples.png (decode);
         the other modes print to stdout.

Usage:
  python diffusion_latent.py --mode train --latents-dir ./latents_aug \
      --config medvae_4_1_2d --out-dir ./outputs/diffusion --epochs 400
  python diffusion_latent.py --mode sample --out-dir ./outputs/diffusion --n 500
  python diffusion_latent.py --smoke      # synthetic-data self-test
"""
import argparse
import math
from pathlib import Path

import numpy as np

N_LM = 68
N_VERT = 17


# --- Conditioning: landmarks -> heatmap channels ---
def to_canvas_centered(lm136, aspect):
    """Map original-image landmarks to a centre-padded square canvas.

    Empirically (see --mode align) MedVAE's loader pads symmetrically: the
    content is centred with equal bars on both sides, not pasted top-left.
    So the centre is preserved and only the scale changes:

        x_canvas = 0.5 + (x_orig - 0.5) * (W/H)      for tall images
        y_canvas = 0.5 + (y_orig - 0.5) * (H/W)      for wide images

    The raw convention gets the centre right but stretches the spine across the
    full canvas width; the top-left convention gets the scale right but pushes
    everything into the padding. Neither matches the loader."""
    out = lm136.copy()
    if aspect <= 1.0:                       # tall: width is the short side
        out[:N_LM] = 0.5 + (lm136[:N_LM] - 0.5) * aspect
    else:                                   # wide: height is the short side
        out[N_LM:] = 0.5 + (lm136[N_LM:] - 0.5) / aspect
    return out


def to_canvas(lm136, aspect):
    """Map landmarks from original-image normalisation to aspect-padded canvas
    normalisation (top-left convention).

    This convention assumes the pipeline scales each image so max(W,H) fills a
    square canvas and pastes it top-left. Landmarks are stored normalised to
    the original image, so for a tall (1:2) radiograph the spine occupies only
    the left ~half of the canvas, while an unmapped condition map draws it
    across the full width. Without this mapping the conditioning is ~2x too
    wide and misaligned with the latent, and the model cannot follow it even
    in principle.

    MedVAE's loader centres the content instead (see to_canvas_centered and
    --mode align), so this top-left mapping is used only as one of the
    candidates that --mode align compares; training and sampling use
    to_canvas_centered.

    aspect = W/H. For W < H (the usual case): x_canvas = x_orig * (W/H),
    y_canvas = y_orig. For W > H the roles swap."""
    out = lm136.copy()
    if aspect <= 1.0:                       # tall: width is the short side
        out[:N_LM] = lm136[:N_LM] * aspect
    else:                                   # wide: height is the short side
        out[N_LM:] = lm136[N_LM:] / aspect
    return out


def render_condition(lm136, size=96, sigma=3.0, n_ch=5):
    """(136,) normalised landmarks -> (n_ch, size, size) condition maps.

    Channels 0-3: Gaussian heatmaps split by corner index (TL/TR/BL/BR), so the
    model can place a correctly oriented vertebra rather than a blob.
    Channel 4:    filled vertebra bodies (each quad rasterised solid).

    The dense body channel matters. With only sparse corner Gaussians most of
    the condition map is zero, giving the network very little to latch onto,
    which can lead a model to ignore its conditioning entirely."""
    xs, ys = lm136[:N_LM], lm136[N_LM:]
    out = np.zeros((n_ch, size, size), dtype=np.float32)
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32)
    for k in range(N_LM):
        cx, cy = xs[k] * (size - 1), ys[k] * (size - 1)
        out[k % 4] += np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))

    if n_ch > 4:                                    # dense vertebra-body channel
        P = np.stack([xs * (size - 1), ys * (size - 1)], 1).reshape(N_VERT, 4, 2)
        for v in range(N_VERT):
            quad = P[v][[0, 1, 3, 2]]               # TL,TR,BR,BL -> convex order
            inside = np.ones((size, size), dtype=bool)
            for e in range(4):
                a, b = quad[e], quad[(e + 1) % 4]
                cross = (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0])
                inside &= (cross >= 0)
            if not inside.any():                    # reversed winding
                inside = np.ones((size, size), dtype=bool)
                for e in range(4):
                    a, b = quad[e], quad[(e + 1) % 4]
                    cross = (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0])
                    inside &= (cross <= 0)
            out[4][inside] = 1.0
    return np.clip(out, 0, 1)


def load_aspects(uuids, img_dir):
    """W/H per image. Augmented copies (`uuid__augN`) share their source aspect,
    since augmentation preserves the frame."""
    from PIL import Image
    if img_dir is None:
        raise SystemExit("--img-dir is required: landmarks must be mapped to "
                         "aspect-padded canvas coordinates before rendering the "
                         "condition, otherwise it is misaligned with the latent.")
    cache, out = {}, []
    for u in uuids:
        base = str(u).split("__aug")[0]
        if base not in cache:
            hit = None
            for ext in (".jpg", ".jpeg", ".png"):
                q = Path(img_dir) / f"{base}{ext}"
                if q.exists():
                    hit = q; break
            if hit is None:
                cache[base] = 1.0
            else:
                with Image.open(hit) as im:
                    cache[base] = im.width / im.height
        out.append(cache[base])
    return np.array(out, dtype=np.float32)


# --- Statistical shape model, for sampling new plausible anatomy ---
class ShapeModel:
    """PCA over the 68 landmark coordinates of real training spines.

    Sampling within the observed distribution yields new curve magnitudes and
    apex positions that remain anatomically plausible, adding anatomy the
    dataset does not contain without leaving the manifold of real scoliotic
    spines."""

    def __init__(self, lm, n_comp=20):
        self.mu = lm.mean(0)
        X = lm - self.mu
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(n_comp, len(S))
        self.comps = Vt[:k]                       # (k,136)
        self.sd = S[:k] / np.sqrt(max(len(X) - 1, 1))
        self.evr = float((S[:k] ** 2).sum() / (S ** 2).sum())

    def sample(self, rng, n, clip=2.0):
        z = np.clip(rng.standard_normal((n, len(self.sd))), -clip, clip)
        return np.clip(self.mu + (z * self.sd) @ self.comps, 0.0, 1.0).astype(np.float32)

    def project(self, lm):
        return self.mu + ((lm - self.mu) @ self.comps.T) @ self.comps


# --- Diffusion schedule ---
def cosine_betas(T, s=0.008):
    """Nichol & Dhariwal cosine schedule: better than linear for small images,
    as it spends less of the trajectory in pure noise."""
    t = np.linspace(0, T, T + 1, dtype=np.float64) / T
    a = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = a / a[0]
    betas = np.clip(1 - a[1:] / a[:-1], 1e-8, 0.999)
    return betas.astype(np.float32)


class Schedule:
    def __init__(self, T=1000):
        self.T = T
        self.betas = cosine_betas(T)
        self.alphas = 1.0 - self.betas
        self.abar = np.cumprod(self.alphas)
        self.sqrt_abar = np.sqrt(self.abar)
        self.sqrt_1mabar = np.sqrt(1.0 - self.abar)


# --- Denoising U-Net ---
def build_unet(in_ch, cond_ch, base=64, cross_attn=False, n_tokens=64):
    """cross_attn=False -> conditioning is concatenated at every resolution.
    cross_attn=True     -> the condition is encoded to a set of tokens and
                           injected by cross-attention at each resolution.

    Why cross-attention might help: concatenation gives every spatial position
    the condition value at that position, so the network must learn long-range
    correspondence through convolution alone. Cross-attention lets any position
    query any part of the condition, which is the standard mechanism in
    text-to-image diffusion and is designed for this kind of spatially
    non-local conditioning.

    Why it might not: the diagnosed limit is generalising conditioning from ~400
    spines. A higher-capacity mechanism has more parameters to overfit with.
    """
    import torch
    import torch.nn as nn

    class TimeEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                     nn.Linear(dim * 4, dim * 4))

        def forward(self, t):
            half = self.dim // 2
            f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
            e = t[:, None].float() * f[None]
            return self.mlp(torch.cat([e.sin(), e.cos()], -1))

    class CondEncoder(nn.Module):
        """Condition maps -> (B, n_tokens, dim) tokens for cross-attention."""
        def __init__(self, cin, dim, n_tok):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(cin, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                nn.Conv2d(64, dim, 3, stride=2, padding=1))
            self.n_tok = n_tok
            self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))

        def forward(self, c):
            h = self.body(c)                       # (B, dim, h, w)
            B, D, H, W = h.shape
            side = int(round(self.n_tok ** 0.5))
            h = torch.nn.functional.adaptive_avg_pool2d(h, (side, side))
            return h.flatten(2).transpose(1, 2) + self.pos[:, :side * side]

    class CrossAttn(nn.Module):
        """Standard multi-head cross-attention from image positions to
        condition tokens."""
        def __init__(self, ch, dim, heads=4):
            super().__init__()
            self.h = heads
            self.norm = nn.GroupNorm(8, ch)
            self.q = nn.Conv2d(ch, ch, 1)
            self.k = nn.Linear(dim, ch)
            self.v = nn.Linear(dim, ch)
            self.o = nn.Conv2d(ch, ch, 1)
            nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)  # start as identity

        def forward(self, x, tok):
            B, C, H, W = x.shape
            q = self.q(self.norm(x)).reshape(B, self.h, C // self.h, H * W).transpose(2, 3)
            k = self.k(tok).reshape(B, -1, self.h, C // self.h).transpose(1, 2)
            v = self.v(tok).reshape(B, -1, self.h, C // self.h).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(2, 3) / (C // self.h) ** 0.5, -1)
            out = (a @ v).transpose(2, 3).reshape(B, C, H, W)
            return x + self.o(out)

    class Block(nn.Module):
        def __init__(self, i, o, tdim):
            super().__init__()
            self.c1 = nn.Conv2d(i, o, 3, padding=1)
            self.c2 = nn.Conv2d(o, o, 3, padding=1)
            self.n1, self.n2 = nn.GroupNorm(8, o), nn.GroupNorm(8, o)
            self.temb = nn.Linear(tdim, o)
            self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

        def forward(self, x, t):
            h = torch.nn.functional.silu(self.n1(self.c1(x)))
            h = h + self.temb(t)[:, :, None, None]
            h = torch.nn.functional.silu(self.n2(self.c2(h)))
            return h + self.skip(x)

    class UNet(nn.Module):
        """Deliberately small (~2M params). With ~2.5k training latents a large
        model memorises rather than generalises."""
        def __init__(self):
            super().__init__()
            td = base * 4
            self.cross = cross_attn
            self.temb = TimeEmb(base)
            if cross_attn:
                cdim = base * 2
                self.cenc = CondEncoder(cond_ch, cdim, n_tokens)
                self.x1 = CrossAttn(base, cdim)
                self.x2 = CrossAttn(base * 2, cdim)
                self.x3 = CrossAttn(base * 2, cdim)
            # conditioning is injected at every resolution, not only at the
            # input: after two downsamples an input-only condition is heavily
            # diluted, which lets the model ignore it.
            self.inp = nn.Conv2d(in_ch + cond_ch, base, 3, padding=1)
            self.d1 = Block(base + cond_ch, base, td)
            self.d2 = Block(base + cond_ch, base * 2, td)
            self.d3 = Block(base * 2 + cond_ch, base * 2, td)
            self.mid = Block(base * 2, base * 2, td)
            self.u3 = Block(base * 4, base * 2, td)
            self.u2 = Block(base * 4, base, td)
            self.u1 = Block(base * 2, base, td)
            self.out = nn.Conv2d(base, in_ch, 3, padding=1)
            self.pool = nn.AvgPool2d(2)

        def forward(self, x, t, cond):
            te = self.temb(t)
            c1 = cond
            c2 = self.pool(c1)
            c3 = self.pool(c2)
            h = self.inp(torch.cat([x, c1], 1))
            h1 = self.d1(torch.cat([h, c1], 1), te)
            h2 = self.d2(torch.cat([self.pool(h1), c2], 1), te)
            h3 = self.d3(torch.cat([self.pool(h2), c3], 1), te)
            if self.cross:
                tok = self.cenc(cond)
                h1 = self.x1(h1, tok)
                h2 = self.x2(h2, tok)
                h3 = self.x3(h3, tok)
            m = self.mid(h3, te)
            u = self.u3(torch.cat([m, h3], 1), te)
            u = torch.nn.functional.interpolate(u, scale_factor=2, mode="nearest")
            u = self.u2(torch.cat([u, h2], 1), te)
            u = torch.nn.functional.interpolate(u, scale_factor=2, mode="nearest")
            u = self.u1(torch.cat([u, h1], 1), te)
            return self.out(u)

    return UNet()


# --- Mode: train ---
def train(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    d = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                allow_pickle=True)
    lat, lm = d["latents"].astype(np.float32), d["landmarks"].astype(np.float32)
    print(f"training latents {lat.shape}, landmarks {lm.shape}")

    # standardise: MedVAE latents are not unit-normal, and diffusion assumes
    # roughly standard-normal data at t=T
    mu, sd = float(lat.mean()), float(lat.std())
    print(f"latent stats: mean {mu:.4f}, sd {sd:.4f} -> standardising")
    lat = (lat - mu) / sd

    size = lat.shape[-1]
    aspects = load_aspects(d["uuids"], args.img_dir)
    print(f"aspect ratios W/H: median {np.median(aspects):.3f}, "
          f"range [{aspects.min():.3f},{aspects.max():.3f}]")
    # The centred convention is the one verified empirically (--mode align):
    # MedVAE's loader pads symmetrically, so the centre is preserved and only
    # the scale changes. --mode align reports the mean intensity at the
    # landmark positions under each convention.
    lm_canvas = np.stack([to_canvas_centered(l, a) for l, a in zip(lm, aspects)])
    cond = np.stack([render_condition(l, size=size, sigma=args.sigma,
                                      n_ch=args.cond_ch) for l in lm_canvas])
    print(f"condition maps {cond.shape} (landmarks mapped to canvas coords)")

    shape = ShapeModel(lm, args.n_comp)
    print(f"shape model: {args.n_comp} components, "
          f"{100*shape.evr:.1f}% of landmark variance")

    sch = Schedule(args.T)
    net = build_unet(lat.shape[1], args.cond_ch, args.base,
                     cross_attn=args.cross_attn, n_tokens=args.n_tokens).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"UNet {n_par/1e6:.2f}M params\n")

    X = torch.tensor(lat, device=device)
    C = torch.tensor(cond, device=device)
    sqa = torch.tensor(sch.sqrt_abar, device=device)
    sq1 = torch.tensor(sch.sqrt_1mabar, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = len(X)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x0, c = X[idx], C[idx]
            t = torch.randint(0, sch.T, (len(idx),), device=device)
            # classifier-free guidance: drop the condition on a fraction of
            # samples so the model also learns the unconditional score. At
            # sampling time the two are combined to amplify adherence to the
            # condition. Without this the model is free to ignore it.
            if args.p_uncond > 0:
                keep = (torch.rand(len(idx), 1, 1, 1, device=device) > args.p_uncond)
                c = c * keep
            noise = torch.randn_like(x0)
            xt = sqa[t][:, None, None, None] * x0 + sq1[t][:, None, None, None] * noise
            loss = torch.nn.functional.mse_loss(net(xt, t, c), noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if (ep + 1) % args.log_every == 0 or ep == 0:
            print(f"  epoch {ep+1:>4}/{args.epochs}  loss {tot/n:.4f}", flush=True)

    lo, hi = float(lat.min()), float(lat.max())
    print(f"standardised data range [{lo:.2f}, {hi:.2f}]; used to clamp x0 at sampling")
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "size": size,
                "x0_lo": lo, "x0_hi": hi, "p_uncond": args.p_uncond,
                "in_ch": lat.shape[1], "cond_ch": args.cond_ch, "base": args.base,
                "T": args.T, "sigma": args.sigma,
                "shape_mu": shape.mu, "shape_comps": shape.comps,
                "shape_sd": shape.sd, "config": args.config,
                "aspects": aspects.astype(np.float32),
                "canvas_convention": "centered",
                "cross_attn": bool(args.cross_attn), "n_tokens": args.n_tokens},
               out / "diffusion.pt")
    print(f"\nSaved {out}/diffusion.pt")
    print("\nNext:")
    print(f"  python diffusion_latent.py --mode sample --out-dir {out} --n 500")


# --- Mode: sample ---
def sample(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    ck = torch.load(out / "diffusion.pt", map_location=device, weights_only=False)
    net = build_unet(ck["in_ch"], ck["cond_ch"], ck["base"],
                     cross_attn=ck.get("cross_attn", False),
                     n_tokens=ck.get("n_tokens", 64)).to(device)
    net.load_state_dict(ck["state"]); net.eval()
    sch = Schedule(ck["T"])
    rng = np.random.default_rng(args.seed)

    sm = ShapeModel.__new__(ShapeModel)
    sm.mu, sm.comps, sm.sd = ck["shape_mu"], ck["shape_comps"], ck["shape_sd"]
    new_lm = sm.sample(rng, args.n, clip=args.clip)
    # sample an aspect ratio from the real distribution, then map the sampled
    # landmarks into canvas coordinates as training did
    tr_aspects = ck.get("aspects")
    if tr_aspects is None:
        raise SystemExit("checkpoint has no stored aspect ratios, so the condition cannot be "
                         "mapped to canvas coordinates; retraining with --img-dir is required")
    if ck.get("canvas_convention") != "centered":
        raise SystemExit(
            "this checkpoint was trained with a canvas convention other than 'centered' "
            "(the conditioning is misaligned with the latent). Retraining is required; "
            "--mode align shows the alignment check.")
    asp = rng.choice(np.asarray(tr_aspects), size=args.n)
    cond = np.stack([render_condition(to_canvas_centered(l, a), size=ck["size"],
                                      sigma=ck["sigma"], n_ch=ck["cond_ch"])
                     for l, a in zip(new_lm, asp)])
    print(f"  landmarks mapped to canvas coords (aspect median {np.median(asp):.3f})")
    print(f"Sampling {args.n} latents conditioned on new shapes "
          f"(+/-{args.clip} SD of the real distribution)")
    if args.guidance > 1.0:
        if ck.get("p_uncond", 0) <= 0:
            print("  WARNING: guidance was requested, but the model was trained without "
                  "condition dropout (p_uncond <= 0). Guided samples may be unreliable; "
                  "retraining with --p-uncond 0.15 may help.")
        print(f"  classifier-free guidance scale {args.guidance}")

    # DDIM: deterministic, far fewer steps than ancestral sampling.
    # x0 has to be clamped. At large t, abar ~ 1e-9, so x0 = (x - sqrt(1-abar)*eps)
    # / sqrt(abar) divides by ~5e-5 and amplifies any error in the predicted
    # noise by ~2e4. Without clamping the trajectory diverges and samples come
    # out orders of magnitude outside the data range.
    x0_lo = ck.get("x0_lo", -8.0) - args.clamp_margin
    x0_hi = ck.get("x0_hi", 8.0) + args.clamp_margin
    print(f"clamping x0 to [{x0_lo:.2f}, {x0_hi:.2f}] (standardised units)")
    steps = np.linspace(0, sch.T - 1, args.steps).round().astype(int)[::-1]
    abar = torch.tensor(sch.abar, device=device)
    outs = []
    with torch.no_grad():
        for i in range(0, args.n, args.batch_size):
            c = torch.tensor(cond[i:i + args.batch_size], device=device)
            x = torch.randn(len(c), ck["in_ch"], ck["size"], ck["size"], device=device)
            for j, t in enumerate(steps):
                tt = torch.full((len(c),), int(t), device=device, dtype=torch.long)
                if args.guidance > 1.0:
                    eps_c = net(x, tt, c)
                    eps_u = net(x, tt, torch.zeros_like(c))
                    eps = eps_u + args.guidance * (eps_c - eps_u)
                else:
                    eps = net(x, tt, c)
                a_t = abar[t]
                x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(x0_lo, x0_hi)
                if j + 1 < len(steps):
                    a_prev = abar[steps[j + 1]]
                    x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
                else:
                    x = x0
            outs.append(x.cpu().numpy())
            print(f"  {min(i+args.batch_size, args.n)}/{args.n}", flush=True)

    z = np.concatenate(outs, 0)
    print(f"\nstandardised samples: mean {z.mean():+.3f}  sd {z.std():.3f}  "
          f"range [{z.min():+.2f},{z.max():+.2f}]  (training data was "
          f"mean 0, sd 1, range [{ck.get('x0_lo', float('nan')):.2f},"
          f"{ck.get('x0_hi', float('nan')):.2f}])")
    if abs(z.std() - 1.0) > 0.5:
        print(f"  WARNING: standardised sample SD is {z.std():.2f} (expected about 1.0, "
              "tolerance 0.5). The sampler has likely diverged and these latents are "
              "probably invalid. Reducing --steps or --clamp-margin may help.")
    lat = z * ck["sd"] + ck["mu"]                            # de-standardise
    np.savez_compressed(out / "generated.npz",
                        latents=lat.astype(np.float32),
                        landmarks=new_lm.astype(np.float32),
                        aspects=asp.astype(np.float32),
                        guidance=np.float32(args.guidance),
                        ddim_steps=np.int32(args.steps),
                        sample_sd=np.float32(z.std()),
                        uuids=np.array([f"gen_{i:05d}" for i in range(len(lat))]))
    print(f"\nWrote {out}/generated.npz  latents={lat.shape}")
    print("\nNext:")
    print(f"  python diffusion_latent.py --mode validate --out-dir {out} "
          f"--latents-dir <real latents dir> --config {ck['config']}")


# --- Mode: validate ---
def validate(args):
    out = Path(args.out_dir)
    g = np.load(out / "generated.npz", allow_pickle=True)
    r = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                allow_pickle=True)
    G = g["latents"].reshape(len(g["latents"]), -1)
    R = r["latents"].reshape(len(r["latents"]), -1)
    print(f"generated {G.shape}, real {R.shape}\n")

    print("1) Distribution  (a prerequisite for the checks below)")
    print(f"   real      mean {R.mean():+.4f}  sd {R.std():.4f}  "
          f"range [{R.min():+.2f},{R.max():+.2f}]")
    print(f"   generated mean {G.mean():+.4f}  sd {G.std():.4f}  "
          f"range [{G.min():+.2f},{G.max():+.2f}]")
    sd_ratio = G.std() / max(R.std(), 1e-9)
    rng_ratio = (G.max() - G.min()) / max(R.max() - R.min(), 1e-9)
    print(f"   sd ratio {sd_ratio:.2f}, range ratio {rng_ratio:.2f} "
          f"(both should be ~1)")
    if 1.3 < sd_ratio < 2.0:
        print("   WARNING: generated/real sd ratio is %.2f (over-dispersed above 1.3)." % sd_ratio)
        print("     The memorisation and novelty results below are confounded: over-dispersed")
        print("     samples sit farther from everything. More --steps, longer training or a")
        print("     tighter --clamp-margin may reduce the dispersion.")
    if not (0.5 < sd_ratio < 2.0 and rng_ratio < 3.0):
        print("\n" + "=" * 66)
        print("   FAIL: the generated latents are outside the data distribution.")
        print("   The sampler has diverged; the samples are not spine latents.")
        print("   The checks below would be invalid: a 'novelty' or 'no memorisation'")
        print("   verdict here would only mean that the samples are far from everything,")
        print("   including from real data. The remaining checks are therefore skipped.")
        print("=" * 66)
        return

    print("\n2) Memorisation (the critical check)")
    rng = np.random.default_rng(0)
    sub = R[rng.choice(len(R), min(600, len(R)), replace=False)]

    def nn_dist(A, B, exclude_self=False):
        d = []
        for i in range(0, len(A), 64):
            blk = A[i:i + 64]
            dist = np.sqrt(((blk[:, None] - B[None]) ** 2).sum(-1))
            if exclude_self:
                np.fill_diagonal(dist[:, i:i + len(blk)], np.inf)
            d.append(dist.min(1))
        return np.concatenate(d)

    d_gen = nn_dist(G[:min(300, len(G))], sub)
    d_real = nn_dist(sub[:300], sub, exclude_self=True)
    ratio = float(np.median(d_gen) / max(np.median(d_real), 1e-9))
    print(f"   generated -> nearest real : median {np.median(d_gen):.3f}")
    print(f"   real      -> nearest real : median {np.median(d_real):.3f}")
    print(f"   ratio {ratio:.2f}")
    if ratio < 0.5:
        print("   WARNING: ratio is below 0.5; samples sit much closer to training points than")
        print("      training points sit to each other. The model is likely memorising, so the 'new data' is illusory.")
    elif ratio < 0.9:
        print("   WARNING: ratio is below 0.9, indicating partial memorisation; gains should be interpreted with caution.")
    elif ratio > 5.0:
        print("   WARNING: ratio is above 5.0; samples may be far from the data manifold")
        print("      rather than novel. The distribution check above is relevant here.")
    else:
        print("   Check passed: samples are not closer to the training set than it is to itself")

    print("\n3) Shape novelty")
    gl, rl = g["landmarks"], r["landmarks"]
    dl = nn_dist(gl[:min(300, len(gl))], rl)
    dr = nn_dist(rl[:300], rl, exclude_self=True)
    print(f"   generated shape -> nearest real shape: median {np.median(dl):.4f}")
    print(f"   real shape      -> nearest real shape: median {np.median(dr):.4f}")
    print(f"   ratio {np.median(dl)/max(np.median(dr),1e-9):.2f}  "
          "(>~1 means new anatomy)")

    print("\n4) Condition fidelity: run separately:")
    print("   train a landmark head on real latents, predict on generated.npz,")
    print("   and compare against the conditioning landmarks. If the detector")
    print("   recovers them, the conditioning works and the labels are reliable.")


# --- Diagnostics: reference detector and conditioned sampling ---
def _train_detector(Xtr, Ltr, device, epochs=100, seed=0):
    """seed is a parameter so that multi-seed conditions differ: a fixed seed
    would make every seed identical and hide the run-to-run variation needed to
    judge whether a small difference is real or noise."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv2d(Xtr.shape[1], 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        nn.AdaptiveAvgPool2d(3), nn.Flatten(),
        nn.Linear(128 * 9, 256), nn.GELU(), nn.Linear(256, 2 * N_LM)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X = torch.tensor(Xtr, device=device); Y = torch.tensor(Ltr, device=device)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            nn.functional.smooth_l1_loss(net(X[idx]), Y[idx], beta=0.05).backward()
            opt.step()
        sch.step()
    net.eval()
    return net


def _predict(net, A, device):
    import torch
    with torch.no_grad():
        return np.concatenate([net(torch.tensor(A[i:i + 64], device=device)).cpu().numpy()
                               for i in range(0, len(A), 64)], 0)


def _lm_err(pred, true):
    dx = pred[:, :N_LM] - true[:, :N_LM]
    dy = pred[:, N_LM:] - true[:, N_LM:]
    return np.sqrt(dx ** 2 + dy ** 2).mean(1)


def _sample_given(ck, net_d, cond, device, steps, guidance, clamp_margin, seed=0):
    """DDIM sample conditioned on a given condition tensor."""
    import torch
    sch = Schedule(ck["T"])
    x0_lo = ck.get("x0_lo", -8.0) - clamp_margin
    x0_hi = ck.get("x0_hi", 8.0) + clamp_margin
    stepsl = np.linspace(0, sch.T - 1, steps).round().astype(int)[::-1]
    abar = torch.tensor(sch.abar, device=device)
    torch.manual_seed(seed)
    outs = []
    with torch.no_grad():
        for i in range(0, len(cond), 32):
            c = torch.tensor(cond[i:i + 32], device=device)
            x = torch.randn(len(c), ck["in_ch"], ck["size"], ck["size"], device=device)
            for j, t in enumerate(stepsl):
                tt = torch.full((len(c),), int(t), device=device, dtype=torch.long)
                if guidance > 1.0:
                    ec = net_d(x, tt, c); eu = net_d(x, tt, torch.zeros_like(c))
                    eps = eu + guidance * (ec - eu)
                else:
                    eps = net_d(x, tt, c)
                a_t = abar[t]
                xx = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(x0_lo, x0_hi)
                if j + 1 < len(stepsl):
                    ap_ = abar[stepsl[j + 1]]
                    x = ap_.sqrt() * xx + (1 - ap_).sqrt() * eps
                else:
                    x = xx
            outs.append(x.cpu().numpy())
    return np.concatenate(outs, 0) * ck["sd"] + ck["mu"]


def diagnose(args):
    """Why is fidelity low when the samples look right?

    Three explanations make different predictions, so they can be separated:

      H1  Detector transfer failure. The detector is trained on real latents.
          If generated latents are off-manifold it may collapse toward the mean
          shape whatever the input, which looks like ignored conditioning.
          Signature: prediction spread on generated << on real.

      H2  Shape samples out of distribution. The model is conditioned on PCA
          samples at +/-2 SD. If those shapes are unrenderable the model may
          fall back to a generic spine. Signature: conditioning on real
          landmarks scores much better than conditioning on sampled ones.

      H3  Conditioning failure. Signature: neither of the above; and the
          predicted spine position does not track the conditioned position.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    ck = torch.load(out / "diffusion.pt", map_location=device, weights_only=False)
    net_d = build_unet(ck["in_ch"], ck["cond_ch"], ck["base"],
                       cross_attn=ck.get("cross_attn", False),
                       n_tokens=ck.get("n_tokens", 64)).to(device)
    net_d.load_state_dict(ck["state"]); net_d.eval()

    tr = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz", allow_pickle=True)
    te = np.load(Path(args.latents_dir) / f"latents_{args.config}_test.npz", allow_pickle=True)
    Xtr, Ltr = tr["latents"].astype(np.float32), tr["landmarks"].astype(np.float32)
    Xte, Lte = te["latents"].astype(np.float32), te["landmarks"].astype(np.float32)
    asp_te = load_aspects(te["uuids"], args.img_dir)

    print("Training the reference detector on real latents...")
    det = _train_detector(Xtr, Ltr, device, args.fid_epochs)

    g = np.load(out / "generated.npz", allow_pickle=True)
    Xg, Lg = g["latents"].astype(np.float32), g["landmarks"].astype(np.float32)

    p_real = _predict(det, Xte, device)
    p_gen = _predict(det, Xg, device)

    # H1: prediction spread
    gsd = Xg.std() / max(Xte.std(), 1e-9)
    if not (0.7 < gsd < 1.4):
        print("\n" + "=" * 64)
        print(f"WARNING: generated/real latent spread ratio is {gsd:.2f} (expected 0.7 to 1.4).")
        print("The sampler has likely diverged, so the diagnostics below are likely")
        print("measuring noise. Re-sampling at a lower --guidance may help.")
        print("=" * 64)

    print("\n" + "=" * 64)
    print("H1  Detector transfer: does the detector still discriminate?")
    print("=" * 64)
    sr, sg = p_real.std(0).mean(), p_gen.std(0).mean()
    st = Lte.std(0).mean()
    print(f"   sd of predictions on real latents      : {sr:.5f}")
    print(f"   sd of predictions on generated latents : {sg:.5f}")
    print(f"   sd of the true landmarks (reference)   : {st:.5f}")
    print(f"   ratio generated/real = {sg/max(sr,1e-9):.2f}")
    if sg / max(sr, 1e-9) < 0.5:
        print("   -> collapsed: the detector outputs near-constant landmarks on")
        print("      generated latents. This is a measurement failure (H1): the")
        print("      fidelity number says little about the conditioning.")
    else:
        print("   -> the detector still varies its output on generated latents,")
        print("      so it is discriminating; H1 does not explain the low score.")

    # H2: condition on real landmarks
    print("\n" + "=" * 64)
    print("H2  Shape samples out of distribution: condition on real landmarks instead")
    print("=" * 64)
    n = min(args.n_diag, len(Lte))
    cond_real = np.stack([render_condition(to_canvas_centered(Lte[i], asp_te[i]),
                                           size=ck["size"], sigma=ck["sigma"],
                                           n_ch=ck["cond_ch"]) for i in range(n)])
    print(f"   Generating {n} samples conditioned on real test landmarks...")
    Xrc = _sample_given(ck, net_d, cond_real, device, args.steps, args.guidance,
                        args.clamp_margin)
    e_rc = _lm_err(_predict(det, Xrc.astype(np.float32), device), Lte[:n])
    e_sc = _lm_err(p_gen, Lg)
    e_floor = _lm_err(p_real, Lte)
    base_rc = _lm_err(np.repeat(Ltr.mean(0, keepdims=True), n, 0), Lte[:n])
    print(f"\n   real-data floor                       : {e_floor.mean():.4f}")
    print(f"   generated, conditioned on real shapes : {e_rc.mean():.4f}")
    print(f"   generated, conditioned on sampled     : {e_sc.mean():.4f}")
    print(f"   mean-shape baseline (real conds)      : {base_rc.mean():.4f}")
    # Report a percentage only when the value lies between the baseline and the
    # floor. Outside that range the "fraction of the way to the floor" formula
    # divides by a span the value sits outside, giving uninterpretable numbers
    # (for example, large negative percentages).
    if e_rc.mean() > base_rc.mean():
        f_rc = -1.0
        print(f"\n   generated error ({e_rc.mean():.4f}) is worse than the "
              f"mean-shape baseline ({base_rc.mean():.4f});")
        print("   a fidelity percentage is undefined here.")
    else:
        f_rc = (base_rc.mean() - e_rc.mean()) / max(base_rc.mean() - e_floor.mean(), 1e-9)
        print(f"\n   fidelity with real conditioning: {100*f_rc:.0f}%")
    if f_rc > 0.4:
        print("   -> the mechanism works on real shapes; the PCA sampler is the")
        print("      problem (H2). A tighter --clip or a smaller --n-comp may help.")
    else:
        print("   -> real conditioning is no better, so the sampler is not the")
        print("      issue; H2 does not explain the low score.")

    # H3: does output position track conditioned position?
    print("\n" + "=" * 64)
    print("H3  Response: does the spine move when the condition moves?")
    print("=" * 64)
    cx_cond = Lg[:, :N_LM].mean(1)
    cx_pred = p_gen[:, :N_LM].mean(1)
    r = float(np.corrcoef(cx_cond, cx_pred)[0, 1])
    cy_cond = Lg[:, N_LM:].mean(1); cy_pred = p_gen[:, N_LM:].mean(1)
    ry = float(np.corrcoef(cy_cond, cy_pred)[0, 1])
    print(f"   corr(conditioned x-centroid, predicted x-centroid) = {r:+.3f}")
    print(f"   corr(conditioned y-centroid, predicted y-centroid) = {ry:+.3f}")
    if abs(r) < 0.2 and abs(ry) < 0.2:
        print("   -> no response: the generated spine does not move with the")
        print("      condition. This indicates a conditioning failure (H3).")
    else:
        print("   -> the output does track the condition, so conditioning is")
        print("      partially working even though absolute error is high.")
    print("\n" + "=" * 64)


# --- Mode: overfit ---
def overfit(args):
    """Can the model reproduce a handful of conditioned samples exactly?

    This test partitions the failure space:

      * Passes (recovers the 8 conditioned shapes well) -> the conditioning
        mechanism is mechanically sound. The failure is then about
        generalising conditioning from ~400 spines, which points to the
        conditioning architecture (concatenation is weak; cross-attention is
        the standard alternative).

      * Fails (cannot fit even 8 pairs) -> conditioning is broken mechanically
        (plumbing, capacity, or the condition carries no usable gradient).
        More data or guidance would not help, and the fault lies in the
        implementation.

    Overfitting is desirable here: the test measures capability, not
    generalisation, so memorisation is the intended outcome. A model that
    cannot memorise 8 conditioned pairs cannot learn conditioning from 2452
    of them.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    d = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                allow_pickle=True)
    lat, lm, uu = (d["latents"].astype(np.float32),
                   d["landmarks"].astype(np.float32), d["uuids"])

    # pick k spines with maximally different shapes, so "followed the
    # condition" is distinguishable from "produced a generic spine"
    k = args.n_overfit
    rng = np.random.default_rng(0)
    cand = rng.choice(len(lat), size=min(200, len(lat)), replace=False)
    picked = [int(cand[0])]
    while len(picked) < k:
        best, bd = None, -1
        for c in cand:
            if c in picked:
                continue
            dmin = min(np.abs(lm[c] - lm[q]).mean() for q in picked)
            if dmin > bd:
                best, bd = int(c), dmin
        picked.append(best)
    picked = np.array(picked)
    X, L, U = lat[picked], lm[picked], uu[picked]
    spread = np.abs(L - L.mean(0, keepdims=True)).mean()
    print(f"Selected {k} maximally distinct spines; mean |shape - mean shape| "
          f"= {spread:.4f}")

    mu, sd = float(X.mean()), float(X.std())
    Xs = (X - mu) / sd
    size = X.shape[-1]
    asp = load_aspects(U, args.img_dir)
    cond = np.stack([render_condition(to_canvas_centered(L[i], asp[i]), size=size,
                                      sigma=args.sigma, n_ch=args.cond_ch)
                     for i in range(k)])

    sch = Schedule(args.T)
    net = build_unet(X.shape[1], args.cond_ch, args.base,
                     cross_attn=args.cross_attn, n_tokens=args.n_tokens).to(device)
    print(f"UNet {sum(p.numel() for p in net.parameters())/1e6:.2f}M params, "
          f"training on {k} pairs for {args.overfit_epochs} epochs\n")

    Xt = torch.tensor(Xs, device=device)
    Ct = torch.tensor(cond, device=device)
    sqa = torch.tensor(sch.sqrt_abar, device=device)
    sq1 = torch.tensor(sch.sqrt_1mabar, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    for ep in range(args.overfit_epochs):
        net.train()
        # many noise draws per step: with only k samples the gradient is
        # dominated by the noise sampling, not the data
        for _ in range(args.inner):
            t = torch.randint(0, sch.T, (k,), device=device)
            noise = torch.randn_like(Xt)
            xt = sqa[t][:, None, None, None] * Xt + sq1[t][:, None, None, None] * noise
            loss = torch.nn.functional.mse_loss(net(xt, t, Ct), noise)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % max(1, args.overfit_epochs // 10) == 0:
            print(f"  epoch {ep+1:>4}/{args.overfit_epochs}  loss "
                  f"{float(loss.detach()):.5f}", flush=True)

    ck = {"in_ch": X.shape[1], "cond_ch": args.cond_ch, "base": args.base,
          "T": args.T, "size": size, "mu": mu, "sd": sd,
          "x0_lo": float(Xs.min()), "x0_hi": float(Xs.max())}
    net.eval()
    gen = _sample_given(ck, net, cond, device, args.steps, args.guidance,
                        args.clamp_margin)

    # a detector trained on real latents, as in --mode fidelity
    tr = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                 allow_pickle=True)
    print("\nTraining the reference detector on real latents...")
    det = _train_detector(tr["latents"].astype(np.float32),
                          tr["landmarks"].astype(np.float32), device, args.fid_epochs)

    p_gen = _predict(det, gen.astype(np.float32), device)
    p_orig = _predict(det, X, device)
    e_gen = _lm_err(p_gen, L)
    e_orig = _lm_err(p_orig, L)
    base = _lm_err(np.repeat(tr["landmarks"].astype(np.float32).mean(0, keepdims=True),
                             k, 0), L)

    print("\n" + "=" * 64)
    print("Overfit test: can the model reproduce these k conditioned spines?")
    print("=" * 64)
    print(f"   detector on the original {k} latents : {e_orig.mean():.4f}  <- floor")
    print(f"   detector on the generated {k}        : {e_gen.mean():.4f}")
    print(f"   mean-shape baseline                  : {base.mean():.4f}")
    if e_gen.mean() < base.mean():
        frac = (base.mean() - e_gen.mean()) / max(base.mean() - e_orig.mean(), 1e-9)
        print(f"\n   recovers {100*frac:.0f}% of the way from baseline to floor")
    else:
        frac = -1.0
        print("\n   worse than the mean-shape baseline")

    # does each generated sample match its own condition better than the others?
    # this is the sharpest form of the question and is immune to the absolute
    # error scale.
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i, j] = _lm_err(p_gen[i:i + 1], L[j:j + 1])[0]
    correct = int((M.argmin(1) == np.arange(k)).sum())
    print(f"\n   Matching: for each generated sample, is its own condition the")
    print(f"   closest of all {k}?  {correct}/{k} correct "
          f"(chance = {100.0/k:.0f}%)")

    print("\n" + "=" * 64)
    if frac > 0.6 or correct >= max(2, int(0.6 * k)):
        print("Verdict: the conditioning mechanism works.")
        print("  The model can follow conditioning when it is not asked to")
        print("  generalise, so the mechanism, coordinate convention, heatmap")
        print("  rendering and guidance are all sound. The failure at full scale")
        print("  is about learning a generalisable conditioning map from ~400")
        print("  distinct spines.")
        print("\n  Note: this does not by itself imply that cross-attention would fix it.")
        print("  The diagnosed limit is data-scale generalisation; a stronger")
        print("  conditioning mechanism could equally overfit harder. --mode")
        print("  scaling measures where adherence collapses.")
    else:
        print("Verdict: conditioning fails even on k samples.")
        print("  The model cannot reproduce conditioned images it has seen")
        print("  hundreds of times. This is a mechanical failure (plumbing,")
        print("  capacity, or a condition that carries no usable gradient),")
        print("  not a data-scale problem. Neither more data nor guidance nor")
        print("  cross-attention would help.")
    print("=" * 64)

    np.savez_compressed(out / "overfit.npz", generated=gen.astype(np.float32),
                        original=X, landmarks=L, uuids=U, match_matrix=M)
    print(f"\nWrote {out}/overfit.npz")


# --- Mode: scaling ---
def _binom_tail(x, n, p):
    """P(X >= x) under Binomial(n, p): is the match rate above chance?"""
    from math import comb
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x, n + 1)))


def _fit_and_score(lat, lm, asp, args, device, det, mean_shape, seed=0):
    """Train a conditional model on the given subset, sample with its own
    conditions, and score adherence. Returns (fidelity_fraction, match_rate)."""
    import torch
    k = len(lat)
    mu, sd = float(lat.mean()), float(lat.std())
    Xs = (lat - mu) / sd
    size = lat.shape[-1]
    cond = np.stack([render_condition(to_canvas_centered(lm[i], asp[i]), size=size,
                                      sigma=args.sigma, n_ch=args.cond_ch)
                     for i in range(k)])
    sch = Schedule(args.T)
    torch.manual_seed(seed)
    net = build_unet(lat.shape[1], args.cond_ch, args.base,
                     cross_attn=args.cross_attn, n_tokens=args.n_tokens).to(device)
    Xt = torch.tensor(Xs, device=device); Ct = torch.tensor(cond, device=device)
    sqa = torch.tensor(sch.sqrt_abar, device=device)
    sq1 = torch.tensor(sch.sqrt_1mabar, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    # Two ways to balance the sweep, which answer different questions:
    #
    #   scale_mode="steps"     : total gradient steps constant. Larger k then
    #       gets fewer exposures per sample, so a decline confounds data
    #       diversity with undertraining.
    #
    #   scale_mode="exposures" : exposures per sample constant, so total steps
    #       grow with k. Removes the undertraining confound, at the cost of
    #       much more compute at large k. This isolates data diversity.
    bs = min(args.batch_size, k)
    steps_per_epoch = max(1, k // bs)
    if args.scale_mode == "exposures":
        epochs = args.scale_exposures
    else:
        epochs = max(1, int(round(args.scale_steps / steps_per_epoch)))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(k, device=device)
        for i in range(0, k, bs):
            idx = perm[i:i + bs]
            t = torch.randint(0, sch.T, (len(idx),), device=device)
            c = Ct[idx]
            if args.p_uncond > 0:
                c = c * (torch.rand(len(idx), 1, 1, 1, device=device) > args.p_uncond)
            noise = torch.randn_like(Xt[idx])
            xt = (sqa[t][:, None, None, None] * Xt[idx]
                  + sq1[t][:, None, None, None] * noise)
            loss = torch.nn.functional.mse_loss(net(xt, t, c), noise)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    ck = {"in_ch": lat.shape[1], "cond_ch": args.cond_ch, "base": args.base,
          "T": args.T, "size": size, "mu": mu, "sd": sd,
          "x0_lo": float(Xs.min()), "x0_hi": float(Xs.max())}

    # Score on the most distinct spines in the subset, not the first m.
    # Random spines sit close to the mean shape, which collapses the
    # baseline-to-floor range and makes the fidelity metric uninterpretable.
    m = min(args.scale_eval, k)
    dist = np.abs(lm - lm.mean(0, keepdims=True)).mean(1)
    sel = np.argsort(dist)[::-1][:m]
    gen = _sample_given(ck, net, cond[sel], device, args.steps, args.guidance,
                        args.clamp_margin)
    p = _predict(det, gen.astype(np.float32), device)
    lm_e = lm[sel]
    e_gen = _lm_err(p, lm_e).mean()
    e_base = _lm_err(np.repeat(mean_shape, m, 0), lm_e).mean()
    e_floor = _lm_err(_predict(det, lat[sel], device), lm_e).mean()
    frac = ((e_base - e_gen) / max(e_base - e_floor, 1e-9)) if e_gen < e_base else -1.0

    M = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            M[i, j] = _lm_err(p[i:i + 1], lm_e[j:j + 1])[0]
    match = float((M.argmin(1) == np.arange(m)).mean())
    del net
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return frac, match, float(e_gen), float(e_base), float(e_floor), epochs


def scaling(args):
    """How does conditioning generalisation scale with the number of distinct spines?

    Design:

      1. Exposures, not steps. Every k trains for the same number of exposures
         per sample (default 6000, matching the overfit run). Holding total
         *steps* constant instead leaves large k undertrained.

      2. Fixed evaluation sets. A held-out set of spines is reserved up front and
         never trained on at any k, so every k is scored on identical shapes.
         Scoring each k on "its own most distinct spines" would make the
         baseline drift across k.

      3. Nested training subsets. The k=8 set is contained in k=16, which is
         contained in k=32, and so on. This removes the sampling variance that
         would otherwise dominate a 7-point curve.

      4. Held-in and held-out are both measured.
           held-in  : can the model bind conditioning to shapes it trained on?
                      (capacity / memorisation: the overfit test at every k)
           held-out : does the conditioning map generalise to new shapes?
                      (the main question)
         The gap between them measures generalisation. If held-in stays high
         while held-out stays at chance, the model memorises conditioning rather
         than learning it, and more data of this kind will not fix it.

    Primary metric is the own-condition match rate with a binomial p-value.
    Fidelity is reported but is fragile: it depends on how far the evaluation
    shapes sit from the mean shape.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    d = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                allow_pickle=True)
    lat, lm, uu = (d["latents"].astype(np.float32),
                   d["landmarks"].astype(np.float32), d["uuids"])

    # one copy per source image: augmented duplicates add no new anatomy
    seen, keep = set(), []
    for i, u in enumerate(uu):
        b = str(u).split("__aug")[0]
        if b not in seen:
            seen.add(b); keep.append(i)
    keep = np.array(keep)
    lat, lm, uu = lat[keep], lm[keep], uu[keep]
    asp = load_aspects(uu, args.img_dir)
    N = len(lat)
    print(f"{N} distinct source spines (augmented copies excluded)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_ho = args.scale_heldout
    ho = perm[:n_ho]                      # reserved: never trained on, at any k
    pool = perm[n_ho:]                    # nested training subsets drawn in order
    print(f"Reserved {n_ho} spines as a fixed held-out set (never trained on)")

    ks = sorted({k for k in args.scale_ks if k <= len(pool)} | {len(pool)})
    m_in = min(args.scale_eval, ks[0])    # held-in eval size: same spines at every k
    print(f"training pool {len(pool)}; sweeping k = {ks}")
    print(f"held-in eval: the first {m_in} pool spines (identical across k)")
    print(f"exposures per sample: {args.scale_exposures} at every k\n")

    print("Training the shared reference detector on real latents...")
    det = _train_detector(lat, lm, device, args.fid_epochs)
    mean_shape = lm.mean(0, keepdims=True)

    def score(net, ck, idx):
        """Match rate + fidelity for a fixed set of conditions."""
        m = len(idx)
        cond = np.stack([render_condition(to_canvas_centered(lm[i], asp[i]),
                                          size=ck["size"], sigma=args.sigma,
                                          n_ch=args.cond_ch) for i in idx])
        gen = _sample_given(ck, net, cond, device, args.steps, args.guidance,
                            args.clamp_margin)
        p = _predict(det, gen.astype(np.float32), device)
        L = lm[idx]
        e_gen = _lm_err(p, L).mean()
        e_base = _lm_err(np.repeat(mean_shape, m, 0), L).mean()
        e_floor = _lm_err(_predict(det, lat[idx], device), L).mean()
        frac = ((e_base - e_gen) / max(e_base - e_floor, 1e-9)) if e_gen < e_base else -1.0
        M = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                M[i, j] = _lm_err(p[i:i + 1], L[j:j + 1])[0]
        x = int((M.argmin(1) == np.arange(m)).sum())
        return x, m, _binom_tail(x, m, 1.0 / m), frac, float(e_gen), float(e_base)

    rows = []
    for k in ks:
        idx_tr = pool[:k]                              # nested
        sub_lat, sub_lm, sub_asp = lat[idx_tr], lm[idx_tr], asp[idx_tr]

        mu, sd = float(sub_lat.mean()), float(sub_lat.std())
        Xs = (sub_lat - mu) / sd
        size = sub_lat.shape[-1]
        cond = np.stack([render_condition(to_canvas_centered(sub_lm[i], sub_asp[i]),
                                          size=size, sigma=args.sigma,
                                          n_ch=args.cond_ch) for i in range(k)])
        sch = Schedule(args.T)
        torch.manual_seed(0)
        net = build_unet(sub_lat.shape[1], args.cond_ch, args.base,
                         cross_attn=args.cross_attn, n_tokens=args.n_tokens).to(device)
        Xt = torch.tensor(Xs, device=device); Ct = torch.tensor(cond, device=device)
        sqa = torch.tensor(sch.sqrt_abar, device=device)
        sq1 = torch.tensor(sch.sqrt_1mabar, device=device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

        bs = min(args.batch_size, k)
        epochs = args.scale_exposures          # == exposures per sample
        total = epochs * max(1, k // bs)
        print(f"k={k:>4}: {epochs} epochs x {max(1,k//bs)} steps = {total} steps", flush=True)
        for ep in range(epochs):
            net.train()
            pm = torch.randperm(k, device=device)
            for i in range(0, k, bs):
                bidx = pm[i:i + bs]
                t = torch.randint(0, sch.T, (len(bidx),), device=device)
                c = Ct[bidx]
                if args.p_uncond > 0:
                    c = c * (torch.rand(len(bidx), 1, 1, 1, device=device) > args.p_uncond)
                noise = torch.randn_like(Xt[bidx])
                xt = (sqa[t][:, None, None, None] * Xt[bidx]
                      + sq1[t][:, None, None, None] * noise)
                loss = torch.nn.functional.mse_loss(net(xt, t, c), noise)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()

        ck = {"in_ch": sub_lat.shape[1], "cond_ch": args.cond_ch, "base": args.base,
              "T": args.T, "size": size, "mu": mu, "sd": sd,
              "x0_lo": float(Xs.min()), "x0_hi": float(Xs.max())}

        xi, mi, pi, fi, egi, ebi = score(net, ck, pool[:m_in])   # held-in
        xo, mo, po, fo, ego, ebo = score(net, ck, ho)            # held-out
        rows.append({"k": k, "steps": total,
                     "in_match": xi / mi, "in_x": xi, "in_n": mi, "in_p": pi,
                     "in_fidelity": fi, "in_err": egi, "in_base": ebi,
                     "out_match": xo / mo, "out_x": xo, "out_n": mo, "out_p": po,
                     "out_fidelity": fo, "out_err": ego, "out_base": ebo})
        print(f"        held-in  {xi:>2}/{mi} ({100*xi/mi:3.0f}%) p={pi:8.2e}"
              f"   held-out {xo:>2}/{mo} ({100*xo/mo:3.0f}%) p={po:8.2e}", flush=True)
        del net
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "conditioning_scaling.csv", index=False)

    print("\n" + "=" * 72)
    print("Results  (chance = 1/n; * marks p < 0.05)")
    print("=" * 72)
    print(f"{'k':>5}{'steps':>9}{'held-in':>14}{'p':>11}{'held-out':>14}{'p':>11}")
    for r in rows:
        si = "*" if r["in_p"] < 0.05 else " "
        so = "*" if r["out_p"] < 0.05 else " "
        print(f"{r['k']:>5}{r['steps']:>9}"
              f"{r['in_x']:>8}/{r['in_n']:<4}{r['in_p']:>10.2e}{si}"
              f"{r['out_x']:>8}/{r['out_n']:<4}{r['out_p']:>10.2e}{so}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(df.k, 100 * df.in_match, "-o", color="#1C7293", lw=2.2, label="held-in (memorisation)")
        ax.plot(df.k, 100 * df.out_match, "-o", color="#B85042", lw=2.2, label="held-out (generalisation)")
        ax.axhline(100.0 / max(df.out_n.iloc[0], 1), color="0.6", ls="--", lw=1, label="chance")
        ax.set_xscale("log"); ax.set_xlabel("distinct training spines (k)")
        ax.set_ylabel("own-condition match rate (%)")
        ax.set_title("Conditioning: memorisation vs generalisation")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "conditioning_scaling.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {out}/conditioning_scaling.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\n" + "=" * 72)
    print("Interpretation")
    print("=" * 72)
    print("  held-in high, held-out at chance, at every k")
    print("     -> the model memorises conditioning rather than learning it.")
    print("        More data of this kind will not help; the conditioning signal")
    print("        is being used as an index into training examples.")
    print("  held-out rising with k")
    print("     -> generalisation is improving with anatomical diversity; the")
    print("        slope says how much more data would be needed.")
    print("  both at chance")
    print("     -> undertrained at this exposure count, or the run differs from")
    print("        the overfit test in some other way. Held-in at k=8 can be")
    print("        compared with the overfit result before the rest is interpreted.")
    print(f"\nWrote {out}/conditioning_scaling.csv")


# --- Mode: align ---
def align(args):
    """Determine empirically where landmarks sit in the latent's canvas.

    Conditioning depends on this, so it is measured rather than assumed. Three
    candidate conventions are overlaid on decoded real latents
    (alignment_check.png) and scored by the mean image intensity at the
    landmark positions:

      RAW      : landmarks scaled directly across the full canvas
      TOP-LEFT : landmarks scaled by the aspect ratio, image pasted top-left
                 (to_canvas)
      CENTRED  : landmarks scaled by the aspect ratio, image centred
                 (to_canvas_centered)

    Whichever lands on the actual spine is the correct convention. If none
    does, the encoder's preprocessing differs from all three and conditioning
    cannot be aligned without reading that loader.
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from medvae import MVAE

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(args.latents_dir) / f"latents_{args.config}_test.npz",
                allow_pickle=True)
    lat, lm, uu = d["latents"], d["landmarks"], d["uuids"]
    asp = load_aspects(uu, args.img_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvae = MVAE(model_name=args.config, modality="xray").to(device).eval()

    def dec(z):
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32, device=device)
            for obj, name in ((mvae.model, "decode"), (mvae, "decode")):
                fn = getattr(obj, name, None)
                if fn is None:
                    continue
                try:
                    o = fn(t)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    if torch.is_tensor(o):
                        return o.float().cpu().numpy()
                except Exception:
                    continue
        raise RuntimeError("decode failed")

    n = min(args.n_show, len(lat))
    img = dec(lat[:n])
    S = img.shape[-1]
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.2))
    axes = np.atleast_2d(axes)
    for i in range(n):
        pic = img[i].mean(0)
        for row, (name, coords) in enumerate(
                [("RAW", lm[i]),
                 ("TOP-LEFT", to_canvas(lm[i], asp[i])),
                 ("CENTRED", to_canvas_centered(lm[i], asp[i]))]):
            ax = axes[row, i]
            ax.imshow(pic, cmap="gray")
            ax.scatter(coords[:N_LM] * (S - 1), coords[N_LM:] * (S - 1),
                       s=6, c=["red", "lime", "cyan"][row])
            ax.set_title(f"{name}  (W/H={asp[i]:.2f})", fontsize=8)
            ax.axis("off")
    fig.suptitle("Which convention lands on the spine?  "
                 "RAW (red) / TOP-LEFT (green) / CENTRED (cyan)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "alignment_check.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # numeric version: image intensity sampled at the landmark positions.
    # The spine is bright, so the correct convention should sample brighter
    # pixels than the wrong one.
    def sample_at(coords, pic):
        xs = np.clip((coords[:N_LM] * (S - 1)).astype(int), 0, S - 1)
        ys = np.clip((coords[N_LM:] * (S - 1)).astype(int), 0, S - 1)
        return float(pic[ys, xs].mean())

    raw_v, tl_v, ct_v, bg = [], [], [], []
    for i in range(min(40, len(lat))):
        pic = dec(lat[i:i + 1])[0].mean(0)
        raw_v.append(sample_at(lm[i], pic))
        tl_v.append(sample_at(to_canvas(lm[i], asp[i]), pic))
        ct_v.append(sample_at(to_canvas_centered(lm[i], asp[i]), pic))
        bg.append(float(pic.mean()))
    scores = {"RAW": np.mean(raw_v), "TOP-LEFT": np.mean(tl_v),
              "CENTRED": np.mean(ct_v)}
    print(f"\nMean image intensity at the landmark positions (spine is bright):")
    for k, v in scores.items():
        print(f"   {k:<9}: {v:+.4f}")
    print(f"   {'background':<9}: {np.mean(bg):+.4f}  <- reference")
    best = max(scores, key=scores.get)
    print(f"\n   -> {best} samples the brightest pixels, so it is the convention")
    print(f"      that tracks the spine.")
    if max(scores.values()) < np.mean(bg) + 0.05:
        print("   WARNING: no convention samples clearly brighter than the background (margin 0.05).")
        print("      The encoder's preprocessing likely differs from these assumptions; load_2d_finetune_aspect defines it.")
    print(f"\nWrote {out}/alignment_check.png for visual confirmation.")


# --- Mode: fidelity ---
def fidelity(args):
    """Condition fidelity: checks whether the sampled labels match the generated images.

    Train a landmark detector on real latents only, then run it on the
    generated latents and compare its predictions against the landmarks the
    samples were conditioned on. Three reference points make the number
    interpretable:

      * detector error on real held-out data      -> the achievable floor
      * detector error on generated data          -> the measurement
      * mean-shape predictor on the conditions    -> the trivial baseline

    If generated error is near the real floor, the conditioning works and the
    sampled landmarks are reliable labels. If it is near the mean-shape
    baseline, the model ignored the conditioning and the labels are invalid;
    training on this data would inject systematic label noise.
    """
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    g = np.load(out / "generated.npz", allow_pickle=True)
    Xg, Lg = g["latents"].astype(np.float32), g["landmarks"].astype(np.float32)

    tr = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz",
                 allow_pickle=True)
    te = np.load(Path(args.latents_dir) / f"latents_{args.config}_test.npz",
                 allow_pickle=True)
    Xtr, Ltr = tr["latents"].astype(np.float32), tr["landmarks"].astype(np.float32)
    Xte, Lte = te["latents"].astype(np.float32), te["landmarks"].astype(np.float32)
    print(f"real train {Xtr.shape}, real test {Xte.shape}, generated {Xg.shape}\n")

    def head(in_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(3), nn.Flatten(),
            nn.Linear(128 * 9, 256), nn.GELU(), nn.Linear(256, 2 * N_LM))

    def err(pred, true):
        dx = pred[:, :N_LM] - true[:, :N_LM]
        dy = pred[:, N_LM:] - true[:, N_LM:]
        return np.sqrt(dx ** 2 + dy ** 2).mean(1)

    torch.manual_seed(0)
    net = head(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.fid_epochs)
    X = torch.tensor(Xtr, device=device); Y = torch.tensor(Ltr, device=device)
    n = len(X)
    for ep in range(args.fid_epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(net(X[idx]), Y[idx], beta=0.05)
            loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 25 == 0:
            print(f"  detector epoch {ep+1}/{args.fid_epochs} loss {float(loss.detach()):.5f}",
                  flush=True)

    net.eval()

    def predict(A):
        with torch.no_grad():
            return np.concatenate(
                [net(torch.tensor(A[i:i + 64], device=device)).cpu().numpy()
                 for i in range(0, len(A), 64)], 0)

    e_real = err(predict(Xte), Lte)
    e_gen = err(predict(Xg), Lg)
    mean_shape = np.repeat(Ltr.mean(0, keepdims=True), len(Lg), 0)
    e_base = err(mean_shape, Lg)

    print("\n" + "=" * 62)
    print("Condition fidelity")
    print("=" * 62)
    print(f"  detector on real held-out   : {e_real.mean():.4f}  <- achievable floor")
    print(f"  detector on generated       : {e_gen.mean():.4f}  <- the measurement")
    print(f"  mean-shape on the conditions: {e_base.mean():.4f}  <- trivial baseline")
    span = max(e_base.mean() - e_real.mean(), 1e-9)
    frac = (e_base.mean() - e_gen.mean()) / span
    if e_gen.mean() > e_base.mean():
        print(f"\n  generated error is worse than the trivial baseline; a "
              f"'fraction of the way to the floor' figure is undefined.")
    else:
        print(f"\n  generated recovers {100*frac:.0f}% of the way from the trivial")
        print(f"  baseline to the real-data floor")
    if frac > 0.7:
        print("\n  Result: strong; the conditioning works and the sampled landmarks are usable labels.")
    elif frac > 0.4:
        print("\n  Result: partial; conditioning is followed loosely. Labels carry noise;")
        print("  any downstream gain may be offset by the label error introduced.")
    else:
        print("\n  WARNING: fidelity is weak (at most 0.4 of the way from baseline to floor). The model")
        print("  largely ignored the conditioning, so the sampled landmarks are likely invalid labels for these images.")
    print("\n  Caveat: the detector was trained on real latents, so generated error")
    print("  also reflects any domain gap between real and generated latents.")


# --- Mode: utility ---
def utility(args):
    """Are the synthetic labels usable? Two questions, answered in order.

    Q1  Filtering. Label quality can vary considerably per sample within one
        batch. If the best fraction is near the real-data
        reference, rejection sampling gives a usable subset even when the
        average is poor. This reports the error distribution by percentile.

    Q2  Utility (the main test). Train a landmark detector on synthetic data
        only and evaluate it on the real test set. If the labels are wrong, the
        detector learns wrong associations and fails. If they are broadly
        right, it should transfer. This avoids the proxy problem: for a
        generated image there is no ground truth, so the other measurements
        compare the conditioning against a detector rather than against truth.
        Downstream transfer needs no such proxy.

    Conditions reported:
        real-only          : trained on real train latents        (upper reference)
        synthetic-all      : trained on every generated sample
        synthetic-filtered : trained on the best `--keep-frac` by detector agreement
        real + synthetic   : augmentation setting (the realistic use case)

    Caveat on filtering: keeping samples a detector already agrees with
    selects for anatomy that detector finds easy, which is likely typical rather
    than unusual anatomy; the opposite of what a data-limited task needs. The
    selection bias is reported alongside the gain so that it can be judged.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    g = np.load(out / "generated.npz", allow_pickle=True)
    Xg, Lg = g["latents"].astype(np.float32), g["landmarks"].astype(np.float32)
    if "guidance" in g.files:
        gd, ssd = float(g["guidance"]), float(g.get("sample_sd", np.float32(1.0)))
        print(f"generated samples: guidance={gd}, standardised sd={ssd:.3f}")
        if gd > 4.0 or abs(ssd - 1.0) > 0.3:
            raise SystemExit("these samples appear to come from a diverged run (guidance above 4.0 "
                             "or standardised sd more than 0.3 from 1.0); re-sampling at "
                             "--guidance 3.0 is required")

    tr = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz", allow_pickle=True)
    va = np.load(Path(args.latents_dir) / f"latents_{args.config}_val.npz", allow_pickle=True)
    te = np.load(Path(args.latents_dir) / f"latents_{args.config}_test.npz", allow_pickle=True)
    Xtr, Ltr = tr["latents"].astype(np.float32), tr["landmarks"].astype(np.float32)
    Xva, Lva = va["latents"].astype(np.float32), va["landmarks"].astype(np.float32)
    Xte, Lte = te["latents"].astype(np.float32), te["landmarks"].astype(np.float32)
    print(f"real train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}, "
          f"generated {Xg.shape}\n")

    # Q1: how is label quality distributed?
    print("=" * 68)
    print("Q1  Label quality distribution")
    print("=" * 68)
    print("Training the reference detector on real latents...")
    ref = _train_detector(Xtr, Ltr, device, args.fid_epochs)
    agree = _lm_err(_predict(ref, Xg, device), Lg)          # detector-vs-condition
    real_ref = _lm_err(_predict(ref, Xte, device), Lte).mean()
    print(f"\n   detector error on real test data: {real_ref:.4f}  <- the reference")
    print(f"\n   {'percentile':>12}{'agreement':>12}{'vs real ref':>14}")
    for q in (10, 25, 50, 75, 90):
        v = float(np.percentile(agree, q))
        print(f"   {q:>10}%{v:>12.4f}{v/real_ref:>13.2f}x")
    n_good = int((agree < real_ref).sum())
    print(f"\n   {n_good}/{len(agree)} samples ({100*n_good/len(agree):.0f}%) have "
          f"agreement better than the detector's own error on real data")

    order = np.argsort(agree)
    k_keep = max(16, int(args.keep_frac * len(Xg)))
    keep = order[:k_keep]
    print(f"\n   keeping the best {k_keep} ({100*k_keep/len(Xg):.0f}%): "
          f"mean agreement {agree[keep].mean():.4f} vs {agree.mean():.4f} overall")

    # selection bias: are the kept shapes less extreme than the rest?
    dev = np.abs(Lg - Ltr.mean(0, keepdims=True)).mean(1)
    print(f"\n   Selection bias check: distance from the mean shape:")
    print(f"     kept    {dev[keep].mean():.4f}")
    print(f"     dropped {dev[order[k_keep:]].mean():.4f}")
    print(f"     real    {np.abs(Ltr - Ltr.mean(0, keepdims=True)).mean(1).mean():.4f}")
    if dev[keep].mean() < 0.9 * dev[order[k_keep:]].mean():
        print("     -> filtering favours typical anatomy; the unusual spines a")
        print("        data-limited task most needs are preferentially discarded.")
    else:
        print("     -> no strong bias toward typical anatomy")

    # Q2: does training on it work?
    print("\n" + "=" * 68)
    print("Q2  Downstream utility  (the main test)")
    print("=" * 68)
    print("No proxy is needed here: a detector trained on synthetic data is judged")
    print("on real test data with real labels.\n")

    arms = [
        ("real only", Xtr, Ltr),
        ("synthetic all", Xg, Lg),
        ("synthetic filtered", Xg[keep], Lg[keep]),
        ("real + synthetic filt.", np.concatenate([Xtr, Xg[keep]]),
                                   np.concatenate([Ltr, Lg[keep]])),
        ("real + synthetic all", np.concatenate([Xtr, Xg]),
                                 np.concatenate([Ltr, Lg])),
    ]
    base = _lm_err(np.repeat(Ltr.mean(0, keepdims=True), len(Lte), 0), Lte).mean()

    rows, per_image = [], {}
    for name, X, L in arms:
        errs, ens = [], []
        for seed in args.util_seeds:
            net = _train_detector(X, L, device, args.fid_epochs, seed=seed)
            pv = _predict(net, Xte, device)
            ens.append(pv)
            errs.append(_lm_err(pv, Lte).mean())
            del net
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        per_image[name] = _lm_err(np.mean(ens, axis=0), Lte)   # seed-ensembled
        m, sd = float(np.mean(errs)), float(np.std(errs))
        rows.append({"arm": name, "n_train": len(X), "test_err": m, "seed_sd": sd,
                     "ens_err": float(per_image[name].mean())})
        print(f"   {name:<24} n={len(X):>5}   test {m:.4f} (+/-{sd:.4f})"
              f"   ensembled {per_image[name].mean():.4f}", flush=True)
    print(f"   {'mean-shape baseline':<24} {'':>7}   test {base:.4f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "synthetic_utility.csv", index=False)

    r0 = df[df.arm == "real only"].test_err.iloc[0]
    print("\n" + "=" * 68)
    print("Verdict")
    print("=" * 68)
    # paired test on the same test images: a small difference can be within
    # seed noise, so a difference of means is not sufficient evidence
    from scipy import stats as _st
    ref_pi = per_image["real only"]
    for _, r in df.iterrows():
        if r["arm"] == "real only":
            continue
        d = per_image[r["arm"]] - ref_pi
        _, pv = _st.ttest_rel(per_image[r["arm"]], ref_pi)
        rel = 100.0 * (r["test_err"] - r0) / r0
        print(f"   {r['arm']:<24} paired vs real-only: {d.mean():+.5f} "
              f"(p={pv:.4f}{'*' if pv < 0.05 else ' '})")
    print()
    for _, r in df.iterrows():
        if r["arm"] == "real only":
            continue
        rel = 100.0 * (r["test_err"] - r0) / r0
        tag = "BETTER than real-only" if rel < -1 else (
              "no better than real-only" if rel < 3 else "WORSE than real-only")
        print(f"   {r['arm']:<24} {rel:+6.1f}% vs real-only   -> {tag}")
    syn = df[df.arm == "synthetic all"].test_err.iloc[0]
    print(f"\n   synthetic-only vs the trivial baseline: {syn:.4f} vs {base:.4f}")
    if syn < 0.9 * base:
        print("   -> training on synthetic data alone does learn something;")
        print("      the labels carry usable signal even if they are imprecise.")
    else:
        print("   -> synthetic-only is no better than guessing the mean shape;")
        print("      the labels do not carry usable spatial information.")
    print(f"\nWrote {out}/synthetic_utility.csv")


# --- Mode: showlabels ---
def showlabels(args):
    """Visualise label error: decode generated latents and overlay both the
    conditioning landmarks (the requested shape) and the detector's landmarks
    (what was drawn).

    The image complements the fidelity number, which is a single, abstract
    fraction of the way from baseline to floor. The picture shows whether the
    spine is in roughly the right place with the wrong curve, or somewhere else
    entirely, and those imply different things.

    Real images are shown in the same format as a reference, showing what a
    correctly labelled pair looks like under the identical pipeline.
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from medvae import MVAE

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    g = np.load(out / "generated.npz", allow_pickle=True)
    Xg, Lg = g["latents"].astype(np.float32), g["landmarks"].astype(np.float32)
    asp_g = g["aspects"] if "aspects" in g.files else np.full(len(Xg), 0.5, np.float32)

    # Which samples are these? A guidance sweep overwrites generated.npz on every
    # iteration, so the file can hold the last run, which may have diverged,
    # rather than the intended one. The stored guidance and sample sd are
    # printed so that the figure can be read with this in mind.
    gd = float(g["guidance"]) if "guidance" in g.files else None
    ssd = float(g["sample_sd"]) if "sample_sd" in g.files else None
    if gd is not None:
        print(f"samples were generated with guidance={gd}, "
              f"{int(g['ddim_steps']) if 'ddim_steps' in g.files else '?'} DDIM steps"
              + (f", standardised sd={ssd:.3f}" if ssd is not None else ""))
        if gd > 4.0 or (ssd is not None and abs(ssd - 1.0) > 0.3):
            print("\n" + "=" * 68)
            print("WARNING: guidance is above 4.0 or the standardised sd is more than 0.3 from 1.0;")
            print("these samples are likely from a diverged or high-guidance run. The decoded")
            print("images will look washed out and the landmark errors will be inflated.")
            print("Re-sampling at --guidance 3.0 gives samples suitable for this figure:")
            print(f"   python diffusion_latent.py --mode sample --out-dir {out} "
                  f"--n 500 --steps 500 --guidance 3.0")
            print("=" * 68 + "\n")
    else:
        print("Note: this generated.npz does not record its guidance setting; if a guidance "
              "sweep wrote to this directory, it may hold the last run, which may have diverged.")

    tr = np.load(Path(args.latents_dir) / f"latents_{args.config}_train.npz", allow_pickle=True)
    te = np.load(Path(args.latents_dir) / f"latents_{args.config}_test.npz", allow_pickle=True)
    Xte, Lte = te["latents"].astype(np.float32), te["landmarks"].astype(np.float32)
    asp_te = load_aspects(te["uuids"], args.img_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training the reference detector on real latents...")
    det = _train_detector(tr["latents"].astype(np.float32),
                          tr["landmarks"].astype(np.float32), device, args.fid_epochs)

    mvae = MVAE(model_name=args.config, modality="xray").to(device).eval()

    def dec(z):
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32, device=device)
            for obj, nm in ((mvae.model, "decode"), (mvae, "decode")):
                fn = getattr(obj, nm, None)
                if fn is None:
                    continue
                try:
                    o = fn(t)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    if torch.is_tensor(o):
                        return o.float().cpu().numpy()
                except Exception:
                    continue
        raise RuntimeError("decode failed")

    n = args.n_show
    Pg = _predict(det, Xg[:n], device)
    Pte = _predict(det, Xte[:n], device)
    img_g, img_r = dec(Xg[:n]), dec(Xte[:n])
    S = img_g.shape[-1]

    def draw(ax, pic, asked, drawn, asp, title):
        ax.imshow(pic, cmap="gray")
        a = to_canvas_centered(asked, asp)
        d = to_canvas_centered(drawn, asp)
        ax.scatter(a[:N_LM] * (S - 1), a[N_LM:] * (S - 1), s=7, c="lime",
                   edgecolors="black", linewidths=0.3, label="asked for (condition)")
        ax.scatter(d[:N_LM] * (S - 1), d[N_LM:] * (S - 1), s=7, c="red",
                   edgecolors="black", linewidths=0.3, label="detector found")
        err = _lm_err(drawn[None], asked[None])[0]
        ax.set_title(f"{title}  err {err:.4f}", fontsize=8)
        ax.axis("off")

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.6))
    axes = np.atleast_2d(axes)
    for i in range(n):
        draw(axes[0, i], img_g[i].mean(0), Lg[i], Pg[i], asp_g[i], "GENERATED")
        draw(axes[1, i], img_r[i].mean(0), Lte[i], Pte[i], asp_te[i], "REAL")
    axes[0, 0].legend(loc="upper right", fontsize=6, framealpha=0.85)
    fig.suptitle("Green = landmarks we CONDITIONED on   |   Red = landmarks the detector FOUND\n"
                 "top: generated (labels unusable)      bottom: real (reference)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "label_quality.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    eg = _lm_err(Pg, Lg[:len(Pg)])
    er = _lm_err(Pte, Lte[:len(Pte)])
    print(f"\nmean |asked - found|:  generated {eg.mean():.4f}   real {er.mean():.4f}")
    print(f"in pixels on a {S}px canvas: generated ~{eg.mean()*S:.0f}px, real ~{er.mean()*S:.0f}px")
    print("\nInterpretation of the figure (what kind of error, not whether there is error):")
    print("  * red near green, wrong curve shape -> conditioning loosely followed")
    print("  * red forming a generic centred spine regardless of green -> ignored")
    print("  * red scattered/nonsensical -> the detector is failing, not the model")
    print(f"\nWrote {out}/label_quality.png")


# --- Mode: decode ---
def decode(args):
    """Decode generated latents through MedVAE and save a comparison figure.

    This is the first validation criterion (do samples decode to spine-like
    images?). Distribution statistics cannot tell "structurally correct but
    noisy" from "structured noise that is not a spine"; visual inspection can.
    Real reconstructions are shown alongside so the comparison is against what
    MedVAE can produce, not against the original radiograph."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from medvae import MVAE

    out = Path(args.out_dir)
    g = np.load(out / "generated.npz", allow_pickle=True)
    gen = g["latents"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mvae = MVAE(model_name=args.config, modality="xray").to(device).eval()

    def dec(z):
        with torch.no_grad():
            t = torch.tensor(z, dtype=torch.float32, device=device)
            for obj, name in ((mvae.model, "decode"), (mvae, "decode"),
                              (mvae.model, "decode_first_stage")):
                fn = getattr(obj, name, None)
                if fn is None:
                    continue
                try:
                    o = fn(t)
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    if torch.is_tensor(o):
                        return o.float().cpu().numpy()
                except Exception:
                    continue
        raise RuntimeError("could not decode; the MedVAE API may differ from the entry points tried")

    n = min(args.n_show, len(gen))
    gimg = dec(gen[:n])

    real = None
    if args.latents_dir:
        r = np.load(Path(args.latents_dir) / f"latents_{args.config}_test.npz",
                    allow_pickle=True)["latents"][:n]
        real = dec(r)

    rows = 2 if real is not None else 1
    fig, axes = plt.subplots(rows, n, figsize=(2.0 * n, 2.2 * rows * 2))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[0, i].imshow(gimg[i].mean(0), cmap="gray")
        axes[0, i].set_title("generated", fontsize=8)
        axes[0, i].axis("off")
        if real is not None:
            axes[1, i].imshow(real[i].mean(0), cmap="gray")
            axes[1, i].set_title("real (reconstructed)", fontsize=8)
            axes[1, i].axis("off")
    fig.suptitle("Generated vs real MedVAE reconstructions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "generated_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Decoded {n} samples -> {out}/generated_samples.png")
    print(f"  generated pixel range [{gimg.min():+.2f},{gimg.max():+.2f}]")
    if real is not None:
        print(f"  real      pixel range [{real.min():+.2f},{real.max():+.2f}]")
    print("\nQuestions for the figure:")
    print("  * is there a recognisable vertebral column?")
    print("  * does the curve follow the conditioning shape?")
    print("  * are ribs/pelvis plausible, or is it textured noise?")
    print("If it is not spine-like, the downstream results are likely invalid.")


# --- Synthetic-data self-test ---
def smoke():
    rng = np.random.default_rng(0)

    print("1) cosine schedule")
    for T in (100, 1000):
        s = Schedule(T)
        print(f"   T={T:>4}: abar[0]={s.abar[0]:.4f} -> abar[-1]={s.abar[-1]:.2e}, "
              f"betas in [{s.betas.min():.2e},{s.betas.max():.3f}]")
    s = Schedule(1000)
    mono = bool(np.all(np.diff(s.abar) < 0))
    print(f"   abar monotonically decreasing: {mono}")

    print("\n2) condition rendering")
    lm = np.concatenate([np.linspace(0.4, 0.6, N_LM), np.linspace(0.05, 0.95, N_LM)])
    c = render_condition(lm.astype(np.float32), size=96, n_ch=4)
    peak_ok = c.max() > 0.9
    print(f"   shape {c.shape}, range [{c.min():.3f},{c.max():.3f}], "
          f"peaks present: {peak_ok}")
    # a landmark moved right must move the heatmap mass right
    lm2 = lm.copy(); lm2[:N_LM] += 0.2
    c2 = render_condition(lm2.astype(np.float32), size=96, n_ch=4)
    cx1 = (c.sum((0, 1)) * np.arange(96)).sum() / c.sum()
    cx2 = (c2.sum((0, 1)) * np.arange(96)).sum() / c2.sum()
    print(f"   centroid x: {cx1:.1f} -> {cx2:.1f} after shifting landmarks right")

    print("\n3) shape model")
    real = np.clip(0.5 + 0.05 * rng.standard_normal((200, 136)), 0, 1).astype(np.float32)
    real[:, :N_LM] += 0.1 * rng.standard_normal((200, 1))     # a dominant mode
    sm = ShapeModel(real, n_comp=10)
    samp = sm.sample(rng, 50)
    in_range = bool((samp.min() >= 0) and (samp.max() <= 1))
    print(f"   {100*sm.evr:.1f}% variance in 10 comps; samples {samp.shape}, "
          f"within [0,1]: {in_range}")
    print(f"   sample sd {samp.std():.4f} vs real sd {real.std():.4f}")

    print("\n4) forward diffusion recovers noise")
    try:
        import torch
        net = build_unet(1, 4, base=16)
        npar = sum(p.numel() for p in net.parameters())
        x = torch.randn(2, 1, 96, 96); t = torch.randint(0, 1000, (2,))
        cond = torch.rand(2, 4, 96, 96)
        y = net(x, t, cond)
        print(f"   UNet {npar/1e6:.2f}M params, output {tuple(y.shape)} "
              f"(expect (2, 1, 96, 96))")
        ok_net = tuple(y.shape) == (2, 1, 96, 96)
    except ImportError:
        print("   torch unavailable; skipping the UNet check")
        ok_net = True


    print("\n5) cross-attention variant")
    try:
        import torch
        n1 = build_unet(1, 5, base=16, cross_attn=False)
        n2 = build_unet(1, 5, base=16, cross_attn=True, n_tokens=16)
        p1 = sum(p.numel() for p in n1.parameters())
        p2 = sum(p.numel() for p in n2.parameters())
        x = torch.randn(2, 1, 96, 96); t = torch.randint(0, 1000, (2,))
        c = torch.rand(2, 5, 96, 96)
        y1, y2 = n1(x, t, c), n2(x, t, c)
        print(f"   concat-only    : {p1/1e6:.2f}M params, out {tuple(y1.shape)}")
        print(f"   + cross-attn   : {p2/1e6:.2f}M params, out {tuple(y2.shape)}")
        # the output projection is zero-initialised, so cross-attn starts as a
        # no-op and cannot destabilise early training
        n2b = build_unet(1, 5, base=16, cross_attn=False)
        n2b.load_state_dict({k: v for k, v in n2.state_dict().items()
                             if k in n2b.state_dict()}, strict=False)
        with torch.no_grad():
            d = float((n2(x, t, c) - n2b(x, t, c)).abs().mean())
        print(f"   cross-attn contribution at init: {d:.2e} (zero-init -> ~0)")
        ok_x = tuple(y2.shape) == (2, 1, 96, 96) and p2 > p1
    except ImportError:
        print("   torch unavailable; skipping the cross-attention check")
        ok_x = True

    ok = mono and peak_ok and cx2 > cx1 + 5 and in_range and ok_net and ok_x
    print("\nSelf-test:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["train", "sample", "validate", "decode", "fidelity",
                             "align", "diagnose", "overfit", "scaling", "showlabels",
                             "utility"],
                    default="train")
    ap.add_argument("--latents-dir", type=Path)
    ap.add_argument("--img-dir", type=Path,
                    help="required for training: used to map landmarks into "
                         "aspect-padded canvas coordinates")
    ap.add_argument("--config", default="medvae_4_1_2d")
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/diffusion"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64, help="UNet width")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--cond-ch", type=int, default=5,
                    help="4 corner heatmaps + 1 filled vertebra-body channel")
    ap.add_argument("--sigma", type=float, default=3.0, help="condition heatmap sigma")
    ap.add_argument("--cross-attn", action="store_true",
                    help="inject the condition by cross-attention as well as "
                         "concatenation (the standard mechanism for non-local "
                         "conditioning in text-to-image diffusion)")
    ap.add_argument("--n-tokens", type=int, default=64,
                    help="condition tokens for cross-attention")
    ap.add_argument("--p-uncond", type=float, default=0.15,
                    help="condition-dropout rate for classifier-free guidance")
    ap.add_argument("--guidance", type=float, default=3.0,
                    help="CFG scale at sampling; 1.0 disables guidance")
    ap.add_argument("--n-comp", type=int, default=20, help="shape-model components")
    ap.add_argument("--n", type=int, default=500, help="samples to generate")
    ap.add_argument("--steps", type=int, default=250,
                    help="DDIM steps; too few causes over-dispersion because the "
                         "early large-abar jumps are poorly approximated")
    ap.add_argument("--clip", type=float, default=2.0, help="shape sampling range, SD")
    ap.add_argument("--clamp-margin", type=float, default=1.0,
                    help="slack added to the training data range when clamping x0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--n-show", type=int, default=6, help="samples to decode and plot")
    ap.add_argument("--keep-frac", type=float, default=0.2,
                    help="fraction of generated samples to keep when filtering by "
                         "detector agreement")
    ap.add_argument("--util-seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds per condition in the utility test")
    ap.add_argument("--n-overfit", type=int, default=8,
                    help="how many pairs to overfit")
    ap.add_argument("--overfit-epochs", type=int, default=300)
    ap.add_argument("--inner", type=int, default=20,
                    help="noise draws per epoch (few samples -> noise-dominated gradient)")
    ap.add_argument("--scale-ks", nargs="+", type=int,
                    default=[8, 16, 32, 64, 128, 256],
                    help="training-set sizes to sweep (full size is appended)")
    ap.add_argument("--scale-exposures", type=int, default=6000,
                    help="exposures per sample, identical at every k. 6000 matches "
                         "the overfit run (300 epochs x 20 noise draws); much smaller "
                         "values leave large k undertrained.")
    ap.add_argument("--scale-heldout", type=int, default=16,
                    help="spines reserved from all training sets, used as the "
                         "fixed held-out evaluation set at every k")
    ap.add_argument("--scale-eval", type=int, default=16,
                    help="samples scored per k")
    ap.add_argument("--n-diag", type=int, default=128,
                    help="samples to generate for the real-conditioning test")
    ap.add_argument("--fid-epochs", type=int, default=100,
                    help="epochs for the condition-fidelity detector")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.mode == "train":
        if not args.latents_dir:
            raise SystemExit("--latents-dir is required for training")
        train(args)
    elif args.mode == "sample":
        sample(args)
    elif args.mode == "decode":
        decode(args)
    elif args.mode == "scaling":
        if not (args.latents_dir and args.img_dir):
            raise SystemExit("--latents-dir and --img-dir are required for scaling")
        scaling(args)
    elif args.mode == "overfit":
        if not (args.latents_dir and args.img_dir):
            raise SystemExit("--latents-dir and --img-dir are required for overfit")
        overfit(args)
    elif args.mode == "utility":
        if not args.latents_dir:
            raise SystemExit("--latents-dir is required for utility")
        utility(args)
    elif args.mode == "showlabels":
        if not (args.latents_dir and args.img_dir):
            raise SystemExit("--latents-dir and --img-dir are required for showlabels")
        showlabels(args)
    elif args.mode == "diagnose":
        if not (args.latents_dir and args.img_dir):
            raise SystemExit("--latents-dir and --img-dir are required for diagnose")
        diagnose(args)
    elif args.mode == "align":
        if not (args.latents_dir and args.img_dir):
            raise SystemExit("--latents-dir and --img-dir are required for align")
        align(args)
    elif args.mode == "fidelity":
        if not args.latents_dir:
            raise SystemExit("--latents-dir is required for fidelity")
        fidelity(args)
    else:
        if not args.latents_dir:
            raise SystemExit("--latents-dir is required for validation")
        validate(args)


if __name__ == "__main__":
    main()
