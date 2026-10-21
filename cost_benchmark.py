"""
cost_benchmark.py: storage, encode-time, memory and downstream-head cost per MedVAE configuration.

Accuracy alone cannot determine a compression configuration. This script
measures what each configuration costs, so that the choice can be based on
accuracy per unit cost for a given task rather than on lowest error alone.

Measured:
  storage    latent bytes/image, and the compression ratio against the source
             image on disk (not only the nominal f/C figure)
  encode     per-image encode latency and throughput (images/sec)
  memory     peak GPU memory during encoding
  downstream training-step time and inference latency for the small head that
             consumes the latent; this is the cost that scales in deployment

Timing method (needed for the timings to be meaningful):
  * warmup iterations before measuring, so cuDNN autotuning and lazy init are
    excluded;
  * torch.cuda.synchronize() around every timed region, since CUDA calls are
    asynchronous and wall-clock without sync measures queueing, not compute;
  * repeated trials, reporting the median (robust to scheduler hiccups) with
    the interquartile range.

Scope: these are single-GPU numbers on one machine and will not transfer
unchanged to other hardware. What transfers is the ratio between
configurations, which is what a recommendation needs.

Inputs:  --img-dir and --csv (BoostNet); optionally --latents-dir for the
         downstream-head benchmark.
Outputs: <out-dir>/cost_benchmark.csv and compression_choice_table.csv; the
         joint accuracy-vs-cost table printed to stdout.

Usage:
  python cost_benchmark.py --img-dir ./medvae/data/boostnet \
      --csv ./medvae/data/boostnet_with_angles.csv \
      --latents-dir ./latents_aspect --out-dir ./outputs/cost
  python cost_benchmark.py --smoke      # synthetic-data self-test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = ["medvae_4_3_2d", "medvae_4_1_2d", "medvae_8_4_2d", "medvae_8_1_2d"]
LABEL = {"medvae_4_3_2d": "4_3", "medvae_4_1_2d": "4_1",
         "medvae_8_4_2d": "8_4", "medvae_8_1_2d": "8_1"}
NOMINAL = {"medvae_4_3_2d": 16 / 3, "medvae_4_1_2d": 16.0,
           "medvae_8_4_2d": 16.0, "medvae_8_1_2d": 64.0}
SPATIAL = {"medvae_4_3_2d": "4x", "medvae_4_1_2d": "4x",
           "medvae_8_4_2d": "8x", "medvae_8_1_2d": "8x"}

# accuracy reference values for the joint table (hard-coded from train_frontier.py and reconstruction-fidelity runs)
ACCURACY = {
    "medvae_4_3_2d": {"cobb_deg": 7.783, "landmark_norm": 0.04077, "psnr": 41.04, "ssim": 0.967},
    "medvae_4_1_2d": {"cobb_deg": 7.678, "landmark_norm": 0.04199, "psnr": 32.43, "ssim": 0.673},
    "medvae_8_4_2d": {"cobb_deg": 8.550, "landmark_norm": 0.04189, "psnr": 36.42, "ssim": 0.947},
    "medvae_8_1_2d": {"cobb_deg": 8.235, "landmark_norm": 0.04175, "psnr": 29.60, "ssim": 0.722},
}


# --- Timing helpers ---
def sync(device):
    """Block until queued GPU work finishes. CUDA calls are asynchronous, so
    timing without this measures queueing rather than compute. No-op on CPU
    (and does not require torch there)."""
    if not str(device).startswith("cuda"):
        return
    import torch
    torch.cuda.synchronize()


def timed(fn, device, warmup=3, trials=10):
    """Median wall-clock seconds per call, with IQR. Warms up first and
    synchronises around each trial so async CUDA work is captured."""
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(trials):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 25)), float(np.percentile(ts, 75))


def peak_memory_mb(device):
    if not str(device).startswith("cuda"):
        return float("nan")
    import torch
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_memory(device):
    if not str(device).startswith("cuda"):
        return
    import torch
    torch.cuda.reset_peak_memory_stats()


# --- Storage accounting ---
def latent_bytes(shape, dtype_bytes=4):
    """Bytes for one latent at the given (C,H,W). float32 by default; fp16
    halves it, which is a valid deployment option."""
    return int(np.prod(shape)) * dtype_bytes


def storage_row(config, latent_shape, src_bytes_mean):
    b32 = latent_bytes(latent_shape, 4)
    b16 = latent_bytes(latent_shape, 2)
    return {
        "config": LABEL[config],
        "latent_shape": str(tuple(latent_shape)),
        "latent_elems": int(np.prod(latent_shape)),
        "latent_kb_fp32": b32 / 1024,
        "latent_kb_fp16": b16 / 1024,
        "src_kb_mean": src_bytes_mean / 1024,
        "ratio_vs_src_fp32": src_bytes_mean / b32,
        "ratio_vs_src_fp16": src_bytes_mean / b16,
        "nominal_ratio": NOMINAL[config],
    }


# --- Main benchmark ---
def run(args):
    import torch
    import torch.nn as nn
    from medvae import MVAE
    from medvae.utils.loaders import load_2d_finetune_aspect

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(args.csv)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    uuids = [str(u) for u in sub["image_uuid"]][: args.n_images]

    # mean source-file size on disk (the thing compression is measured against)
    sizes = []
    for u in uuids:
        p = Path(args.img_dir) / f"{u}.jpg"
        if p.exists():
            sizes.append(p.stat().st_size)
    src_mean = float(np.mean(sizes)) if sizes else float("nan")
    print(f"{len(uuids)} images; mean source JPG {src_mean/1024:.1f} KB\n")

    rows = []
    for config in args.configs:
        print(f"--- {LABEL[config]} ---")
        mvae = MVAE(model_name=config, modality="xray").to(device).eval()

        x1 = load_2d_finetune_aspect(
            str(Path(args.img_dir) / f"{uuids[0]}.jpg"), merge_channels=True
        ).unsqueeze(0).to(device)
        n_in = probe_in_channels(mvae, x1, device)
        xin = match_in_channels(x1, n_in)

        with torch.no_grad():
            z = encode(mvae, xin)
        lat_shape = tuple(z.shape[1:])
        row = storage_row(config, lat_shape, src_mean)
        print(f"  latent {lat_shape}  ->  {row['latent_kb_fp32']:.1f} KB fp32 "
              f"({row['latent_kb_fp16']:.1f} KB fp16)")

        # --- Encode latency / throughput ---
        reset_memory(device)

        def _encode():
            with torch.no_grad():
                encode(mvae, xin)

        med, q1, q3 = timed(_encode, device, args.warmup, args.trials)
        row["encode_ms"] = med * 1000
        row["encode_ms_q1"] = q1 * 1000
        row["encode_ms_q3"] = q3 * 1000
        row["encode_imgs_per_s"] = 1.0 / med
        row["encode_peak_mem_mb"] = peak_memory_mb(device)
        print(f"  encode {row['encode_ms']:.1f} ms/img "
              f"({row['encode_imgs_per_s']:.1f} img/s), "
              f"peak mem {row['encode_peak_mem_mb']:.0f} MB")

        # --- Batched encode (deployment-realistic throughput) ---
        if args.batch > 1:
            xb = xin.repeat(args.batch, 1, 1, 1)
            reset_memory(device)

            def _encode_batch():
                with torch.no_grad():
                    encode(mvae, xb)

            medb, _, _ = timed(_encode_batch, device, max(1, args.warmup // 2),
                               max(3, args.trials // 2))
            row["encode_batch_imgs_per_s"] = args.batch / medb
            row["encode_batch_peak_mem_mb"] = peak_memory_mb(device)
            print(f"  batch {args.batch}: {row['encode_batch_imgs_per_s']:.1f} img/s, "
                  f"peak mem {row['encode_batch_peak_mem_mb']:.0f} MB")

        del mvae
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Downstream head cost (the cost that scales in deployment) ---
        if args.latents_dir:
            f = Path(args.latents_dir) / f"latents_{config}_train.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                lat = torch.as_tensor(d["latents"][: args.batch], dtype=torch.float32,
                                      device=device)
                head = SmallHead(lat.shape[1], 3).to(device)
                opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
                tgt = torch.randn(lat.shape[0], 3, device=device)

                def _train_step():
                    opt.zero_grad()
                    loss = nn.functional.smooth_l1_loss(head(lat), tgt)
                    loss.backward()
                    opt.step()

                reset_memory(device)
                medt, _, _ = timed(_train_step, device, args.warmup, args.trials)
                row["head_train_ms_per_batch"] = medt * 1000
                row["head_train_peak_mem_mb"] = peak_memory_mb(device)

                head.eval()

                def _infer():
                    with torch.no_grad():
                        head(lat)

                medi, _, _ = timed(_infer, device, args.warmup, args.trials)
                row["head_infer_ms_per_batch"] = medi * 1000
                row["head_params"] = sum(p.numel() for p in head.parameters())
                print(f"  head: train {row['head_train_ms_per_batch']:.2f} ms/batch, "
                      f"infer {row['head_infer_ms_per_batch']:.2f} ms/batch, "
                      f"{row['head_params']/1e3:.0f}k params")
                del head, opt, lat
                if device == "cuda":
                    torch.cuda.empty_cache()

        rows.append(row)
        print()

    cost = pd.DataFrame(rows)
    cost.to_csv(args.out_dir / "cost_benchmark.csv", index=False)
    joint_table(cost, args.out_dir, args.configs)


class SmallHead(object):
    """Stand-in for train_frontier.py's downstream head, with the same shape of
    computation, so the relative cost across configs is representative."""
    def __new__(cls, in_ch, out_dim):
        import torch.nn as nn

        class _H(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                )
                self.pool = nn.AdaptiveAvgPool2d(3)
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 9, 256),
                                          nn.GELU(), nn.Linear(256, out_dim))

            def forward(self, x):
                return self.head(self.pool(self.body(x)))

        return _H()


# --- Joint accuracy-vs-cost table + Pareto front ---
def pareto_front(points, minimise):
    """points: list of tuples; minimise: list of bools (True = lower better).
    Returns indices of non-dominated points."""
    n = len(points)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            better_or_equal = all(
                (points[j][k] <= points[i][k]) if minimise[k] else (points[j][k] >= points[i][k])
                for k in range(len(minimise))
            )
            strictly_better = any(
                (points[j][k] < points[i][k]) if minimise[k] else (points[j][k] > points[i][k])
                for k in range(len(minimise))
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def joint_table(cost, out_dir, configs):
    print("=" * 78)
    print("Compression-choice table: accuracy vs cost")
    print("=" * 78)
    hdr = (f"{'config':<7}{'spatial':>8}{'KB':>8}{'ratio':>8}{'ms':>7}{'img/s':>8}"
           f"{'Cobb°':>8}{'landmk':>9}{'PSNR':>7}{'SSIM':>7}")
    print(hdr); print("-" * len(hdr))
    recs = []
    for c in configs:
        r = cost[cost.config == LABEL[c]]
        if r.empty:
            continue
        r = r.iloc[0]
        a = ACCURACY.get(c, {})
        print(f"{LABEL[c]:<7}{SPATIAL[c]:>8}{r['latent_kb_fp32']:>8.1f}"
              f"{r['ratio_vs_src_fp32']:>7.1f}x{r['encode_ms']:>7.1f}"
              f"{r['encode_imgs_per_s']:>8.1f}"
              f"{a.get('cobb_deg', np.nan):>8.2f}{a.get('landmark_norm', np.nan):>9.4f}"
              f"{a.get('psnr', np.nan):>7.1f}{a.get('ssim', np.nan):>7.3f}")
        recs.append({"config": LABEL[c], "kb": r["latent_kb_fp32"],
                     "cobb": a.get("cobb_deg", np.nan),
                     "landmark": a.get("landmark_norm", np.nan),
                     "encode_ms": r["encode_ms"]})

    rdf = pd.DataFrame(recs)
    rdf.to_csv(out_dir / "compression_choice_table.csv", index=False)

    # --- Per-task recommendation ---
    print("\n--- Pareto-optimal configs (storage vs task error) ---")
    for task, col in [("Cobb angle", "cobb"), ("Landmark localisation", "landmark")]:
        sub = rdf.dropna(subset=[col])
        if len(sub) < 2:
            continue
        pts = list(zip(sub["kb"].values, sub[col].values))
        idx = pareto_front(pts, minimise=[True, True])
        names = [sub.iloc[i]["config"] for i in idx]
        best = sub.loc[sub[col].idxmin()]
        cheap = sub.loc[sub["kb"].idxmin()]
        print(f"\n{task}:")
        print(f"   Pareto set        : {', '.join(sorted(names))}")
        print(f"   best accuracy     : {best['config']} "
              f"({best[col]:.4g} at {best['kb']:.1f} KB)")
        print(f"   smallest latent   : {cheap['config']} "
              f"({cheap[col]:.4g} at {cheap['kb']:.1f} KB)")
        pen = 100 * (cheap[col] - best[col]) / best[col]
        print(f"   cost of going smallest: {pen:+.1f}% error for "
              f"{best['kb']/cheap['kb']:.1f}x less storage")

    print("\nAbsolute timings are machine-specific; the ratios between")
    print("configurations are the transferable quantity.")
    print(f"\nWrote {out_dir}/cost_benchmark.csv and compression_choice_table.csv")


# --- MedVAE I/O helpers (shared with the other scripts) ---
def match_in_channels(x, n_in):
    if x.shape[1] == n_in:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, n_in, 1, 1)
    raise ValueError(f"cannot match {x.shape[1]}ch -> {n_in}ch")


def encode(mvae, x):
    import torch
    with torch.no_grad():
        out = mvae.model.encode(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out
    for attr in ("mode", "mean", "sample"):
        if hasattr(out, attr):
            a = getattr(out, attr)
            return a() if callable(a) else a
    raise RuntimeError(f"unrecognised encode() return: {type(out)}")


def probe_in_channels(mvae, x1, device):
    for c in (1, 3, 4):
        try:
            encode(mvae, match_in_channels(x1, c).to(device))
            return c
        except Exception:
            continue
    raise RuntimeError("could not determine encoder input channels")


# --- Synthetic-data self-test ---
def smoke():
    """Validates the cost logic that does not need MedVAE: byte accounting,
    timing behaviour, and Pareto selection."""
    # --- storage accounting against known values ---
    assert latent_bytes((1, 96, 96), 4) == 96 * 96 * 4
    assert latent_bytes((4, 48, 48), 4) == 4 * 48 * 48 * 4
    r = storage_row("medvae_4_1_2d", (1, 96, 96), src_bytes_mean=500 * 1024)
    print(f"storage: (1,96,96) fp32 -> {r['latent_kb_fp32']:.1f} KB "
          f"(expect {96*96*4/1024:.1f}), fp16 {r['latent_kb_fp16']:.1f} KB")
    print(f"         vs 500 KB source -> {r['ratio_vs_src_fp32']:.1f}x")

    # 8_4 has 4*48*48 = 9216 elems vs 4_1's 9216: identical (both are the 16x pair)
    e41 = latent_bytes((1, 96, 96)); e84 = latent_bytes((4, 48, 48))
    print(f"4_1 latent elems {e41//4} vs 8_4 {e84//4}  "
          f"-> {'identical (matched 16x pair)' if e41 == e84 else 'differ'}")

    # --- timing: a known-duration function should measure close to it ---
    med, q1, q3 = timed(lambda: time.sleep(0.01), "cpu", warmup=1, trials=5)
    print(f"timing: 10 ms sleep measured as {med*1000:.1f} ms "
          f"[{q1*1000:.1f},{q3*1000:.1f}]")

    # --- Pareto: plant a dominated point and check it is excluded ---
    pts = [(10.0, 5.0),    # cheap, accurate      -> keep
           (20.0, 4.0),    # dearer, more accurate -> keep
           (30.0, 6.0)]    # dearer and worse      -> dominated
    keep = pareto_front(pts, minimise=[True, True])
    print(f"pareto: kept {keep} of 3 (expect [0, 1]; point 2 is dominated)")

    ok = (r["latent_kb_fp32"] == 96 * 96 * 4 / 1024
          and e41 == e84
          and 0.008 < med < 0.02
          and keep == [0, 1])
    print("\nSelf-test:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--latents-dir", type=Path, default=None,
                    help="if given, also benchmarks the downstream head")
    ap.add_argument("--split", default="test")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs/cost"))
    ap.add_argument("--n-images", type=int, default=64,
                    help="images used for the source-size average")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.img_dir and args.csv:
        run(args)
    else:
        raise SystemExit("--img-dir and --csv are required (or --smoke for the synthetic-data self-test)")


if __name__ == "__main__":
    main()
