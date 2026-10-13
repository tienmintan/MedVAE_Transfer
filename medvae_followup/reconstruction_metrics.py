"""Explicit image metrics and regional/spectral diagnostics; CPU analysis only."""
import numpy as np
from scipy.ndimage import convolve1d, distance_transform_edt

from common import require


def bbox(mask):
    y, x = np.where(mask)
    require(len(y) > 0, "Empty content mask")
    return int(y.min()), int(y.max()+1), int(x.min()), int(x.max()+1)


def crop(a, box):
    y0, y1, x0, x1 = box
    return a[y0:y1, x0:x1]


def to01(a):
    return np.clip((np.asarray(a, dtype=np.float64) + 1) / 2, 0, 1)


def ssim_zero_map(a, b, sigma=1.5, size=11):
    """Same Gaussian/population formula as the old implementation, in float64."""
    require(size % 2 == 1 and min(a.shape) >= size, "SSIM crop too small")
    x = np.arange(size) - size//2; g = np.exp(-x*x/(2*sigma*sigma)); g /= g.sum()
    def filt(z):
        return convolve1d(convolve1d(z, g, axis=0, mode="constant", cval=0), g, axis=1, mode="constant", cval=0)
    ma, mb = filt(a), filt(b)
    va, vb, ab = filt(a*a)-ma*ma, filt(b*b)-mb*mb, filt(a*b)-ma*mb
    return ((2*ma*mb+.01**2)*(2*ab+.03**2)) / ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))


def fidelity(a, b, resolution):
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 2 and np.isfinite(a).all() and np.isfinite(b).all(), "Invalid images")
    require(a.min() >= 0 and a.max() <= 1 and b.min() >= 0 and b.max() <= 1, "Fidelity range must be [0,1]")
    mse = float(np.mean((a-b)**2))
    psnr = float(peak_signal_noise_ratio(a, b, data_range=1)) if mse > 0 else None
    manual_psnr = float(-10*np.log10(mse)) if mse > 0 else None
    require(mse == 0 or abs(psnr-manual_psnr) < 1e-10, "PSNR reference mismatch")
    ref, refmap = structural_similarity(a, b, data_range=1, gaussian_weights=True, sigma=1.5,
        win_size=11, use_sample_covariance=False, full=True)
    legacy = ssim_zero_map(a, b)
    inner = (slice(5, -5), slice(5, -5))
    delta = float(np.max(np.abs(legacy[inner]-refmap[inner])))
    require(delta < 1e-8, f"SSIM interior reference check failed: {delta:g}")
    sigma_scaled = 1.5 * resolution / 384
    scaled_size = 2*int(3.5*sigma_scaled+.5)+1
    require(min(a.shape) >= scaled_size, "Content crop too small for scale-matched SSIM")
    scaled = structural_similarity(a, b, data_range=1, gaussian_weights=True, sigma=sigma_scaled,
        win_size=scaled_size, use_sample_covariance=False)
    return dict(mse=mse, psnr_db=psnr, exact_match=mse == 0,
        ssim_legacy_zero_border=float(legacy.mean()), ssim_reference=float(ref),
        ssim_scale_matched=float(scaled), ssim_scaled_sigma=sigma_scaled, ssim_scaled_window=scaled_size,
        ssim_interior_max_abs=delta)


def map_points(lm, width, height, resolution, content_box=None):
    p = np.stack((lm[:68], lm[68:]), axis=1).astype(np.float64)
    if content_box is not None:
        y0,y1,x0,x1 = content_box
        return p * [x1-x0,y1-y0] + [x0,y0]
    side = max(width, height)
    pad = [(side-width)//2, (side-height)//2]
    return (p * [width, height] + pad) * (resolution/side)


def masks(points, shape, content, resolution):
    """Same rasterisation/overlap rule as the original; radii scale with canvas."""
    h,w = shape
    ep, ct = np.ones(shape, bool), np.ones(shape, bool)
    for quad in points.reshape(17,4,2):
        for a,b in ((quad[0],quad[1]),(quad[2],quad[3])):
            steps = int(max(2, np.linalg.norm(b-a)))
            q = np.rint(a + np.linspace(0,1,steps)[:,None]*(b-a)).astype(int)
            valid = (q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
            ep[q[valid,1],q[valid,0]] = False
        x,y = np.rint(quad.mean(0)).astype(int)
        if 0 <= x < w and 0 <= y < h:
            ct[y,x] = False
    require((~ep).any() and (~ct).any(), "Landmarks lie outside the canvas")
    e = (distance_transform_edt(ep) <= 2.5*resolution/384) & content
    c = (distance_transform_edt(ct) <= 4.*resolution/384) & content & ~e
    require(e.any() and c.any(), "Empty endplate/centroid region; inspect geometry instead of dropping this image")
    return e,c


def regional(original, reconstruction, lm, width, height, resolution, mapping):
    content = original > -.8
    box = bbox(content)
    pts = map_points(lm,width,height,resolution,box if mapping == "legacy_content_box" else None)
    e,c = masks(pts, original.shape, content, resolution)
    err = (original-np.clip(reconstruction,-1,1))**2
    me, mc = float(err[e].mean()), float(err[c].mean())
    return dict(mapping=mapping, endplate_mse=me, centroid_mse=mc, regional_difference=me-mc,
        regional_ratio=me/mc if mc>0 else None, endplate_pixels=int(e.sum()), centroid_pixels=int(c.sum()),
        band_half_width=2.5*resolution/384, centroid_radius=4.*resolution/384), e, c, pts


def spectrum(original, reconstructed, resolution):
    """Keep numerator/denominator powers, rather than only unstable ratios."""
    box = bbox(original > -.8)
    a, b = crop(original,box), crop(np.clip(reconstructed,-1,1),box)
    h,w = a.shape
    window = np.hanning(h)[:,None]*np.hanning(w)[None,:]
    norm = h*w*float(np.sum(window*window))
    require(norm > 0, "Spectral crop too small")
    signal = abs(np.fft.fftshift(np.fft.fft2(a*window)))**2
    residual = abs(np.fft.fftshift(np.fft.fft2((a-b)*window)))**2
    yy,xx = np.indices((h,w))
    radius = np.clip(np.hypot((yy-h/2)/(h/2),(xx-w/2)/(w/2))/np.sqrt(2),0,1)
    legacy_index = np.minimum((radius*64).astype(int),63)
    fy = np.fft.fftshift(np.fft.fftfreq(h))*resolution
    fx = np.fft.fftshift(np.fft.fftfreq(w))*resolution
    common_radius = np.hypot(fy[:,None],fx[None,:])
    common_index = np.floor(common_radius/3).astype(int)
    rows=[]
    for axis,index in (("legacy_corner_normalised",legacy_index),("cycles_per_canvas",common_index)):
        for i in range(64):
            select = index == i; n = int(select.sum())
            s,r = float(signal[select].sum()),float(residual[select].sum())
            sp,rp = s/max(1,n),r/max(1,n)
            rows.append(dict(axis=axis, bin=i, lower=i/64 if axis.startswith("legacy") else i*3,
                upper=(i+1)/64 if axis.startswith("legacy") else (i+1)*3, count=n,
                signal_sum=s, residual_sum=r, signal_mean=sp, residual_mean=rp,
                signal_energy=s/norm, residual_energy=r/norm,
                ratio_legacy_floor=rp/max(sp,1e-12) if n else None,
                floor_applied=bool(n and sp<1e-12), ratio_without_floor=r/s if s>0 else None))
    # Equal physical frequency bands at both resolutions: cycles across the padded canvas.
    bands=[]
    for lo,hi in ((0,64),(64,128),(128,192)):
        select = (common_radius>=lo)&(common_radius<hi)
        s,r = float(signal[select].sum()/norm),float(residual[select].sum()/norm)
        bands.append(dict(band=f"{lo}-{hi}_cycles_per_canvas", signal_energy=s, residual_energy=r,
            ratio=r/s if s>0 else None, coefficients=int(select.sum())))
    return rows,bands
