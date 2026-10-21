from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    Orientation,
    ScaleIntensity,
    CropForeground,
    ScaleIntensityRange,
    RandSpatialCrop,
)
import torch
import torch.nn.functional as F
from monai.transforms import Transform
import torchvision
from PIL import Image
import polars as pl
import numpy as np


class MonaiNormalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return torchvision.transforms.Normalize(self.mean, self.std)(img)


class MonaiPad(Transform):
    def __init__(self, size, mode="constant", value=0):
        self.size = size
        self.mode = mode
        self.value = value

    def __call__(self, img):
        padding = []
        for i in range(len(img.shape) - 1, 0, -1):  # Start from the last dimension
            total_pad = max(self.size[i - 1] - img.shape[i], 0)
            padding.extend([total_pad // 2, total_pad - total_pad // 2])
        padded_img = F.pad(img, padding, mode=self.mode, value=self.value)
        return padded_img


class MonaiImageOpen(Transform):
    def __init__(self):
        pass

    def __call__(self, path):
        return Image.open(path)


"""
Custom transform to normalize and pad 2D images
@input: path to image (str)
@Output: padding (np.array)
"""


def load_2d(
    path: str,
    merge_channels: bool = False,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()

        if merge_channels:
            img = img.mean(0, keepdim=True)

        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


def load_2d_finetune(
    path: str, dtype: torch.dtype = torch.float32, merge_channels: bool = True, **kwargs
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((384, 384), interpolation=3, antialias=True),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()
        if merge_channels:
            img = img.mean(0, keepdim=True)
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_mri_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            CropForeground(k_divisible=[16, 16, 16]),
        ],
        lazy=True,
    )

    try:
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 128, 128, 128))


"""
Custom transform to normalize, crop, and pad 3D CT volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_ct_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            CropForeground(k_divisible=[16, 16, 16]),
        ],
        lazy=True,
    )

    try:
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 256, 256, 256))


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_mri_3d_finetune(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            MonaiPad(size=[64, 64, 64], value=-1),
            RandSpatialCrop(roi_size=[64, 64, 64]),
        ],
        lazy=True,
    )

    try:
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 64, 64, 64))


"""
Custom transform to normalize, crop, and pad ct 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_ct_3d_finetune(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            MonaiPad(size=[64, 64, 64], value=-1),
            RandSpatialCrop(roi_size=[64, 64, 64]),
        ],
        lazy=True,
    )

    try:
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 64, 64, 64))


def load_labels(
    df: pl.DataFrame,
    dtype: np.dtype = None,
    # fill_null=None,
    fill_nan: float = None,
    squeeze: int = None,
) -> torch.Tensor:
    """Load the labels from a dataframe."""
    # BUG: Polars hangs when trying to convert to numpy in a DataLoader
    x = df.to_pandas().to_numpy()
    if dtype is not None:
        x = x.astype(dtype)

    if isinstance(squeeze, int):
        out = torch.from_numpy(x).squeeze(dim=squeeze)
    else:
        out = torch.from_numpy(x).squeeze()

    if isinstance(fill_nan, float):
        out = torch.where(out.isnan(), fill_nan, out)

    return out
# =============================================================================
# Aspect-ratio-preserving loader for tall/wide medical images
# =============================================================================
#
# Add this function to MedVAE/medvae/utils/loaders.py, just below the existing
# load_2d_finetune function. It reuses MonaiPad (already defined in that file)
# to pad to a square BEFORE resizing, which preserves aspect ratio.
#
# For BoostNet (~880x2480), this means:
#   load_2d_finetune        : 880x2480 -> 384x384  (vertical squash ~3.2x)
#   load_2d_finetune_aspect : 880x2480 -> 2480x2480 (zero-padded sides) -> 384x384
#                             (no squash; vertebrae now span ~more pixels)
# =============================================================================

def load_2d_finetune_aspect(
    path: str,
    dtype: torch.dtype = torch.float32,
    merge_channels: bool = True,
    target_size: int = 384,
    **kwargs,
):
    """Drop-in replacement for load_2d_finetune that preserves aspect ratio.

    Pipeline:
        open -> tensor -> pad to square (zero pad on shorter side) ->
        resize to (target_size, target_size) -> intensity scale -> normalize.
    """
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            # Note: MonaiPad reads img.shape[1:] (H, W). We pad to a square
            # whose side = max(H, W). At 880x2480 the result is 2480x2480.
            _PadToSquare(),
            torchvision.transforms.Resize(
                (target_size, target_size), interpolation=3, antialias=True
            ),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
#        lazy=True,
    )
    try:
        img = img_transforms(path).as_tensor()
        if merge_channels:
            img = img.mean(0, keepdim=True)
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, target_size, target_size))


class _PadToSquare(Transform):
    """Zero-pad a (C, H, W) tensor so that H == W == max(H, W).

    Uses centred padding (equal on both sides where possible).
    """
    def __call__(self, img):
        _, h, w = img.shape
        side = max(h, w)
        pad_h_total = side - h
        pad_w_total = side - w
        # F.pad order for 2D: (left, right, top, bottom)
        padding = (
            pad_w_total // 2, pad_w_total - pad_w_total // 2,
            pad_h_total // 2, pad_h_total - pad_h_total // 2,
        )
        return F.pad(img, padding, mode="constant", value=0)
# =============================================================================
# Patch loader for native-resolution training (Phase 1 of patch-based pipeline)
# =============================================================================
#
# Append this to MedVAE/medvae/utils/loaders.py.
#
# Unlike load_2d_finetune and load_2d_finetune_aspect, this loader does NOT
# resize the input. It opens the native-resolution image (~880x2480 for
# BoostNet) and returns a single 384x384 random crop, biased toward the spine
# bounding box so we don't waste capacity on empty corners.
#
# The bbox argument is normalised [0, 1] coordinates of the spine region,
# precomputed by prepare_spine_bboxes.py.
# =============================================================================

import random


def load_2d_finetune_patch(
    path: str,
    bbox: tuple = (0.0, 0.0, 1.0, 1.0),
    dtype: torch.dtype = torch.float32,
    merge_channels: bool = True,
    patch_size: int = 384,
    flip_p: float = 0.5,
    **kwargs,
):
    """Open a native-resolution image and return one random 384x384 crop
    biased toward the spine bounding box.

    Parameters
    ----------
    path : str
        Path to the image file.
    bbox : (x0, y0, x1, y1)
        Normalised [0, 1] coordinates of the spine bounding box. The crop's
        TOP-LEFT corner is sampled uniformly within this bbox; the resulting
        384x384 patch is then guaranteed to contain at least the top-left
        portion of the spine, and on average will be well-centred over it.
        If the image is smaller than the patch in either dim, we fall back
        to the largest valid crop and pad-center to (patch_size, patch_size).
    flip_p : float
        Probability of horizontal flip. Cobb angles are reported as magnitudes,
        so flipping doesn't break labels.

    Returns
    -------
    torch.Tensor of shape (1, patch_size, patch_size) in [-1, 1].

    Notes on the design
    -------------------
    - We sample the TOP-LEFT corner uniformly from the bbox region (rather
      than centring the patch on the bbox centre) so that successive epochs
      see different parts of the same image. Random crop = data augmentation
      effectively for free.
    - We clamp the corner so that (corner + patch_size) <= image dim, so the
      crop never goes off the edge.
    - Per-image ScaleIntensity is applied AFTER cropping. This means each
      patch has its own dynamic range stretched to fill [0, 1]. For a patch
      that's mostly soft tissue (low contrast) this exaggerates noise; for
      a patch over a vertebral body (high contrast) it preserves detail.
      This matches the existing pipeline; revisit if reconstruction quality
      looks weird on low-contrast patches.
    """
    try:
        # Open at native resolution
        pil = Image.open(path)
        W, H = pil.size                # PIL uses (W, H)

        # Convert to tensor: (C, H, W) in [0, 1]
        img = torchvision.transforms.functional.to_tensor(pil)
        if img.shape[0] > 1:
            img = img.mean(0, keepdim=True)  # collapse to single channel

        # Compute bbox in pixel coords
        x0, y0, x1, y1 = bbox
        bx0 = int(round(x0 * W));  by0 = int(round(y0 * H))
        bx1 = int(round(x1 * W));  by1 = int(round(y1 * H))

        # Random top-left corner: uniformly within the bbox, but constrained
        # so the patch fits inside the image.
        max_left = max(0, min(bx1 - 1, W - patch_size))
        max_top  = max(0, min(by1 - 1, H - patch_size))
        min_left = max(0, min(bx0,    max_left))
        min_top  = max(0, min(by0,    max_top))
        if max_left <= min_left:
            left = min_left
        else:
            left = random.randint(min_left, max_left)
        if max_top <= min_top:
            top = min_top
        else:
            top = random.randint(min_top, max_top)

        # Crop. If image is smaller than patch, the crop will be smaller too,
        # then we pad with zeros to (patch_size, patch_size).
        crop = img[:, top:top + patch_size, left:left + patch_size]
        ch, cH, cW = crop.shape
        if cH < patch_size or cW < patch_size:
            pad_bot = patch_size - cH
            pad_right = patch_size - cW
            crop = F.pad(crop, (0, pad_right, 0, pad_bot),
                         mode="constant", value=0)

        # Optional horizontal flip
        if flip_p > 0 and random.random() < flip_p:
            crop = torch.flip(crop, dims=[-1])

        # ScaleIntensity per-patch to [0, 1], then Normalize to [-1, 1].
        # We do this manually rather than via MONAI Compose to avoid the
        # lazy-execution issues we hit before, and because applying these
        # to a tensor we already own is trivial.
        cmin, cmax = crop.min(), crop.max()
        if cmax > cmin:
            crop = (crop - cmin) / (cmax - cmin)
        else:
            crop = crop * 0
        crop = (crop - 0.5) / 0.5     # [0, 1] -> [-1, 1]

        return crop.to(dtype=dtype)

    except Exception as e:
        print(f"Error in patch loading {path} with error: {e}")
        return torch.zeros((1, patch_size, patch_size), dtype=dtype)
