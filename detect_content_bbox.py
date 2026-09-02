"""
detect_content_bbox.py

Utility for detecting the bounding box of "content" (non-padding) regions
in a preprocessed grayscale image. Used by extract_latents_*_cropped.py to
crop out aspect-padding before SD-VAE encoding.

The detection logic:
  1. Normalise the input image to [0, 1].
  2. Threshold at the lower of (5% of the dynamic range) or (a small
     absolute floor at 0.02), to catch faint tissue against black
     background.
  3. Apply a small morphological opening to suppress isolated bright
     noise (DICOM artifacts, edge markers, etc).
  4. Take the bounding box of the remaining "True" pixels.
  5. If the bbox is degenerate (too small, e.g., the image is mostly
     black), fall back to using the full image.

Returns: (y_min, x_min, y_max, x_max) tuple, all in pixel coordinates,
inclusive of the bounds. y_max and x_max are exclusive (i.e., suitable
for array slicing as img[y_min:y_max, x_min:x_max]).
"""

import numpy as np
from scipy import ndimage


def detect_content_bbox(
        img: np.ndarray,
        threshold_pct: float = 0.05,
        absolute_floor: float = 0.02,
        morph_kernel_size: int = 5,
        min_content_fraction: float = 0.05,
) -> tuple:
    """Detect the bounding box of non-padding content in a 2D grayscale image.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image. Any value range; will be min-max normalised
        internally for thresholding.
    threshold_pct : float
        Fraction of the image's dynamic range above which a pixel is
        considered "content". Default 0.05 = 5%.
    absolute_floor : float
        Minimum threshold in normalised [0, 1] units. Prevents the
        threshold from being too low when the image has very high contrast.
    morph_kernel_size : int
        Size of the morphological opening kernel (in pixels). Suppresses
        isolated noise/artifacts.
    min_content_fraction : float
        If the detected content region is smaller than this fraction of
        the total image, the image is considered "mostly empty" and we
        return the full bbox (fallback).

    Returns
    -------
    (y_min, x_min, y_max, x_max) : tuple of int
        Bbox coordinates with y_max, x_max exclusive (suitable for slicing).
    """
    H, W = img.shape
    img = img.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-6:
        # Degenerate: all-uniform image
        return (0, 0, H, W)

    # Normalise to [0, 1] for thresholding
    norm = (img - img_min) / (img_max - img_min)

    # Threshold: maximum of percentile-based and absolute floor
    threshold = max(threshold_pct, absolute_floor)
    mask = norm > threshold

    # Morphological opening to suppress isolated bright noise
    if morph_kernel_size > 1:
        struct = np.ones((morph_kernel_size, morph_kernel_size), dtype=bool)
        mask = ndimage.binary_opening(mask, structure=struct)

    # Bounding box of remaining True pixels
    if not mask.any():
        return (0, 0, H, W)

    rows_any = mask.any(axis=1)
    cols_any = mask.any(axis=0)
    y_min, y_max = int(np.argmax(rows_any)), int(H - np.argmax(rows_any[::-1]))
    x_min, x_max = int(np.argmax(cols_any)), int(W - np.argmax(cols_any[::-1]))

    # Sanity check: is the bbox big enough?
    content_area = (y_max - y_min) * (x_max - x_min)
    total_area = H * W
    if content_area / total_area < min_content_fraction:
        # Too small — image is mostly noise, fall back to full
        return (0, 0, H, W)

    return (y_min, x_min, y_max, x_max)


def crop_and_repad_to_aspect(
        img: np.ndarray,
        target_aspect: float = 1.5,
        target_size: int = 512,
        bbox: tuple = None,
) -> np.ndarray:
    """Crop an image to its content bbox, then resize to a target aspect
    ratio and pad to a square output of side target_size.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image in any range.
    target_aspect : float
        Desired height-to-width aspect ratio of the content after cropping.
        E.g. 1.5 means content is 50% taller than wide. Wider than BoostNet's
        native 2.8, taller than VinDr's typical ~1.0 — chosen as a
        symmetric compromise.
    target_size : int
        Side length of the output square image (after padding).
    bbox : tuple or None
        Pre-computed content bbox (y_min, x_min, y_max, x_max). If None,
        detect via detect_content_bbox().

    Returns
    -------
    out : np.ndarray
        2D grayscale image of shape (target_size, target_size). Content
        occupies the centre, with minimal symmetric padding.
    """
    from PIL import Image as PILImage

    if bbox is None:
        bbox = detect_content_bbox(img)
    y_min, x_min, y_max, x_max = bbox

    # Crop to content
    content = img[y_min:y_max, x_min:x_max]
    ch, cw = content.shape
    if ch == 0 or cw == 0:
        # Degenerate fallback: use the full image
        content = img
        ch, cw = content.shape
    content_aspect = ch / cw

    # Bring content to target_aspect by extending the shorter side
    # (we extend by padding with black, not by stretching)
    if content_aspect > target_aspect:
        # Content is taller than target — extend width
        new_w = int(round(ch / target_aspect))
        pad_w = (new_w - cw) // 2
        new_content = np.zeros((ch, new_w), dtype=content.dtype)
        new_content[:, pad_w:pad_w + cw] = content
        content = new_content
    elif content_aspect < target_aspect:
        # Content is wider than target — extend height
        new_h = int(round(cw * target_aspect))
        pad_h = (new_h - ch) // 2
        new_content = np.zeros((new_h, cw), dtype=content.dtype)
        new_content[pad_h:pad_h + ch, :] = content
        content = new_content
    # Else: aspect matches exactly, no padding needed

    # Resize to fit inside target_size x target_size square
    # The content currently has aspect target_aspect (taller than wide).
    # Fit by height, then centre horizontally.
    new_h_to_target = target_size
    new_w_to_target = int(round(target_size / target_aspect))

    pil = PILImage.fromarray(content.astype(np.float32))
    pil = pil.resize((new_w_to_target, new_h_to_target),
                       PILImage.BICUBIC)
    resized = np.array(pil)

    # Pad horizontally to target_size
    pad_left = (target_size - new_w_to_target) // 2
    out = np.zeros((target_size, target_size), dtype=resized.dtype)
    out[:, pad_left:pad_left + new_w_to_target] = resized

    return out


# ---------------------------------------------------------------------------
# Quick sanity check (run as: python detect_content_bbox.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthesise a test image: 200×500 black with content in the middle
    test = np.zeros((200, 500), dtype=np.float32)
    test[50:150, 100:400] = 0.6  # Centred bright rectangle
    test[50:150, 100:400] += 0.2 * np.random.randn(100, 300)

    bbox = detect_content_bbox(test)
    print(f"Test input shape: {test.shape}")
    print(f"Detected bbox: {bbox}")
    print(f"Expected bbox: (50, 100, 150, 400)")
    cropped = crop_and_repad_to_aspect(test, target_aspect=1.5,
                                          target_size=512)
    print(f"Output shape after crop+repad: {cropped.shape}")
    print(f"Output content fraction: "
          f"{(cropped > 0).sum() / cropped.size * 100:.1f}%")
