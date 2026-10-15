# loader_patch_v2.py: random-crop patch loader for native-resolution MedVAE fine-tuning.
#
# Intended to be appended to MedVAE/medvae/utils/loaders.py.
#
# Unlike load_2d_finetune and load_2d_finetune_aspect, this loader does not
# resize the input. It opens the native-resolution image (~880x2480 for
# BoostNet) and returns a single 384x384 random crop, biased toward the spine
# bounding box so that training capacity is not spent on empty corners.
#
# The bbox argument is normalised [0, 1] coordinates of the spine region,
# precomputed by prepare_spine_bboxes.py.

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
        top-left corner is sampled uniformly within this bbox; the resulting
        384x384 patch is then guaranteed to contain at least the top-left
        portion of the spine, and on average will be well-centred over it.
        If the image is smaller than the patch in either dim, the loader falls
        back to the largest valid crop and zero-pads it at the bottom/right to
        (patch_size, patch_size).
    flip_p : float
        Probability of horizontal flip. Cobb angles are reported as magnitudes,
        so flipping does not change the labels.

    Returns
    -------
    torch.Tensor of shape (1, patch_size, patch_size) in [-1, 1].

    Notes on the design
    -------------------
    - The top-left corner is sampled uniformly from the bbox region (rather
      than centring the patch on the bbox centre) so that successive epochs
      see different parts of the same image. The random crop acts as data
      augmentation at no extra cost.
    - The corner is clamped so that (corner + patch_size) <= image dim, so
      the crop never goes off the edge.
    - Per-image ScaleIntensity is applied after cropping. This means each
      patch has its own dynamic range stretched to fill [0, 1]. For a patch
      that is mostly soft tissue (low contrast) this exaggerates noise; for
      a patch over a vertebral body (high contrast) it preserves detail.
      This matches the normalisation of the rest of the pipeline;
      reconstruction quality on low-contrast patches may be affected.
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

        # Crop. If the image is smaller than the patch, the crop is smaller too
        # and is then padded with zeros to (patch_size, patch_size).
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

        # ScaleIntensity per-patch to [0, 1], then normalise to [-1, 1].
        # Done manually rather than via MONAI Compose, to avoid MONAI's
        # lazy-execution behaviour and because applying these to an existing
        # tensor is straightforward.
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
