"""
patch_dataset.py: BoostNetPatchDataset, a GenericDataset subclass that returns spine-biased random patches.

This file belongs in MedVAE/medvae/dataloaders/ alongside generic_dataset.py
and is registered in medvae/dataloaders/__init__.py with:
    from .patch_dataset import BoostNetPatchDataset

Why a subclass and not a fresh Dataset:
    GenericDataset already handles the required CSV -> file path resolution
    (split filtering, image directory, suffix). The only thing it cannot do
    is pass per-row arguments to the img_transform. __getitem__ is overridden
    minimally to look up the bbox from self.df and call the loader with it.
"""

from typing import Optional

from medvae.dataloaders.generic_dataset import GenericDataset
from medvae.utils.loaders import load_2d_finetune_patch


class BoostNetPatchDataset(GenericDataset):
    """Returns one random 384x384 patch per image, biased to the spine bbox.

    The CSV must contain bbox columns: bbox_x0, bbox_y0, bbox_x1, bbox_y1
    (normalised [0, 1] coordinates), as produced by prepare_spine_bboxes.py.

    Compared to GenericDataset:
      - The img_transform yaml argument is ignored and load_2d_finetune_patch
        is always used internally, since the patch loader needs a bbox
        argument that the standard transform interface cannot supply.
      - The merge_channels, patch_size, and flip_p kwargs are exposed for
        configuration via the yaml.
    """
    def __init__(
        self,
        *args,
        merge_channels: bool = True,
        patch_size: int = 384,
        flip_p: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.merge_channels = merge_channels
        self.patch_size = patch_size
        self.flip_p = flip_p

        # Validate required columns up front so failures happen at construction
        # time, not deep inside a training loop.
        cols = list(self.df.columns)   # works for both polars and pandas
        for col in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]:
            if col not in cols:
                raise ValueError(
                    f"BoostNetPatchDataset: CSV missing required column "
                    f"{col!r}. The bbox columns are populated by "
                    f"prepare_spine_bboxes.py.")

        # Pre-extract bboxes as a list of tuples for type-agnostic, fast
        # per-sample lookup. self.df may be polars (MedVAE's default) or
        # pandas; both expose to_dict / to_pandas; a small dispatch handles both.
        if hasattr(self.df, "to_pandas"):
            # polars DataFrame
            bb = self.df.select(
                ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]
            ).to_pandas()
        else:
            # pandas DataFrame
            bb = self.df[["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]]
        self.bboxes = list(zip(
            bb["bbox_x0"].tolist(),
            bb["bbox_y0"].tolist(),
            bb["bbox_x1"].tolist(),
            bb["bbox_y1"].tolist(),
        ))

    def __getitem__(self, idx: int):
        # Build the same dict GenericDataset returns, but replace the img
        # entry with one produced via the patch loader (which needs the bbox).
        sample = {"group_id": self.dataset_id}

        # Image: get path from samples["img"], look up bbox from pre-extracted list.
        img_path = self.samples["img"][idx]
        bbox = self.bboxes[idx]
        sample["img"] = load_2d_finetune_patch(
            str(img_path),
            bbox=bbox,
            merge_channels=self.merge_channels,
            patch_size=self.patch_size,
            flip_p=self.flip_p,
        )

        # Labels (none for reconstruction training but kept for compatibility)
        if "lbl" in self.samples:
            sample["lbl"] = self.samples["lbl"][idx]
            if callable(self.lbl_transform):
                sample["lbl"] = self.lbl_transform(sample["lbl"])

        # Common-transform hook from parent
        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample
