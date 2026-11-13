"""Reading images and uint16 masks; writing uint16 predictions (T-01).

The challenge release has no metadata files: the filename (``sample_XXXX.tif``) is the only key (docs/notes/02-challenge-and-data.md §2).
"""

from pathlib import Path

import numpy as np
import tifffile

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "train-dataset"
IMAGE_DIR = "train_samples"
MASK_DIR = "train_instance_labels"
SHAPE = (3008, 4096)  # H, W


def read_image(path: str | Path) -> np.ndarray:
    return tifffile.imread(path)


def read_mask(path: str | Path) -> np.ndarray:
    return tifffile.imread(path).astype(np.uint16)


def write_mask(path: str | Path, labels: np.ndarray) -> None:
    tifffile.imwrite(path, labels.astype(np.uint16), compression="zlib")


def list_samples(root: str | Path = DATA_ROOT) -> list[str]:
    """Sorted sample names (``sample_0001``, ...) that have both an image and a mask."""
    root = Path(root)
    imgs = {p.stem for p in (root / IMAGE_DIR).glob("sample_*.tif")}
    masks = {p.stem for p in (root / MASK_DIR).glob("sample_*.tif")}
    return sorted(imgs & masks)


def load_sample(name: str, root: str | Path = DATA_ROOT, validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (image uint8 H×W, mask uint16 H×W) for ``name`` (with or without ``.tif``)."""
    root, stem = Path(root), Path(name).stem
    img = read_image(root / IMAGE_DIR / f"{stem}.tif")
    mask = read_mask(root / MASK_DIR / f"{stem}.tif")
    if validate:
        if img.shape != SHAPE or mask.shape != SHAPE:
            raise ValueError(f"{stem}: shape image {img.shape}, mask {mask.shape}; expected {SHAPE}")
        if img.dtype != np.uint8:
            raise ValueError(f"{stem}: image dtype {img.dtype}, expected uint8")
    return img, mask
