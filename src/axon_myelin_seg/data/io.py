"""Reading images and uint16 masks; writing uint16 predictions."""

from pathlib import Path

import numpy as np
import tifffile


def read_image(path: str | Path) -> np.ndarray:
    return tifffile.imread(path)


def read_mask(path: str | Path) -> np.ndarray:
    return tifffile.imread(path).astype(np.uint16)


def write_mask(path: str | Path, labels: np.ndarray) -> None:
    tifffile.imwrite(path, labels.astype(np.uint16), compression="zlib")
