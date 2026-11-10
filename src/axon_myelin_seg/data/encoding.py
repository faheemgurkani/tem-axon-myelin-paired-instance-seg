"""Challenge label encoding: 0 background, 1 artifact (ignored), even X axon, X+1 its myelin."""

import numpy as np

BACKGROUND = 0
ARTIFACT = 1


def axon_mask(labels: np.ndarray) -> np.ndarray:
    return (labels >= 2) & (labels % 2 == 0)


def myelin_mask(labels: np.ndarray) -> np.ndarray:
    return (labels >= 3) & (labels % 2 == 1)


def ignore_mask(labels: np.ndarray) -> np.ndarray:
    return labels == ARTIFACT


def fiber_ids(labels: np.ndarray) -> np.ndarray:
    """Even axon IDs X that have a paired myelin X+1."""
    present = set(np.unique(labels).tolist())
    return np.array(sorted(x for x in present if x >= 2 and x % 2 == 0 and x + 1 in present), dtype=labels.dtype)


def unpaired_ids(labels: np.ndarray) -> dict[str, list[int]]:
    """IDs violating the (X, X+1) pairing rule."""
    present = set(np.unique(labels).tolist())
    return {
        "axon_without_myelin": sorted(x for x in present if x >= 2 and x % 2 == 0 and x + 1 not in present),
        "myelin_without_axon": sorted(x for x in present if x >= 3 and x % 2 == 1 and x - 1 not in present),
    }


def encode_pairs(axon_instances: np.ndarray, myelin_instances: np.ndarray) -> np.ndarray:
    """Merge per-fiber axon/myelin instance maps that share fiber index k >= 1 into the (2k, 2k+1) uint16 encoding."""
    out = np.zeros(axon_instances.shape, dtype=np.uint16)
    out[myelin_instances > 0] = 2 * myelin_instances[myelin_instances > 0] + 1
    out[axon_instances > 0] = 2 * axon_instances[axon_instances > 0]
    return out
