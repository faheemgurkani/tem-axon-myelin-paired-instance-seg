"""One call that scores a prediction with everything we report: official metrics (T-04), diagnostics (T-05) and
the error taxonomy (T-06). Also exposes the per-fiber class tables for the oracle experiments (T-13)."""

import numpy as np

from .diagnostics import diagnose_image
from .official import evaluate_image
from .taxonomy import classify_errors, ring_candidates


def score_image(pred: np.ndarray, gt: np.ndarray, image: np.ndarray | None = None,
                candidates: np.ndarray | None = None, return_classes: bool = False) -> dict:
    if candidates is None and image is not None:
        candidates = ring_candidates(image, gt)
    row = evaluate_image(pred, gt)
    row.update(diagnose_image(pred, gt))
    tax = classify_errors(pred, gt, candidates)
    row.update(tax["counts"])
    if return_classes:
        row["_classes"] = {"pred": tax["pred"], "gt": tax["gt"]}
    return row
