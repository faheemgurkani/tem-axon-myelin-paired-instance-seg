"""Replica of the official challenge metrics (docs/notes/01-challenge-spec.md, Evaluation).

Dice (axon, myelin, mean), HD95 in nm (pixel size 2.152 nm), and per-fiber g-ratio MAE over fibers matched
by IoU on the fiber mask (X ∪ X+1). Ambiguous details are controlled by the `evaluation` block in configs/default.yaml.
"""


def dice(pred, gt, ignore=None):
    raise NotImplementedError


def hd95_nm(pred, gt, pixel_size_nm=2.152, ignore=None):
    raise NotImplementedError


def match_fibers(pred_labels, gt_labels, iou_threshold=0.0):
    raise NotImplementedError


def g_ratio_mae(pred_labels, gt_labels, matches):
    raise NotImplementedError


def evaluate_image(pred_labels, gt_labels, cfg):
    raise NotImplementedError
