"""Leakage-free splits: grouped by image (or animal once metadata is known), never by tile.

See docs/notes/05-evaluation-framework.md. Split files are frozen JSON under configs/splits/.
"""


def make_grouped_kfold(image_ids, groups=None, n_splits=5, seed=0):
    raise NotImplementedError
