"""Leakage-free splits (T-03, D-03): grouped by overlap group, stratified by fiber-density tercile.

Why groups: 40 images form 18 groups of tiles that show the *same tissue* (docs/notes/11-deep-eda.md §D). If two of
them land on opposite sides of a split, the model is validated on tissue it trained on. Every image that is not in
``configs/splits/overlap_groups.json`` is its own group.

Why density strata: fiber density varies ~13× between images (E9). A random split can put all sparse images in
validation, which would make the validation score unrepresentative.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[3]
OVERLAP_GROUPS = REPO / "configs" / "splits" / "overlap_groups.json"


def image_groups(names: list[str], groups_file: str | Path = OVERLAP_GROUPS) -> dict[str, str]:
    """Map each sample name (stem) to a group id. Overlap groups are 'g00'..; singletons are their own name."""
    groups = json.loads(Path(groups_file).read_text())
    out = {n: n for n in names}
    for i, members in enumerate(groups):
        for m in members:
            stem = Path(m).stem
            if stem in out:
                out[stem] = f"g{i:02d}"
    return out


def density_tercile(density: dict[str, float]) -> dict[str, int]:
    vals = np.array(list(density.values()))
    q = np.quantile(vals, [1 / 3, 2 / 3])
    return {k: int(np.searchsorted(q, v, side="right")) for k, v in density.items()}


def _group_strata(names, groups, strata):
    """Stratum per image = tercile of the group's mean density, so a group's members share one stratum."""
    by_group: dict[str, list[float]] = {}
    for n in names:
        by_group.setdefault(groups[n], []).append(strata[n])
    g_stratum = {g: round(float(np.mean(v))) for g, v in by_group.items()}
    return [g_stratum[groups[n]] for n in names]


def make_splits(names: list[str], density: dict[str, float], n_splits: int, seed: int = 0) -> list[dict]:
    """StratifiedGroupKFold folds as [{'train': [...], 'val': [...]}, ...] (sorted stems)."""
    names = sorted(names)
    groups = image_groups(names)
    y = _group_strata(names, groups, density_tercile(density))
    g = [groups[n] for n in names]
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, va in skf.split(np.zeros(len(names)), y, g):
        folds.append({"train": sorted(names[i] for i in tr), "val": sorted(names[i] for i in va)})
    return folds


def check_split(fold: dict, names: list[str]) -> None:
    """Assert disjoint, complete and group-respecting."""
    tr, va = set(fold["train"]), set(fold["val"])
    assert not tr & va, "train/val overlap"
    assert tr | va == set(names), "split does not cover all images"
    groups = image_groups(sorted(names))
    gtr, gva = {groups[n] for n in tr}, {groups[n] for n in va}
    assert not gtr & gva, f"overlap group crosses the split: {sorted(gtr & gva)}"
