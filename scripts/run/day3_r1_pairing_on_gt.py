"""T-09 / result R1 (oracle O-pair-GT): GT *semantics* → shared watershed pairing → compare with GT *instances*.

The pixels are perfect here, so every error is caused by the pairing rule itself. The main case is the
watershed line between touching sheaths. This bounds how good *any* semantic-segmentation-plus-pairing baseline
can be at the instance level.

Runs on all 266 images (CPU, local). Output: results/day3/r1_pairing_on_gt.csv (+ summary json).
"""

import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from axon_myelin_seg.data.io import list_samples, load_sample
from axon_myelin_seg.data.splits import image_groups
from axon_myelin_seg.evaluation.diagnostics import fiber_checks
from axon_myelin_seg.evaluation.runner import score_image
from axon_myelin_seg.evaluation.stats import cluster_bootstrap, micro_f1
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]


def one(name):
    img, gt = load_sample(name)
    ax = (gt >= 2) & (gt % 2 == 0)
    my = (gt >= 3) & (gt % 2 == 1)
    pred = semantic_to_paired_instances(ax, my)
    pred[gt == 1] = 0
    r = score_image(pred, gt, image=img)
    r["name"] = name
    gc = [v for k, v in fiber_checks(gt).items()]  # GT violation baseline (T-05 expectation)
    r["gt_n_objects"] = len(gc)
    for key in ("axon_without_ring", "ring_without_axon", "axon_touches_bg", "ring_euler_bad", "violation"):
        r[f"gt_n_{key}"] = int(sum(v[key] for v in gc))
    return r


def main():
    names = list_samples()
    with Pool(4) as p:
        rows = p.map(one, names)
    df = pd.DataFrame(rows)
    df["group"] = df.name.map(image_groups(names))
    dev_val = set(json.loads((REPO / "configs/splits/dev.json").read_text())["val"])
    df["dev_val"] = df.name.isin(dev_val)
    out = REPO / "results" / "day3"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "r1_pairing_on_gt.csv", index=False)
    summ = {}
    for subset, d in (("all_266", df), ("dev_val_53", df[df.dev_val])):
        s = {"f1@0.5": cluster_bootstrap(d, micro_f1), "f1@0.75": cluster_bootstrap(d, lambda x: micro_f1(x, "0.75"))}
        for c in ("dice_axon", "dice_myelin", "g_mae", "pairing_correct_rate", "violation_rate", "pq"):
            s[c] = float(d[c].mean())
        for c in [c for c in d.columns if c.startswith(("pred_", "gt_", "n_merges", "n_splits", "tp@", "fp@", "fn@"))]:
            s[c] = int(d[c].sum())
        summ[subset] = s
    (out / "r1_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
