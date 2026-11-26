"""T-30 (G1, H11): fine-tune Cellpose-SAM (cpsam_v2) on our labels, on Modal (GPU). One run per target.

Reads /vol/g1_cellpose/{train,val}/<name>_{img,<target>}.tif (written locally by scripts/run/g1_cellpose_prep.py
and uploaded), trains with Cellpose's own recipe for fine-tuning (AdamW, lr 1e-5, weight decay 0.1, 256-px crops,
random rotation/scaling) and writes the model to /vol/runs/g1_cellpose/<target>/models/<target>_cpsam_ft.
The val images are only monitored (test loss), never trained on.

    python /root/repo/baselines/cellpose_sam/finetune.py --target fiber --epochs 100 [--limit 8]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

VOL = Path("/vol")


def load(part: str, target: str, limit: int) -> tuple[list, list]:
    d = VOL / "g1_cellpose" / part
    stems = sorted(p.name.removesuffix("_img.tif") for p in d.glob("*_img.tif"))
    stems = stems[:limit] if limit else stems
    imgs = [tifffile.imread(d / f"{s}_img.tif").astype(np.float32) for s in stems]
    labs = [tifffile.imread(d / f"{s}_{target}.tif").astype(np.int32) for s in stems]
    return imgs, labs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["fiber", "axon"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from cellpose import io, models, train
    io.logger_setup()
    out = VOL / "runs" / "g1_cellpose" / a.target
    out.mkdir(parents=True, exist_ok=True)
    tr_x, tr_y = load("train", a.target, a.limit)
    va_x, va_y = load("val", a.target, a.limit // 4 if a.limit else 0)
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
    t0 = time.time()
    path, train_losses, test_losses = train.train_seg(
        model.net, train_data=tr_x, train_labels=tr_y, test_data=va_x, test_labels=va_y,
        batch_size=a.batch, learning_rate=1e-5, weight_decay=0.1, n_epochs=a.epochs, save_path=str(out),
        save_every=max(1, a.epochs // 4), min_train_masks=1, model_name=f"{a.target}_cpsam_ft")
    log = {"target": a.target, "epochs": a.epochs, "batch": a.batch, "n_train": len(tr_x), "n_val": len(va_x),
           "seconds": time.time() - t0, "model": str(path), "train_loss": [float(x) for x in train_losses],
           "test_loss": [float(x) for x in test_losses]}
    (out / f"train_log{'_smoke' if a.limit else ''}.json").write_text(json.dumps(log))
    print(json.dumps({k: v for k, v in log.items() if "loss" not in k}), flush=True)


if __name__ == "__main__":
    main()
