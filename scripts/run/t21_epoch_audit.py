"""T-21 epoch audit (16 Nov): was 150 epochs worth it? Parses nnU-Net training logs (results/phase1/t21_logs/fold*.txt,
renamed from the Volume logs to log<fold>.txt in the target dir) into per-epoch train/val loss, pseudo Dice and the
nnU-Net EMA (0.9 old + 0.1 new), then reports optima and the first epoch within 0.01 / 0.005 / 0.002 of the best EMA.

    .venv/bin/python scripts/run/t21_epoch_audit.py <dir with log1.txt log2.txt log3.txt>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
rows = []
for f in sorted(D.glob("log*.txt")):
    fold = int(f.stem[3:])
    ep = None
    cur = {}
    for line in f.read_text().splitlines():
        if m := re.search(r"Epoch (\d+)\s*$", line):
            if cur:
                rows.append(cur)
            ep = int(m.group(1))
            cur = {"fold": fold, "epoch": ep}
        elif ep is None:
            continue
        elif m := re.search(r"Current learning rate: ([\d.e-]+)", line):
            cur["lr"] = float(m.group(1))
        elif m := re.search(r"train_loss (-?[\d.]+)", line):
            cur["train_loss"] = float(m.group(1))
        elif m := re.search(r"val_loss (-?[\d.]+)", line):
            cur["val_loss"] = float(m.group(1))
        elif "Pseudo dice" in line:
            v = [float(x) for x in re.findall(r"np\.float32\(([\d.]+)\)", line)]
            cur["dice_axon"], cur["dice_myelin"] = v
            cur["dice_mean"] = float(np.mean(v))
        elif m := re.search(r"Epoch time: ([\d.]+)", line):
            cur["sec"] = float(m.group(1))
        elif m := re.search(r"New best EMA pseudo Dice: ([\d.]+)", line):
            cur["new_best_ema"] = float(m.group(1))
        elif m := re.search(r"Mean Validation Dice:\s+([\d.]+)", line):
            cur["final_val_dice"] = float(m.group(1))
    if cur:
        rows.append(cur)
df = pd.DataFrame(rows)
df = df[df.train_loss.notna()].copy()
# nnU-Net's EMA: ema = 0.9 * ema + 0.1 * mean pseudo dice
out = []
for fold, d in df.groupby("fold"):
    d = d.sort_values("epoch").copy()
    ema = []
    e = None
    for x in d.dice_mean:
        e = x if e is None else 0.9 * e + 0.1 * x
        ema.append(e)
    d["ema"] = ema
    d["val_loss_s"] = d.val_loss.rolling(5, center=True, min_periods=1).mean()
    last = d.epoch.max()
    best_ema = d.ema.max()
    r = {"fold": fold, "last_epoch": last, "sec_per_ep": d.sec.iloc[1:].median(),
         "best_ema": best_ema, "best_ema_ep": int(d.epoch[d.ema.idxmax()]),
         "min_val_loss": d.val_loss.min(), "min_val_loss_ep": int(d.epoch[d.val_loss.idxmin()]),
         "min_val_loss_s_ep": int(d.epoch[d.val_loss_s.idxmin()]),
         "min_train_loss_ep": int(d.epoch[d.train_loss.idxmin()]),
         "max_dice_ep": int(d.epoch[d.dice_mean.idxmax()]), "max_dice": d.dice_mean.max()}
    for tol in (0.01, 0.005, 0.002):
        r[f"ema_within_{tol}"] = int(d.epoch[d.ema >= best_ema - tol].min())
    r["final_ema"] = d.ema.iloc[-1]
    r["gap_last"] = d.val_loss.iloc[-1] - d.train_loss.iloc[-1]
    r["gap_at_best"] = float((d.val_loss - d.train_loss)[d.ema.idxmax()])
    # val loss trend in the last 30 epochs (overfitting check)
    tail = d[d.epoch > last - 30]
    r["val_loss_slope_last30"] = np.polyfit(tail.epoch, tail.val_loss, 1)[0]
    r["train_loss_slope_last30"] = np.polyfit(tail.epoch, tail.train_loss, 1)[0]
    out.append(r)
    for a, b in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 151)):
        s = d[(d.epoch >= a) & (d.epoch < b)]
        if len(s):
            print(f"fold {fold} ep {a:3d}-{b-1:3d}: train {s.train_loss.mean():+.4f} val {s.val_loss.mean():+.4f} "
                  f"dice {s.dice_mean.mean():.4f} (ax {s.dice_axon.mean():.4f} my {s.dice_myelin.mean():.4f}) "
                  f"ema_end {s.ema.iloc[-1]:.4f} lr {s.lr.iloc[-1]:.5f}")
df.to_csv(D / "epochs.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(out).round(4).T.to_string())
