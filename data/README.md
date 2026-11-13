# Data (local only, git-ignored)

The challenge data's licence is unstated, so nothing under `data/` is ever committed or re-hosted. That includes images, masks, predictions, crops and checkpoints.
A private copy lives on the Modal Volume `tem-axon-myelin` (never public).

```
data/
  raw/train-dataset/
    train_samples/sample_0001..0266.tif           # uint8 TEM images, 3008×4096 (H×W), 2.152 nm/px
    train_instance_labels/sample_0001..0266.tif   # uint16 masks: 0 bg, 1 ignore, X axon, X+1 its myelin
  nnunet/
    raw/Dataset501_TEMAxonMyelin/          # baselines/nnunet/convert.py output (labels 0 bg, 1 axon, 2 myelin, 3 ignore)
    preprocessed/Dataset501_TEMAxonMyelin/ # nnUNetv2_plan_and_preprocess output + splits_final.json (uploaded to Modal)
    results_modal/                         # B1 files downloaded from the Volume (validation PNGs, logs, budget/GPU logs)
    logs/                                  # preprocessing, upload and Modal launch logs; the original plans file
  predictions/<baseline>/sample_XXXX.tif   # uint16 paired-instance predictions on dev-val (axonpath, b1, cellpose_b1, …)
  predictions/b1_sem/                      # B1 semantic maps (used by the O-sep oracle)
  processed/
    eda/                                   # EDA tables and figures (contain image crops)
    day2/, day4/                           # visual checks (overlap crops, FP galleries)
    modal_runs/                            # GPU-utilisation CSVs
```

**Getting the data:** download it from the Dataset section of https://alexandrojim.github.io/tem-challenge/ (ICM OneDrive) and unzip it to `data/raw/`.
The 134 test images are not used.

**External datasets** (ASTIH TEM1, AxonPath, …): see `docs/notes/02-challenge-and-data.md` §5. None has been downloaded yet.
