# Data (local only — git-ignored)

Expected layout:

```
data/
  raw/challenge/train/images/*.tif     # 266 annotated TEM images, 4096x3008
  raw/challenge/train/masks/*.tif      # uint16 paired-instance masks
  raw/challenge/test/images/*.tif      # 134 unannotated test images
  raw/external/{astih_tem1,astih_tem2,aimseg}/
  processed/                           # tiles, nnU-Net datasets, etc.
```

Challenge data: download from the Dataset section of https://alexandrojim.github.io/tem-challenge/ (ICM OneDrive).
Licence and redistribution terms are unstated, so never commit or re-host it. External datasets: see `docs/notes/02-challenge-and-data.md` §5.
