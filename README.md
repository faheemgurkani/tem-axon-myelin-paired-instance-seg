# tem-axon-myelin-paired-instance-seg

Paired (nested) instance segmentation of **axons and myelin sheaths** in **transmission electron microscopy (TEM)**
images of mouse corpus callosum, with per-fiber **g-ratio** morphometry.

The problem statement and benchmark protocol come from the
[SIPAIM 2026 TEM Axon & Myelin Segmentation Challenge](https://alexandrojim.github.io/tem-challenge/)
(Paris Brain Institute, ICM.Quant). This repository is an independent research project and is not affiliated with the organizers.

## Task

Each myelinated fiber is a linked pair: axon `X` (even ≥ 2) and its myelin ring `X+1`. Label `0` is background and `1`
is artifacts (ignored). Outputs are `uint16` instance maps. Evaluation: Dice (axon/myelin), HD95 in nm, and per-fiber
g-ratio MAE, combined by aggregate rank. Full spec: [`docs/notes/01-challenge-spec.md`](docs/notes/01-challenge-spec.md).

## Roadmap

1. Evaluation framework: official-metric replica + instance/pairing diagnostics + morphometry
2. Baselines: nnU-Net v2 + pairing, AxonDeepSeg/ASTIH, Cellpose-SAM, μSAM
3. Error analysis by failure type and fiber size
4. Our method: nested-instance representation with containment-topology and g-ratio-aware losses

Research notes: [`docs/notes/00-index.md`](docs/notes/00-index.md).

## Layout

```
configs/            experiment, data and split configs
data/               local data only (git-ignored); see data/README.md
docs/notes/         research notes (spec, literature, plan)
src/axon_myelin_seg/
  data/             I/O, label encoding, tiling, splits
  evaluation/       official metrics, ranking, diagnostics, morphometry
  postprocessing/   semantic → paired-instance conversion
  models/           our methods
baselines/          wrappers and configs for third-party baselines
scripts/            train.py, predict.py, evaluate.py, package_submission.py
submission/         challenge submission template
tests/
```

## Setup

```bash
conda create -n axmy python=3.11 -y && conda activate axmy
pip install -e ".[dev]"
```

## Data

The challenge data is distributed by the organizers and is **not** included here. See [`data/README.md`](data/README.md).

## License

Code: MIT (see `LICENSE`). Datasets keep their own licences.
