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

## Research question and roadmap (plan v2)

How much of the fiber-level error in paired axon–myelin segmentation comes from **fiber selection**, **separating touching
fibers / pairing**, and **boundary placement**, once annotation noise is accounted for? Does a decoder that makes each
fiber's axon and ring nested by construction remove the pairing part?

1. Evaluation framework: official-metric replica + pairing-correctness, FP taxonomy and error decomposition via oracle experiments
2. Baselines: nnU-Net v2 + watershed pairing, Cellpose (fibers), AxonPath
3. Go/no-go: is pairing/separation a larger error source than selection, beyond annotation noise?
4. Our method (if go): nested-by-construction decoder; otherwise learning with incomplete fiber annotation
5. Replication on public data (ASTIH TEM1)

Full plan: [`docs/notes/09-research-plan.md`](docs/notes/09-research-plan.md).

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
