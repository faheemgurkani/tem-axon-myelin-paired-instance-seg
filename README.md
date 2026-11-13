# tem-axon-myelin-paired-instance-seg

Paired (nested) instance segmentation of **axons and myelin sheaths** in **transmission electron microscopy (TEM)**
images of mouse corpus callosum, with per-fiber **g-ratio** morphometry.

The problem statement and benchmark protocol come from the
[SIPAIM 2026 TEM Axon & Myelin Segmentation Challenge](https://alexandrojim.github.io/tem-challenge/)
(Paris Brain Institute, ICM.Quant). This repository is an independent research project. It is not affiliated with the
organisers and is not a challenge entry.

**Start here:** [`docs/notes/01-study-guide.md`](docs/notes/01-study-guide.md) covers the problem, the gap, the hypotheses, what was done, what was found and where things stand.

## Task

- Each myelinated fiber is a linked pair: axon `X` (even ≥ 2) and its myelin ring `X+1`.
- Label `0` is background (including unmyelinated axons). Label `1` is ignored; in practice it marks border-cut and deliberately excluded fibers.
- Masks are `uint16`, 4096×3008 px at 2.152 nm/px.
- Metrics: Dice (axon, myelin), HD95 in nm, and the g-ratio MAE over matched fibers, combined by aggregate rank.

Full spec and data facts: [`docs/notes/02-challenge-and-data.md`](docs/notes/02-challenge-and-data.md).

## Research question

How much of the fiber-level error in paired axon–myelin segmentation comes from each source, once annotation noise is accounted for?
- **selection:** which fibers the annotators chose to label;
- **separation/pairing:** grouping pixels into fibers and linking each axon to its ring;
- **boundary placement.**

And is a decoder that makes each fiber's axon and ring nested *by construction* worth building?

The plan is evaluation-first ([`docs/notes/07-plan.md`](docs/notes/07-plan.md)):
1. **C1:** an evaluator that decomposes errors (official-metric replica, pairing correctness, an error taxonomy, oracle experiments, human ceilings from overlapping tiles).
2. Strong baselines.
3. A pre-registered go/no-go (H1) on whether pairing is a large error source.
4. Then either the nested decoder (**C2**) or learning with incomplete annotation (**C3**).
5. External replication on ASTIH TEM1.

## Status (13 Nov 2026): days 0–5 done, H1 = NO-GO

Results on 53 held-out dev images (1,619 fibers), with 95% cluster-bootstrap CIs over overlap groups:

| | Fiber F1@0.5 | Dice axon / myelin | g-MAE | Separation share of errors |
|---|---|---|---|---|
| Human vs human (overlapping tiles) | 0.929 | 0.987 / 0.951 | 0.0067 | 0% |
| **B1:** nnU-Net ResEnc-M (896×1152 patches) + watershed pairing | **0.904** [0.881, 0.928] | 0.882 / 0.859 | 0.016 | **0.6%** |
| Cellpose-SAM zero-shot regions split by B1's pixels | 0.885 | 0.879 / 0.857 | 0.011 | 1.0% |
| AxonPath zero-shot | 0.784 | 0.71 / 0.62 | 0.060 | 9% |

**Findings:**
- **Separation/pairing is not the bottleneck on this data.** Perfect grouping of B1's pixels gains −0.001 F1.
- **Selection is.** 78% of B1's errors are predictions on fibers the annotators left unlabelled, and selection is the only thing two humans disagree on.
- **Grouping does matter for the g-ratio.** Perfect grouping cuts B1's g-MAE by 42%, and fibers with broken predicted rings carry 56% of the g error. This is recorded as a post-hoc hypothesis (H5).

Next steps, pending a budget decision: C3, more C1 analyses, ASTIH TEM1 replication.

All results: [`docs/notes/05-results.md`](docs/notes/05-results.md).

## Layout

```
configs/
  splits/              frozen, group-aware splits (dev.json, cv3.json) and overlap groups
  default.yaml         evaluator choices for every ambiguity in the challenge spec
src/axon_myelin_seg/
  data/                loader (io), label encoding, training-label cleaning, group-aware splits
  evaluation/          official-metric replica, diagnostics, error taxonomy, oracles, bootstrap stats, runner
  postprocessing/      shared semantic → paired-instance step (watershed)
  models/              our methods (empty; C2/C3 not built yet)
baselines/
  nnunet/              dataset conversion and custom trainers (B1)
  axonpath/            AxonPath zero-shot wrapper
  cellpose_sam/        Cellpose-SAM zero-shot
  axondeepseg/, micro_sam/   placeholders (deferred)
infra/modal/           GPU app for Modal (training, inference) and a resumable uploader
scripts/
  eda/                 exploratory data analysis
  run/                 one script per plan step (day1_*, day2_*, …, score_predictions, day4_oracles, day5_analysis)
  train.py, predict.py, evaluate.py, package_submission.py   challenge-format entry points (stubs)
results/day1..day5/    committed metric tables and logs (no images or masks); see results/README.md
docs/
  notes/               the 7 research notes (start with 01-study-guide.md)
  reports/             deep-research reports (literature review, gap validation)
tests/                 evaluator, pairing, data and early-stopping tests
data/                  local only, git-ignored; see data/README.md
```

**Not yet implemented** (docstring stubs): `evaluation/ranking.py`, `evaluation/morphometry.py`, `data/tiling.py`, and the four `scripts/*.py` entry points. Evaluation runs through `scripts/run/score_predictions.py`.

## Setup

Python 3.12 with [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev,analysis,nnunet,cellpose,cloud]"
uv pip install -e external/axonpath   # AxonPath baseline: git clone https://github.com/paucabar/axonpath external/axonpath (commit 6d633bf)
.venv/bin/pytest -q tests             # run "tests", not bare pytest: external/ has its own test suites
```

**Compute split:**
- CPU work runs locally: conversion, nnU-Net preprocessing, pairing, evaluation, zero-shot baselines.
- GPU training and inference run on [Modal](https://modal.com) through `infra/modal/tem_modal.py`.

Setup, costs and the full reproduce recipe: [`docs/notes/06-compute-and-reproduce.md`](docs/notes/06-compute-and-reproduce.md).

## Quickstart: score a folder of predictions

Predictions are uint16 maps named like the images (`sample_XXXX.tif`), using the X / X+1 encoding.

```bash
.venv/bin/python scripts/run/score_predictions.py --pred-dir data/predictions/<name> --name <name>
.venv/bin/python scripts/run/day4_oracles.py      --pred-dir data/predictions/<name> --name <name>
```

The outputs are written to `results/day4/`:
- per-image metrics;
- pooled F1 with CIs;
- the error taxonomy;
- oracle gains.

## Data

The challenge data comes from the organisers and is **not** included here. Its licence is unstated, so it is never committed or re-hosted. See [`data/README.md`](data/README.md).

## License

Code: MIT (see `LICENSE`). Datasets and third-party code keep their own licences.
