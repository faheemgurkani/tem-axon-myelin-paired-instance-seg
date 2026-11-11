# 12 — Resources and pre-flight for days 0–5 (12 Nov 2026)

This lists everything collected for tasks T-00 to T-14 in [09](09-research-plan.md): code, papers, environment and data.
It also records the facts we verified in those resources, the corrections they force on the plan, and what is still missing.

**Status: collection done. No training or preprocessing has run.** The GPU and large-disk machine arrives on 13 Nov.

## 1. Hardware, now and needed

| | Local Mac (now) | Needed for T-08/T-10 (requested) | Why |
|---|---|---|---|
| Compute | Apple M4, 10 cores, 16 GB RAM, MPS (Metal) only, no CUDA | 1 NVIDIA GPU with ≥ 24 GB VRAM (L4, A10G, RTX 4090, A100), ≥ 8 CPU cores, ≥ 32 GB RAM | nnU-Net ResEnc-L targets 24 GB VRAM; ResEnc-M needs 9–11 GB (nnU-Net `resenc_presets.md`). nnU-Net augmentation is CPU-bound (`nnUNet_n_proc_DA`). The nnU-Net docs say training on MPS/CPU "will take a really long time" (`installation_instructions.md`) |
| Disk | 4.7 GB free after setup (98% full) | ≥ 100 GB | nnU-Net raw copy (~3.3 GB) + preprocessed arrays (estimate 5–10 GB) + checkpoints + predictions with softmax `.npz` for 266 full images (est. tens of GB) |

**Local use until then:** evaluator development (T-01–T-07), AxonPath inference (T-11, measured at ~2 s/image on MPS) and the analysis code.
Training (T-08 nnU-Net, T-10 Cellpose fine-tuning) waits for the GPU machine.

## 2. Code (all under `external/`, git-ignored, shallow clones)

| Repo | Commit (date) | Used in | What we use | Facts verified in the code/docs |
|---|---|---|---|---|
| `nnUNet` (MIC-DKFZ) | 7bb9261 (2026-11-10) | T-08 (B1) | Planning, training, `splits_final.json`, ignore label | (a) `nnUNetTrainer_250epochs` exists (`variants/training_length/nnUNetTrainer_Xepochs.py`) → D-10 confirmed. (b) The default trainer runs 1000 epochs × 250 iterations, plus 50 validation iterations per epoch. (c) The ignore label must be the **highest** integer and be named `ignore` (`ignore_label.md`) → our labels {0 bg, 1 axon, 2 myelin, 3 ignore}. (d) 2D `.tif` is read by `NaturalImage2DIO` (no spacing JSON needed; that is only for `Tiff3DIO`). (e) Custom splits go in `nnUNet_preprocessed/<Dataset>/splits_final.json`: a list of {train, val} dicts (`manual_data_splits.md`). (f) Normalisation defaults to per-case z-score for non-CT channels. (g) `-device mps` is supported but not recommended for training |
| `cellpose` (MouseLand) | a54cb48 (2026-06-14), v4.2.1.1 installed | T-10 (B3) | `cpsam_v2` / `cpdino` fine-tuning, `train_seg` | Cellpose-SAM was trained on ROI diameters **7.5–120 px**, mean 30 (`models.rst`, `settings.rst`) → **D-12 correction: downsample ×6, not ×4.** Fiber median 755 nm = 351 px → 59 px at ×6; 5–95% range 481–1279 nm → 37–99 px. At ×4 the 95th percentile would be 149 px, outside the training range. Models `cpsam_v2` and `cpdino` were released June 2026 (`models.rst`). Training CLI: `python -m cellpose --train --dir … --learning_rate 1e-5 --weight_decay 0.1 --n_epochs 100` (`train.rst`); masks are named `*_masks` |
| `axonpath` (paucabar) | 6d633bf (2026-10-01), v0.1.0 installed | T-11 | Zero-shot baseline | Weights are **bundled** (`axonpath/weights/em.pth`, `bf.pth`), so no Zenodo download is needed. `em_cns` runs at **8 nm/px**, `min_diameter` 20 px. Input must be percentile-normalised (1–99%) with `axonpath.utils.image_processing.normalize`. `run_inference(image, model, device, roi_size=(512,512), min_diameter, …)` returns (fibre labels, axon labels, inner-cylinder labels, semantic map). Axons and inner cylinders carry the containing fibre's ID. Semantic classes: 0 bg, 1 myelin, 2 inner cylinder. **Smoke test:** sample_0217 at 2.152→8 nm/px ran in 1.9 s on MPS, giving 54 fibres / 49 axons / 48 inner cylinders against 27 GT fibers. GT excludes border-cut (label 1) and unlabelled fibers, so this count is not an error measure |
| `ASTIH` (axondeepseg) | 089896e (2026-10-07) | T-17 (day 9), and the evaluator design (T-04/T-05) | Data download, splits, the evaluation protocol | TEM1 = DANDI 001436 (v0.250512.1625); test subject `sub-nyuMouse26` (`scripts/dataset_list.json`). Evaluation (`evaluate_models.py`): MONAI `DiceMetric`/`MeanIoU`; instances from ADS `get_watershed_segmentation` on **both** prediction and GT (axon centroids as seeds); fiber matching uses `stardist.matching.matching`. **The CLI default threshold is 0.3, though its help text says 0.5.** Any reproduction must therefore set the threshold explicitly (0.3 and 0.5 are both reported in the paper) |
| `axondeepseg` | 05a8dd1 (2026-10-30) | T-09 reference, T-17 | `get_watershed_segmentation` (the ADS pairing rule we compare against) | Centroid-seeded watershed (report A) |
| `TopoInteraction`, `micro-sam` | — | Not needed days 0–5 | — | — |

## 3. Python environment

- Location: `.venv/` in the repo (git-ignored), Python 3.12.15, created with `uv`.
- Size: 1.6 GB.
- Installed: `-e .[dev,nnunet,cellpose]` plus `-e external/axonpath`, `opencv-python-headless` and `PyWavelets`.
- Versions: torch 2.15.0 (MPS available), nnunetv2 2.8.1, cellpose 4.2.1.1, monai 1.6.1, axonpath 0.1.0, scikit-image 0.26.0, numpy 2.5.4.
- On the GPU machine, recreate it with a CUDA torch build: `uv venv --python 3.12 && uv pip install -e ".[dev,nnunet,cellpose]" -e external/axonpath opencv-python-headless PyWavelets`, using torch from the matching CUDA index.
- **Not installed:** `stardist`, because it pulls TensorFlow. Its `matching` function is reimplemented in T-04: IoU matrix + Hungarian at a threshold. A unit test will check it against a small hand-computed case.

## 4. Papers (all in `docs/references/papers/`, git-ignored)

| Paper | File | Used for |
|---|---|---|
| Isensee et al., nnU-Net (arXiv 1809.10486; Nature Methods 2021) | `isensee2018_nnunet_1809.10486.pdf` | T-08 design, why nnU-Net is the bar |
| Isensee et al., nnU-Net Revisited (MICCAI 2024, arXiv 2404.09556) | `isensee2024_nnunet_revisited_2404.09556.pdf` | ResEnc presets; the argument for strong baselines |
| Stringer et al., Cellpose (bioRxiv 2020; Nature Methods 2021) | `stringer2020_cellpose.pdf` | T-10: flow representation; why rings need filled fibers |
| Pachitariu et al., Cellpose-SAM (bioRxiv 2025.04.28.651001) | `pachitariu2025_cellpose_sam.pdf` | T-10: `cpsam` backbone, size invariance |
| Schmidt et al., StarDist (MICCAI 2018, arXiv 1806.03535) | `schmidt2018_stardist_1806.03535.pdf` | T-04/T-05: the `matching` definition ASTIH uses |
| Kirillov et al., Panoptic Segmentation (CVPR 2019, arXiv 1801.00868) | `kirillov2019_panoptic_quality_1801.00868.pdf` | T-05: PQ = SQ × RQ |
| Maier-Hein et al., Metrics Reloaded (Nat Methods 2024, arXiv 2206.01653) | `maierhein2024_metrics_reloaded_2206.01653.pdf` | T-04/T-05: metric choice, aggregation |
| Reinke et al., Metric pitfalls (Nat Methods 2024, arXiv 2302.01790) | `reinke2024_metric_pitfalls_2302.01790.pdf` | D-07/D-22: HD95 variants, non-independent samples |
| Zaimi et al., AxonDeepSeg (Sci Rep 2018) | `zaimi2018_axondeepseg_scirep.pdf` | T-09: the ADS watershed pairing rule |
| Collin et al., ASTIH (MIDL 2026) | `collin26a_ASTIH_MIDL2026.pdf` (+ `.txt`) | T-04/T-05/T-13 reference numbers |
| Collin et al., generalist model (arXiv 2409.11552) | `generalist_2409.11552.pdf` | Context |
| de Geus et al., Part-aware panoptic (CVPR 2021) | `degeus2021_part_aware_panoptic_2106.06351.pdf` | C2 (day 6+) and the instance-first, part-second principle |
| Gupta et al., TopoInteraction (ECCV 2022) | `arxiv_2207.09654.pdf` | T-05 violation definitions |
| Valverde et al., TopoMortar (arXiv 2503.03365) | `arxiv_2503.03365.pdf` | D-19: augmentation in every arm |

**Missing (not needed for days 0–5):**
- **Bi-Polar Mask** (IEEE, paywalled) — needed for the C2 novelty wording.
- **AxonPath preprint** — not out yet.
- **AimSeg**: its data and paper are open access but not downloaded.

## 5. Data (unchanged; no new downloads)

- **Challenge train set:** `data/raw/train-dataset/{train_samples,train_instance_labels}/sample_0001..0266.tif`. The zip is still in `data/` (3.3 GB).
  **Deleting the zip would free 3.3 GB locally. That is the user's call; not done.**
- **EDA tables and figures:** `data/processed/eda/`. Overlap groups: `configs/splits/overlap_groups.json`.
- **External sets** (ASTIH TEM1, AxonPath data): deferred to day 9 and the GPU machine.

## 6. Plan corrections forced by what we found (applied to [09](09-research-plan.md))

| Item | Was | Now | Evidence |
|---|---|---|---|
| T-00 compute | "confirm 1 GPU" | No local GPU. A GPU machine arrives 13 Nov. Before then, only the CPU/MPS tasks above | §1 |
| D-09 / T-08 | ResEnc-L | **ResEnc-L if the GPU has ≥ 24 GB, else ResEnc-M (9–11 GB)**. The decision depends on the machine we get | nnU-Net `resenc_presets.md` |
| D-10 | trainer name to verify | `nnUNetTrainer_250epochs` confirmed. Training time is **still unmeasured**: the MPS benchmark was cancelled, so measure the epoch time on the GPU in the first hour | §2 |
| D-12 / T-10 | Cellpose at ×4 | **×6** (12.9 nm/px) | §2 cellpose row |
| D-13 / T-11 | AxonPath via Zenodo | Bundled `em` weights; percentile normalisation; runs locally (~2 s/image) | §2 axonpath row |
| T-04 / T-17 | "ASTIH F1 0.791" | Always set the matching threshold explicitly (0.3 and 0.5). ASTIH's CLI default is 0.3 | §2 ASTIH row |

## 7. Pre-flight checklist for the GPU machine (run in this order on 13 Nov; nothing run yet)

1. `git clone` the repo, then copy `data/raw/train-dataset` (or the zip) and `external/` (or re-clone at the commits in §2).
2. Create the environment (§3) with CUDA torch. Check that `torch.cuda.is_available()` and the GPU name/VRAM are reported.
3. Set `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` on the large disk.
4. T-01 to T-03 (loader, cleaner, splits). Commit the split JSONs.
5. T-08 conversion → `nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEnc{M|L}` → copy `configs/splits/dev.json` to `splits_final.json` → train fold 0 with `-tr nnUNetTrainer_250epochs --npz`. **Record the seconds per epoch after epoch 2** and update the D-10 estimate.
6. While B1 trains, do T-04 to T-07 (evaluator, ceilings) and T-11 (AxonPath), and prepare the T-10 Cellpose data at ×6.
