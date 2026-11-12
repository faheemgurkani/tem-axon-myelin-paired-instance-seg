# 16 — Day 3: baselines (T-08 to T-11), executed 13–14 Nov 2026

Three baselines are planned for the comparison: **B1** nnU-Net + the shared watershed pairing, **B3** Cellpose-SAM, and **AxonPath** zero-shot. Every number is scored by the day-2 evaluator ([15](15-day2-evaluator.md)) on the 53 dev-val images (`configs/splits/dev.json`, v2).

## T-08 nnU-Net dataset — `baselines/nnunet/convert.py` (local, CPU)

**Conversion.** It ran on the M4 in 2 min 20 s with 3 workers. The output is 266 PNG image/label pairs (2.5 GB) in `data/nnunet/raw/Dataset501_TEMAxonMyelin`.

Semantic labels:

| Label | Meaning |
|---|---|
| 0 | background |
| 1 | axon |
| 2 | myelin |
| 3 | ignore = GT label 1 + the 47 orphan IDs + the 296 topology-defect fibers |

**Label fractions** (mean over 266 images; `results/day3/nnunet_label_fractions.csv`):

| Class | Mean | Range |
|---|---|---|
| background | 56.4% | |
| axon | 16.6% | |
| myelin | 11.3% | |
| ignore | **15.8%** | 0–51% |

**Easy to miss:**
- **About one sixth of all pixels give no training signal at all**: border-cut fibers, excluded fibers, cleaned defects. In the worst image it is half. The network never learns what to do on "excluded" large or oblique fibers, so at inference it will segment them.
- That is why the evaluator ignores predictions on GT label 1. Without that rule, nnU-Net would be punished for doing the obvious thing.
- nnU-Net requires the ignore label to be the **highest** label and to be named `"ignore"`. If you reorder the labels, ignore silently becomes a real class.

**Planning and preprocessing.**
- Command: `nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncM -np 4 --verify_dataset_integrity`, on the M4.
- Preprocessing took 3 min 57 s. Planning and fingerprinting took a few minutes more.
- Output: 7.1 GB (266 × {`.b2nd` image, `_seg.b2nd`, `.pkl`}). The integrity check passed. The log is at `data/nnunet/logs/plan_and_preprocess.log`.
- **Fingerprint:** intensity mean 112.7, sd 40.3, 0.5–99.5 percentiles 30–212. Spacing (1, 1), so there is no resampling.
- **Normalisation:** z-score per image, because the channel is named "TEM", not "CT".

**Planner output for ResEnc-M 2D:**
- Patch **448 × 576**, batch 13.
- 7 stages with features 32 → 512 and residual blocks [1, 3, 4, 6, 6, 6, 6].
- Instance norm, LeakyReLU.
- The full plans file is in `results/day3/nnUNetResEncUNetMPlans.json`.

### D-24 (new): train on 896 × 1152 patches, batch 3 (configuration `2d_bigpatch`)

**Problem.** A 448 × 576 patch is 0.96 × 1.24 µm. The median fiber is 755 nm across (351 px), and the 95th percentile is 1,279 nm (595 px). So:
- a patch typically holds about one fiber;
- large fibers don't fit vertically;
- the network rarely sees a fiber *together with* the neighbours it touches, which is exactly what separation needs.

nnU-Net sized the patch from its 8 GB VRAM target, not from the size of the objects. The plan expected a patch "on the order of 1024²" (T-08).

**Change.** A second configuration in the same plans file:

```
"2d_bigpatch": {"inherits_from": "2d", "patch_size": [896, 1152], "batch_size": 3}
```

**Why this is a fair change:**
- Pixels per step stay almost the same: 3 × 896 × 1152 = 3.10 M against 13 × 448 × 576 = 3.35 M. So GPU cost per iteration should not rise (to be confirmed by the timing run).
- Network, loss, augmentation and preprocessed data are identical; the data identifier is still `nnUNetPlans_2d`, so nothing is re-preprocessed.
- 896 and 1152 are multiples of 64, as the 6 downsamplings need.
- The **original plans file** is saved in `data/nnunet/logs/nnUNetResEncUNetMPlans.original.json` (local). The edited one is committed under `results/day3/`.

**Cost of the change.**
- Batch 3 instead of 13 means noisier gradients. Instance norm doesn't depend on batch size, so that part is safe.
- This is no longer "out-of-the-box nnU-Net". The write-up must say B1 = nnU-Net ResEnc-M with a patch enlarged to fit the object scale.
- If time allows, a default-patch run is the ablation.

### splits_final.json

Written into `data/nnunet/preprocessed/Dataset501_TEMAxonMyelin/` with 4 folds:

| nnU-Net fold | Source | Train / val |
|---|---|---|
| 0 | dev (v2) | 213 / 53 |
| 1 | cv3 fold 0 | 177 / 89 |
| 2 | cv3 fold 1 | 177 / 89 |
| 3 | cv3 fold 2 | 178 / 88 |

Without this file, nnU-Net builds its own random 5-fold split, which ignores overlap groups and leaks.

### Upload to Modal (local → private Volume)

- **First attempt:** `modal volume put` on the 7.1 GB folder **failed after 2 min 50 s** with `'Connection' object has no attribute '_transport'` and uploaded nothing. **The shell reported success** (exit 0), because the command was piped through `tail`, which returns its own exit code.
- **Fix:** `infra/modal/upload_dir.py`. It uploads 40 files per batch, retries each batch up to 5 times, and skips files already on the Volume with the same size. Two batches hit the same connection error and succeeded on retry.
- **Lesson:** never pipe a long transfer through `tail` or `grep` and trust the exit code. Use `set -o pipefail` or check the destination afterwards.

## T-09 Shared pairing and result R1 — `postprocessing/pairing.py`, `scripts/run/day3_r1_pairing_on_gt.py`

**What the pairing does (D-11):**
1. Seeds are the predicted axon components, with holes filled and ≥ 50 px.
2. The landscape is −EDT of (axon ∪ myelin).
3. Marker watershed inside axon ∪ myelin.
4. Fiber k: axon = seed k, myelin = region k ∩ predicted myelin. Rings without an axon are dropped.

Tests are in `tests/test_pairing.py`.

**R1 = O-pair-GT:** the GT *semantics* (perfect pixels) go through the pairing, and the result is scored against the GT *instances*. Ran on all 266 images: 15 min on the M4, 4 workers. Output: `results/day3/r1_pairing_on_gt.csv`, `r1_summary.json`.

| Metric (all 266 images, 7,968 GT fibers) | R1 | Plan expectation |
|---|---|---|
| Fiber F1@0.5 (95% cluster-bootstrap CI) | **0.99994** (0.99981–1.0) | ≥ 0.95 |
| Fiber F1@0.75 | 0.99994 | — |
| TP / FP / FN | 7,967 / 0 / 1 | — |
| Merges / splits | 0 / 0 | — |
| g-MAE | **0.0017** | "lower myelin IoU on touching fibers" |
| Pairing-correct rate | 0.958 | — |
| Dev-val subset (53 images, 1,619 fibers) | F1 0.9997, g-MAE 0.0015 | — |

**Interpretation, which matters for H1:**
- **With correct pixels, the pairing rule makes essentially no instance errors.** Where two sheaths touch, the watershed line is placed close enough to the annotator's line that every fiber keeps IoU > 0.75.
- The only cost is a small g error (0.0017, a quarter of the human noise floor of 0.0067), from slightly different splits of touching myelin.
- So the plan's worry that "the watershed line between touching sheaths is a guess" is **not** a significant error source on this data.
- Any separation errors a semantic baseline makes must come from **pixel** errors: a missing thin myelin line between two axons, two axons predicted as one blob, or a gap predicted as an axon. They cannot come from the rule.
- This sharpens what O-sep measures: it fixes grouping *given the model's pixels*.

**Why pairing-correct is only 0.958 here:** 181 GT axons touch background (2.3%), and a predicted axon that touches background fails the "enclosed" test by definition. These are inherited from the GT.

**GT violation baseline** (T-05 expectation; checks on the raw GT, 8,015 objects):

| Violation | Count |
|---|---|
| Axon without ring | 4 |
| Ring without axon | 43 |
| Axon touching background | 181 |
| Ring Euler ≠ 0 (not exactly one hole) | 334 |
| **Any violation** | **489 (6.1%)** |

Compare model violation rates with **6.1%**, not with 0.

## T-11 AxonPath zero-shot — `baselines/axonpath/run.py` (local, MPS)

**Setup (D-13):**
- Bundled `em` weights (AxonPath 6d633bf).
- The image is downscaled ×0.269 (2.152 → 8 nm/px) and percentile-normalised (1–99%).
- `run_inference(min_diameter=20)`. Labels are upscaled back with nearest neighbour.
- **Speed:** median 3.4 s per image on the M4's GPU (MPS); 53 images in about 3 min. No cloud GPU was needed.

**Mapping to our encoding:**
- AxonPath's *inner cylinder* (axon + inner tongue) becomes our axon, because our GT counts the tongue as axon.
- Myelin = fibre − inner cylinder.

**Finding:**
- **v1, inner cylinder only:** dropped **834 of 3,277 fibres (25%)** because they had no inner-cylinder pixels. That is unfair to AxonPath.
- **v2, fallback added:** when there is no inner cylinder, AxonPath's own *axon* instance is used instead. This recovered 354. **480 (15%)** still have neither and are dropped. Many are probably its false fibres (dark blobs without a lumen), but this is not verified.
- Logs: `results/day3/axonpath_runlog_v1_inner_only.csv`, `axonpath_runlog.csv`.

**Scores** (v2, dev-val; `results/day4/summary_axonpath.json`, `scores_axonpath.csv`):

| Metric | AxonPath zero-shot |
|---|---|
| Fiber F1@0.5 (95% CI) | **0.784** (0.746–0.824) |
| Fiber F1@0.75 | 0.732 (0.678–0.787) |
| Recall / precision @0.5 | 0.883 / 0.705 |
| Dice axon / myelin (image mean) | 0.714 / 0.621 |
| HD95 axon / myelin (mean) | 1,062 / 977 nm |
| g-MAE (fiber-weighted) / bias | **0.060** / **+0.030** |
| PQ (mean) | 0.708 |
| Pairing correct (matched fibers) | 0.80 |
| Violation rate | 39% (GT: 6.1%) |
| Predictions on GT label 1 (ignored) | 767 |
| Merges / splits | 42 / 19 |

**Error taxonomy** (787 fiber errors = FP + FN at IoU 0.5):

| Unmatched predictions | Count | | Unmatched GT | Count |
|---|---|---|---|---|
| other_bg | **449** | | missed | 96 |
| unlabelled_real | 88 | | merged | 46 |
| low_iou | 42 | | low_iou | 41 |
| split_piece | 11 | | split | 7 |
| merge | 7 | | | |
| gap | 0 | | | |

**Separation share: 9%** (71 of 787).

**Reading:**
- **Detection is good** (recall 0.88). But AxonPath fires on many structures the annotators did not label: 449 + 88 FPs on background, plus 767 on label-1 regions.
- **Pixel accuracy is poor:** Dice 0.71 / 0.62, and g is biased +0.03, i.e. axons are too big relative to fibers.
  - The scale and domain shift (our tissue at 8 nm/px, trained on other EM data) is the likely cause.
  - So is the definition gap: AxonPath's inner cylinder vs our "axon including tongue".
- Its errors are **not** separation errors. It has explicit fibre instances (SDT watershed), and only 9% of errors are merges, splits or pieces.
- **The large `other_bg` class needs inspection** (day 4). It mixes true FPs, such as organelles and blobs, with unlabelled fibers the ring-candidate detector missed. The candidate detector's recall decides how O-sel splits these.

**Easy to miss:**
- AxonPath is far from its own paper's numbers. **That is not evidence against AxonPath**: it is zero-shot at 3.7× its training pixel size, on another lab's tissue and with another annotation convention. It is reported as "off-the-shelf", not as "AxonPath's capability".
- The F1 CI half-width (0.039) is larger than H1's 0.03 threshold. Comparing two conditions must therefore use **paired** bootstrap differences (same resampled groups), as `day4_oracles.py` does, not overlapping marginal CIs.

## T-10 Cellpose-SAM (B3)

Status: see §Budget below.

## Budget and GPU log (Modal, L4)

| Run | GPU-h | Est. cost | Notes |
|---|---|---|---|
| Smoke checks (12–13 Nov) | ~0.03 | ~$0.05 | L40S + L4 |

(Updated as runs complete.)
