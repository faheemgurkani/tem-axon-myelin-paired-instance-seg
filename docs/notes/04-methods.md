# 04 — Methods: how every number is produced

Code paths are relative to the repo root. Decision IDs (D-xx) are in [07](07-plan.md).

## 1. Data handling (day 1)

**Loader** — `src/axon_myelin_seg/data/io.py`:
- `list_samples()` returns the 266 names. `load_sample()` returns (uint8 image, uint16 mask).
- It validates shape (3008, 4096) and dtype, and raises on any mismatch.

**Training-label cleaner** — `data/cleaning.py` (D-21). It sets the following to label 1 (ignore):
- **orphan IDs:** X without X+1, or the reverse;
- **topology-defect fibers:** the ring has more than one 8-connected component, the ring's Euler number ≠ 0, or the axon is not fully inside the filled ring.

**Training only.** Evaluation always uses the raw GT. A "defect" means geometry that contradicts the encoding, not wrong biology; it is ignored rather than relabelled.

**Splits** — `data/splits.py` (D-03, D-25):
- **Groups:** overlap groups v2; every other image is its own group, giving 236 groups.
- **Stratum:** fiber-density tercile of each **group's** mean density. Per-image strata would give one group conflicting labels.
- `StratifiedGroupKFold(shuffle=True, random_state=0)`:
  - fold 0 of 5 is the **dev split** (`configs/splits/dev.json`, 213 train / 53 val);
  - a 3-fold version is `configs/splits/cv3.json`.
- `check_split()` asserts three things: train and val are disjoint, together they cover all images, and **no group crosses the split**.
- **The JSONs are frozen.** Every result is tied to them.

## 2. The evaluator (day 2; contribution C1)

Code: `src/axon_myelin_seg/evaluation/`.
- `official.py`: challenge metrics.
- `diagnostics.py`: instance and pairing checks.
- `taxonomy.py`: why each error happened.
- `runner.py::score_image`: all of the above in one call.
- `stats.py`: bootstrap.
- `oracles.py`.

Tests are in `tests/test_evaluator.py`.

### Official metrics (`official.py`)

| Point | Our rule | Why |
|---|---|---|
| Label 1 | Removed from **both** maps before any metric (D-05) | Spec |
| A predicted fiber on label 1 | If ≥ 50% of its area is on GT label 1, it is *ignored*: neither TP nor FP | The model correctly segments border and excluded fibers; the GT just doesn't score them |
| What a fiber is | Even X ≥ 2 **with** X+1 present. Orphans count for Dice/HD95 but can't match | GT has orphans |
| Matching | One-to-one at fiber IoU > 0.5 on X ∪ X+1 (D-06). Above 0.5 a match is unique, so greedy = Hungarian | Spec gives no threshold |
| g | √(A_axon / (A_axon + A_myelin)), the equivalent-circle ratio | Spec |
| Dice | Per image, per compartment; reported as the mean over images | — |
| HD95 | Per image and compartment: the max of the two directed 95th-percentile boundary distances × 2.152 nm. One side empty → image diagonal 10,936 nm (D-07) | Most common variant |
| g-MAE | Pooled over matched fibers ("fiber-weighted"), always reported with recall (D-08) | Ignores misses |

**Implementation trick:** for any label L ≥ 2, `L // 2` is the same for axon X and ring X+1.
- So `labels // 2` is a fiber-instance map.
- One `np.unique` over (GT fiber, predicted fiber) codes gives every intersection at once.
- That scores a full image in seconds.

### Diagnostics (`diagnostics.py`)

- **F1@0.5 and F1@0.75; PQ = SQ × RQ.**
- **Merge:** one predicted fiber covers ≥ 20% of each of ≥ 2 GT fibers.
- **Split:** ≥ 2 predicted fibers each have ≥ 20% of their own area inside one GT fiber.
- **Pairing correct** (matched fibers only), all of:
  - the predicted axon is one component;
  - it touches neither background nor another fiber;
  - its IoU with the GT axon is ≥ 0.5.
- **Violations** (every scored predicted fiber), any of:
  - axon without ring, or ring without axon;
  - axon touching background;
  - ring Euler number ≠ 0 (not one closed annulus).
- **Checks run on the raw prediction**, not the ignore-masked one. Otherwise every axon next to an ignored region "touches background". This was a bug, fixed on day 3 before any model was scored.

### Error taxonomy (`taxonomy.py`)

Each **unmatched predicted fiber** gets exactly one class; the first matching rule wins.

| Class | Rule | Meaning |
|---|---|---|
| gap | Its axon is ≥ 50% on GT background, and the 5-px band around it is ≥ 75% myelin of ≥ 2 GT fibers | The interstitial-triangle pseudo-fiber |
| unlabelled_real | Its axon is ≥ 50% on GT background and overlaps a **ring candidate** by ≥ 50% | A real fiber the annotators skipped |
| merge | As above | |
| split_piece | ≥ 20% of it lies inside a GT fiber matched by another prediction | |
| low_iou | ≥ 20% of it lies inside a GT fiber nothing matched | Right place, poor outline |
| other_bg | < 20% on any GT fiber, no gap, no candidate | On background (mixed: real unlabelled fibers and true FPs) |

Each **unmatched GT fiber** is classed as missed (< 10% covered), merged, split or low_iou.

**Ring candidate:** an unlabelled dark closed band around a hole, d ≥ 300 nm, at 1/4 resolution (`ring_candidates`).
- Its intensity threshold sits half-way between the GT myelin median and the GT background median. Using the GT is fine for an *evaluator*.
- **Its recall is limited:** it misses many thin-myelin fibers. That is why O-sel has a strict and a broad version.

**Gap is checked first.** A gap prediction overlaps its neighbours' myelin, so checked after merge it would be called a merge.

**Error families** (used by H1):

| Family | Classes |
|---|---|
| **Separation** | pred gap, merge, split_piece; GT merged, split |
| **Selection** | pred unlabelled_real (+ other_bg in the broad view) |
| Boundary | pred and GT low_iou |
| Detection | GT missed |

**Separation share** = separation errors ÷ (FP + FN at IoU 0.5).

### Oracles (`oracles.py`, `scripts/run/day4_oracles.py`)

Each oracle uses the GT to fix **one** error source; the gain over the baseline is that source's cost.

| Oracle | What it does | What stays wrong |
|---|---|---|
| **O-sel** | Deletes predicted fibers classed `unlabelled_real` | All else |
| **O-sel-broad** | Also deletes `other_bg` | All else. An **upper bound**: it also deletes true FPs |
| **O-sep** | Keeps the model's axon/myelin *pixels*, but assigns every predicted pixel inside GT fiber k to fiber k. Predictions outside GT fibers keep their own instances; a model instance mostly (≥ 50%) inside GT fibers drops its outside remnant | Selection, boundary, missed pixels |
| **O-bnd** | Replaces each matched predicted fiber by its GT fiber | Selection, detection. Its g-MAE is 0 by construction |
| **R1 (O-pair-GT)** | GT *semantics* → our pairing rule, scored against GT instances | Measures the rule alone |

O-sep is slightly pessimistic for detection: any GT fiber containing a few predicted pixels becomes a small (FP) fiber.

### Statistics (`stats.py`, D-22)

- **Fiber F1 is micro-averaged:** summed TP/FP/FN over images.
- **CIs come from a cluster bootstrap:** 1,000 resamples of *overlap groups* with all their images, 95% percentile.
- **Differences** between methods or oracle conditions use a **paired** bootstrap: the same resampled groups for both sides, rows aligned by image name.

### Human ceiling method (`scripts/run/day2_ceilings.py`)

1. For each overlapping tile pair: ORB features give an initial transform. ECC refines it (template = A, input = B), then B is warped into A's frame with `WARP_INVERSE_MAP`.
2. Keep ORB or ECC, whichever has the higher image NCC. Use only pairs with NCC ≥ 0.8.
3. Ignore pixels that are label 1 in **either** tile (a fiber cut by B's border is a whole fiber in A) and pixels outside the valid warped window.
4. Score B as a "prediction" of A.

## 3. Shared pairing rule (`src/axon_myelin_seg/postprocessing/pairing.py`, D-11)

Every semantic baseline goes through the same function, so baselines differ only in their pixels:
1. **Seeds** = 8-connected components of the predicted axon, holes filled, ≥ 50 px. Whole components are used, not centroids: a centroid can fall outside a concave axon, which is AxonDeepSeg's failure.
2. **Landscape** = −distance transform of (axon ∪ myelin). Touching fibers split where their sheaths are thinnest.
3. Marker watershed inside axon ∪ myelin.
4. Fiber k: axon = seed k; ring = (region k) ∩ predicted myelin. Fibers with an empty ring are dropped.
5. Output axon 2k, ring 2k+1, as uint16.

## 4. Baselines

### B1 — nnU-Net v2 (`baselines/nnunet/`)

**Conversion** (`convert.py`, local): `Dataset501_TEMAxonMyelin`, PNG with a `_0000` suffix.

| Label | Meaning |
|---|---|
| 0 | background |
| 1 | axon |
| 2 | myelin |
| 3 | ignore = GT label 1 + orphans + defect fibers |

- Label shares: 56.4% / 16.6% / 11.3% / **15.8%** (ignore ranges 0–51% per image).
- **One sixth of all pixels give no training signal.**
- `convert.py splits` writes `splits_final.json`: fold 0 = dev, folds 1–3 = cv3.

**Planning** (local): `nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncM`.
- Z-score per image; no resampling (spacing 1, 1).
- The planner picked patch 448×576, batch 13.

**D-24, the big patch.**
- 448×576 px is about 1 µm, roughly one fiber, so the network would rarely see a fiber *with* the neighbours it touches.
- We added configuration `2d_bigpatch`: patch **896×1152, batch 3**, inheriting everything else.
- The pixels per step are nearly equal, so the cost is the same.
- B1 is therefore "nnU-Net ResEnc-M with a patch enlarged to the object scale", not out-of-the-box nnU-Net.

**Trainers** (`trainers.py`; copied into nnU-Net's package on Modal, because nnU-Net only discovers trainers there):

| Trainer | Use | Behaviour |
|---|---|---|
| `nnUNetTrainer_150epochs_ckpt10_es` | **B1** | 150 epochs; checkpoint every 10 (GPUs are preemptible); early stopping; final validation with the best checkpoint (`--val_best`) |
| `nnUNetTrainer_250epochs_ckpt10_es` | Default for future runs | As above, over 250 epochs |
| `nnUNetTrainer_5epochs_ckpt1` | Timing runs | Skips the final validation |

**Early-stopping rule:**
- Stop when nnU-Net's EMA pseudo-Dice has not improved by more than 0.001 for 30 epochs, and never before epoch 60.
- It follows **Keras semantics**: the reference "best" only moves on a gain of more than 0.001.
- It is *not* nnU-Net's "New best" log line, which reacts to any gain.

**Post-processing** (`scripts/run/day4_postprocess.py`, local): B1's semantic PNGs → the shared pairing → `data/predictions/b1/`.

### AxonPath zero-shot (`baselines/axonpath/run.py`, local MPS, D-13)

1. Bundled `em` weights. Downscale ×0.269 (2.152 → 8 nm/px, the model's training scale) and normalise percentiles 1–99%.
2. `run_inference(min_diameter=20)`, then upsample the labels with nearest neighbour.
3. Map to our encoding: axon = AxonPath's **inner cylinder**, because our axon includes the tongue. If a fibre has no inner cylinder, fall back to its axon instance. Myelin = fibre − axon.
4. 15% of fibres have neither and are dropped.

### Cellpose+B1 (`baselines/cellpose_sam/zeroshot.py` + `day4_postprocess.py`, D-26)

1. Cellpose-SAM `cpsam_v2` zero-shot on the M4 (MPS) at ×1/6, so fibers fall into Cellpose's 7.5–120 px training range.
2. Cellpose tiles the whole image into cell-like regions. Each region is split by **B1's pixels**:
   - axon = the largest B1-axon component inside it, holes filled;
   - ring = B1 myelin inside it, minus the axon;
   - regions without an axon or a ring are dropped.
3. Its pixels equal B1's, so **B1 vs Cellpose+B1 is a controlled grouping comparison**: bottom-up watershed vs top-down regions.

The planned fine-tuned Cellpose (B3) did not fit the budget.

## 5. Day-5 analyses (`scripts/run/day5_analysis.py`)

- **Paired bootstrap differences** between baselines on F1@0.5, F1@0.75 and fiber-weighted g-MAE, plus O-sep vs base on g-MAE.
- **Per matched B1 fiber:** g error, GT fiber diameter, and the topology flags of the *predicted* fiber (ring Euler ≠ 0, axon touching background, axon in more than one component).
- **Gap to the human ceiling.**
