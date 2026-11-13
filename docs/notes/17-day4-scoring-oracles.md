# 17 — Day 4: B1 outcome, scoring (T-12) and oracle experiments (T-13), executed 13 Nov 2026

Part of the day 1–5 execution of [09](09-research-plan.md). Previous: [16](16-day3-baselines.md) (baselines and B1 launch). Next: [18](18-day5-go-no-go.md) (decision).

Everything here is on the **53 dev-val images** (`configs/splits/dev.json`, v2; 1,619 GT fibers), scored by the day-2 evaluator ([15](15-day2-evaluator.md)). CIs are 95% cluster-bootstrap over overlap groups, 1,000 resamples (D-22). Differences between methods or oracle conditions are **paired** (the same resampled groups for both sides).

**Glossary for a reader new to the problem:**
- **Fiber** = one myelinated axon: an axon (label X, even) plus its myelin ring (label X+1).
- **F1@0.5** = fiber detection F1: a predicted fiber counts as found if its fiber region (axon ∪ ring) overlaps a GT fiber with IoU > 0.5.
- **g-ratio** = axon diameter / fiber diameter, from equivalent-circle areas. It is the biological readout, and its error (g-MAE) is computed only on matched fibers.
- **Label 1** = "ignore" in the GT: border-cut fibers, artefacts, etc. A predicted fiber with ≥ 50% of its area on label 1 is ignored, neither TP nor FP.

---

## 1. B1 training outcome (T-08 complete)

| Item | Value | Source |
|---|---|---|
| Trainer / plans / config | `nnUNetTrainer_150epochs_ckpt10_es` / `nnUNetResEncUNetMPlans` / `2d_bigpatch` (896×1152, batch 3, D-24), fold 0 = dev | `data/nnunet/logs/modal_b1_launch.log` |
| Modal app | `ap-AqCMZOMna1npScaK8QcGOT` (L4, 6 cores, 16 GiB) | same |
| Epochs run | **150 of 150**; early stopping did **not** fire (see §1.1) | same |
| Best EMA pseudo-Dice | **0.8743 at epoch 134**; final validation used this checkpoint (`--val_best`) | same |
| Final LR | 1.1e-4 (poly schedule fully annealed) | same |
| Epoch time | mean 146.5 s (epoch 0: compile), steady ~145 s | same |
| Container wall time | 23:38 → 05:58 UTC = **6.34 h**, 1 attempt (no preemption), 7 h cap not reached | `data/nnunet/results_modal/runs/budget/…150epochs….json` |
| GPU utilisation | mean 94%, median 100%, ≥ 50% busy in 97% of 15-s samples; memory 9.2 GB of 22 GB | `data/processed/modal_runs/…150epochs….csv` (1,521 samples) |
| Validation inference | 53 full images at ~6.4 s/image, ~7 min | launch log |
| nnU-Net's own validation Dice | **axon 0.882, myelin 0.861** (mean 0.871) | `…/fold_0/validation/summary.json` |
| **Cost (Modal billing)** | **$7.69** ($0.44 on 12 Nov UTC + $7.25 on 13 Nov UTC) | `modal billing report` |

Learning curve (pseudo-Dice axon / myelin on nnU-Net's random validation patches; best EMA so far):

| Epoch | 0 | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 | 149 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Axon | 0.40 | 0.85 | 0.88 | 0.89 | 0.90 | 0.88 | 0.87 | 0.86 | 0.88 | 0.88 | 0.91 |
| Myelin | 0.70 | 0.82 | 0.83 | 0.85 | 0.84 | 0.84 | 0.84 | 0.83 | 0.84 | 0.85 | 0.86 |
| Best EMA | — | 0.737 | 0.821 | 0.842 | 0.851 | 0.859 | 0.859 | 0.866 | 0.871 | 0.874 | 0.874 |

Per-epoch pseudo-Dice is noisy (±0.02 between consecutive epochs), because it comes from a few random patches. That is why early stopping uses the EMA, never the raw value.

### 1.1 Why early stopping did not fire (easy to miss)

**What happened:** the EMA's printed "New best" went from 0.8594 (epoch 73) to 0.8658 (epoch 104), about 30 epochs apart, yet training continued.

**Why:**
- Our rule (`baselines/nnunet/trainers.py::epochs_since_improvement`) counts an epoch as an improvement only if it beats the **reference best by more than `min_delta` = 0.001**.
- The reference is updated only on such an improvement. These are Keras `EarlyStopping` semantics.
- nnU-Net's own "New best EMA" message counts *any* gain, however small.
- So the two disagree. Epoch 73 was < 0.001 above epoch 72, so the reference stayed at epoch 72. Around epoch 92 the EMA cleared epoch 72 + 0.001 (while still below the epoch-73 value), which reset our counter.

**Check:** replaying the rule on the EMA rebuilt from the logged pseudo-Dice (`ema ← 0.9·ema + 0.1·mean(dice)`) gives a longest stale stretch of **19 epochs**, never 30.

**Conclusion:** the rule worked as written. Since the schedule anneals the LR to ~0 by epoch 150, the extra epochs were not wasted: the best checkpoint is from epoch 134, deep in the low-LR phase.

**Lesson:** don't judge the early stop by nnU-Net's "New best" lines; replay the rule on the EMA. Stopping earlier would have needed a lower patience, which a 150-epoch poly schedule doesn't need.

### 1.2 What was downloaded from the Volume (and what was not)

Downloaded to `data/nnunet/results_modal/nnUNet_results/Dataset501_TEMAxonMyelin/nnUNetTrainer_150epochs_ckpt10_es__nnUNetResEncUNetMPlans__2d_bigpatch/fold_0/` (git-ignored):
- `validation/*.png` (53 semantic maps: 0 bg, 1 axon, 2 myelin) and `summary.json`;
- `training_log_*.txt`, `progress.png`, `debug.json`.

Downloaded to `data/nnunet/results_modal/runs/`: the GPU-utilisation CSVs and budget ledgers.

**Not downloaded:**
- `validation/*.npz`: softmax probabilities, 6.8 GB.
- `validation/*.pkl`: 237 MB.
- The checkpoints.

They stay on the private Volume. Probabilities are only needed if a later step thresholds or calibrates them (e.g. D-18's "B1 probability ≥ 0.9"). Fetch them then.

Gotchas:
- `modal volume get` of a whole directory would have pulled the 6.8 GB of npz. Files were fetched one by one, 8 in parallel.
- 2 of 53 downloads failed transiently in the parallel batch and succeeded on retry. **Always count the files after a download** (53/53 checked).

---

## 2. Environment incident #2: the venv launchers still pointed at the old repo copy (fixed)

**Symptom:** `scripts/run/day4_postprocess.py` crashed with `NotImplementedError` inside `postprocessing/pairing.py`, a file that is fully implemented.

**Cause:**
- The traceback path showed `~/Local Projects/…/src/axon_myelin_seg/postprocessing/pairing.py`, the **old copy** of the repo, where `pairing.py` is still a stub.
- The day-1 fix ([14](14-day1-data-plumbing.md)) re-pointed the *editable installs*, but not the venv's launchers.
- `uv` writes the absolute venv path into `.venv/bin/activate` and into the shebang of every console script (`pytest`, `modal`, `nnUNetv2_*`, …).
- After the move, 109 of 123 files in `.venv/bin` still pointed at `~/Local Projects/…/.venv/bin/python`. `source .venv/bin/activate` therefore activated the **old** venv, whose editable install points at the old `src/`.

**Why earlier results are not affected:**
- Days 1–3 were run with `.venv/bin/python` explicitly (the new interpreter).
- In the old copy, `official.py`, `splits.py` and `pairing.py` are `NotImplementedError` stubs, and `cleaning.py`, `taxonomy.py`, `oracles.py`, `stats.py` and `runner.py` don't exist. Any accidental run on the old venv would have **crashed**, not returned wrong numbers.
- Also checked: `.venv/bin/modal` ran under the old venv's Python. That is harmless, because the Modal app mounts the repo from the path of `tem_modal.py`, which is the new one, and the image pins its own packages.

**Fix:**
- Replaced the old absolute path with the new one in every text file in `.venv/bin` (109 files).
- Verified:
  - `source .venv/bin/activate; which python` gives the new path;
  - `axon_myelin_seg.__file__` and `axonpath.__file__` are under `/Volumes/Sandisk 2TB/…`;
  - `.venv/bin/pytest -q tests` gives 20 passed;
  - `.venv/bin/modal --version` works.

**Lesson (extends day 1):** after moving a repo, a venv is not relocatable. Either recreate it or rewrite the paths in `.venv/bin/*`, then check `which python` *after* activating. Run `pytest tests`, not bare `pytest`: bare `pytest` also collects `external/nnUNet`'s own tests, which error out.

---

## 3. Post-processing B1 into fibers — `scripts/run/day4_postprocess.py` (local CPU, 79 s for 53 images)

| Output (git-ignored) | What |
|---|---|
| `data/predictions/b1_sem/*.png` | B1 semantics (copy), used by O-sep |
| `data/predictions/b1/*.tif` | **B1** = semantics → shared watershed pairing (D-11, `postprocessing/pairing.py`) |
| `data/predictions/cellpose_b1/*.tif` | **Cellpose+B1** = Cellpose-SAM zero-shot regions (day 3), each split into axon/ring by B1's semantics (D-26) |

B1 and Cellpose+B1 have **identical pixels** (B1's semantics). They differ only in how pixels are grouped into fibers: a bottom-up watershed from axon seeds vs. top-down whole-fiber regions. That makes their comparison a controlled grouping experiment.

---

## 4. T-12 Scores — `scripts/run/score_predictions.py`

Outputs:
- `results/day4/scores_<name>.csv`: per image, about 50 columns;
- `results/day4/summary_<name>.json`;
- `results/day4/score_<name>.log`.

| Metric | Human ceiling (day 2, 28 overlap pairs) | AxonPath zero-shot | Cellpose+B1 | **B1** |
|---|---|---|---|---|
| **Fiber F1@0.5** [95% CI] | 0.929 | 0.784 [0.746, 0.824] | 0.885 | **0.904 [0.881, 0.928]** |
| Fiber F1@0.75 | 0.929 | 0.732 | 0.869 | **0.881 [0.854, 0.910]** |
| Recall / precision @0.5 | — | 0.883 / 0.705 | **0.969** / 0.814 | 0.963 / **0.852** |
| TP / FP / FN | 118 / 7 / 11 | 1,429 / 597 / 190 | 1,569 / 359 / 50 | 1,559 / 271 / 60 |
| Predicted fibers on label 1 (ignored) | — | 767 | 773 | 752 |
| Dice axon / myelin (mean over images) | 0.987 / 0.951 (median) | 0.714 / 0.621 | 0.879 / 0.857 | **0.882 / 0.859** |
| HD95 axon / myelin (nm, mean) | 9 / 644 (median) | 1,062 / 977 | 635 / 552 | 629 / 555 |
| **g-MAE** (fiber-weighted) | **0.0067** | 0.060 | **0.0114** | 0.0160 |
| g bias (pred − GT) | −0.0005 | +0.030 | +0.003 | −0.0002 |
| PQ (mean) | — | 0.708 | 0.843 | **0.855** |
| Merges / splits | 0 / 0 | 42 / 19 | 1 / 3 | 4 / 2 |
| Pairing-correct (matched fibers) | 1.00 | 0.80 | **0.88** | 0.87 |
| Violation rate (predicted fibers) | 0.15 | 0.39 | 0.50 | 0.37 |

Paired differences (`results/day5/paired_differences.json`, produced by `scripts/run/day5_analysis.py`):

| Comparison | ΔF1@0.5 [CI] | ΔF1@0.75 [CI] | Δg-MAE [CI] |
|---|---|---|---|
| B1 − AxonPath | **+0.120** [0.096, 0.142] | +0.149 [0.110, 0.187] | −0.044 [−0.056, −0.033] |
| B1 − Cellpose+B1 | **+0.019** [0.012, 0.028] | +0.012 [0.003, 0.021] | **+0.0046** [0.0031, 0.0061] (B1 *worse*) |
| Cellpose+B1 − AxonPath | +0.101 [0.077, 0.123] | +0.137 [0.098, 0.176] | −0.049 [−0.060, −0.038] |

**Reading the table:**
1. **B1 is a strong baseline.**
   - F1@0.5 = 0.904 is within 0.025 of the human-vs-human ceiling (0.929). The ceiling itself comes from only 129 fibers in 28 overlap pairs, so it is uncertain.
   - B1 is also far above ASTIH's published TEM1 number (0.791, another dataset). That is **not** a like-for-like comparison: different data, scale and annotation protocol.
2. **The expectation in T-10 ("Cellpose: higher precision, lower recall than B1, as in ASTIH") was wrong here.** Cellpose+B1 has *higher recall and lower precision*. Zero-shot Cellpose partitions the whole image into regions, so it proposes a region for every ring-like structure. Any region that contains B1-axon and B1-myelin becomes a fiber. (ASTIH's Cellpose was *trained* on fibers; ours is zero-shot.)
3. **Same pixels, different grouping → different g-ratio.** Cellpose+B1's g-MAE is 29% lower than B1's (0.0114 vs 0.0160, CI excludes 0), with the same pixels. So the watershed's grouping costs g-ratio accuracy even though it costs almost nothing in detection (§5, O-sep).
4. **The violation rate is not an error count**, and 0.37 does not mean "37% of fibers are wrong" (§6). Even the human annotations show 0.15 on the overlap crops, because crops cut rings.

**B1's Dice is below the plan's expectation, and the reason is selection, not pixel quality.**
- T-08 expected val Dice axon ≥ 0.95 and myelin ≥ 0.88; measured 0.882 and 0.861.
- Image-level Dice counts the pixels of every predicted fiber the annotators left unlabelled as FP pixels.
- Deleting the unmatched background predictions (O-sel-broad) lifts Dice to **0.948 / 0.908**. Replacing matched fibers by GT (O-bnd) only reaches 0.896 / 0.914.
- So about two-thirds of the axon-Dice shortfall is selection. Myelin is split between selection and boundary placement.
- D-24's fallback ("B1 Dice far below the ceiling → run the default patch as an ablation") is therefore **not triggered**: the patch size is not the cause. Logged in the plan.
- The "If not" branch of T-08 (myelin < 0.85 → check normalisation) is not triggered either (0.861).

**Easy to miss:**
- **`n_pred` excludes predictions on label 1.** B1 output 2,582 fibers: 1,830 are scored and 752 lie on label 1. Label 1 covers 15.8% of pixels (border fibers and artefacts), and the model correctly segments the fibers there. So those 752 are mostly *not* errors, and the evaluator correctly ignores them. Precision computed over all 2,582 would be badly wrong.
- **Our evaluator agrees with nnU-Net's.** Our mean axon Dice for B1 (0.8822) equals nnU-Net's validation Dice (0.8823). Myelin is 0.8593 vs 0.8607; nnU-Net's ignore label also covers the orphan and defect fibers that we clean in training but keep in evaluation (D-21). This is an independent cross-check of D-05 (ignore handling) in our Dice.
- **HD95 is image-level and dominated by extra or missing objects**, not by boundary placement. O-sel-broad (deleting background FPs) cuts B1's myelin HD95 from 555 to 305 nm, while O-bnd (perfect boundaries on matched fibers) only reaches 541 nm. Never read HD95 as a "boundary accuracy" number here.
- **g-MAE ignores missed fibers**, so always read it next to recall (D-08). The three baselines' recalls differ (0.88 / 0.97 / 0.96). The B1 vs Cellpose+B1 g comparison is fair: recalls 0.963 vs 0.969.

---

## 5. T-13 Oracle experiments — `scripts/run/day4_oracles.py`

Outputs:
- `results/day4/oracles_<name>.csv`: per image × condition;
- `results/day4/oracles_<name>.json`: pooled values with paired gains;
- `results/day4/oracles_<name>.log`.

Each oracle uses the GT to fix **one** error source and leaves the rest of the prediction alone. Its gain is the cost of that source. Definitions are in `src/axon_myelin_seg/evaluation/oracles.py`:
- **O-sel:** delete predicted fibers classed `unlabelled_real` (a ring-like structure the annotators left as background).
- **O-sel-broad:** also delete `other_bg` (unmatched predictions on background with no ring candidate).
- **O-sep:** keep the model's axon/myelin pixels, but group every pixel inside a GT fiber with that GT fiber. Predictions outside GT fibers are unchanged.
- **O-bnd:** replace every matched predicted fiber by its GT fiber, pixel for pixel.
- **O-pair-GT (= R1, day 3):** GT pixels through our pairing rule. F1 0.99994 and g-MAE 0.0017, so the rule itself is lossless on perfect pixels.

### B1

| Condition | F1@0.5 | Gain [paired CI] | F1@0.75 gain [CI] | TP / FP / FN | Dice A / M | g-MAE |
|---|---|---|---|---|---|---|
| base | 0.904 | — | — | 1,559 / 271 / 60 | 0.882 / 0.859 | 0.0160 |
| O-sel (strict) | 0.920 | **+0.016** [0.009, 0.024] | +0.016 [0.009, 0.024] | 1,559 / 210 / 60 | 0.899 / 0.873 | 0.0160 |
| O-sel-broad | 0.978 | **+0.074** [0.051, 0.095] | +0.072 [0.050, 0.092] | 1,559 / 10 / 60 | 0.948 / 0.908 | 0.0160 |
| **O-sep** | 0.903 | **−0.001** [−0.006, +0.003] | +0.006 [−0.001, +0.013] | 1,572 / 292 / 47 | 0.888 / 0.878 | **0.0093** |
| O-bnd | 0.904 | 0 (by construction) | +0.023 [0.015, 0.034] | 1,559 / 271 / 60 | 0.896 / 0.914 | 0 (by construction) |

### Error taxonomy of B1 (331 errors at IoU 0.5 = 271 FP + 60 FN)

| Class | Count | Share | Meaning |
|---|---|---|---|
| pred `other_bg` | 200 | 60% | unmatched prediction on background, no ring candidate there |
| pred `unlabelled_real` | 61 | 18% | unmatched prediction on a ring-like structure the annotators left unlabelled |
| pred `low_iou` | 10 | 3% | overlaps a GT fiber, but IoU ≤ 0.5 |
| pred `gap` / `merge` / `split_piece` | 0 / 0 / 0 | 0% | **separation errors** |
| GT `missed` | 46 | 14% | GT fiber with no prediction overlapping it |
| GT `low_iou` | 12 | 4% | found, but IoU ≤ 0.5 |
| GT `merged` / `split` | 2 / 0 | 0.6% | **separation errors** |

**Separation share = 2 / 331 = 0.6%.** For AxonPath it is 9.0% (71/787) and for Cellpose+B1 1.0% (4/409).

### Other baselines (same oracles)

| Baseline | Base F1@0.5 | O-sel gain | O-sel-broad gain | O-sep gain | O-bnd F1@0.75 gain | Sep. share |
|---|---|---|---|---|---|---|
| AxonPath | 0.784 | +0.019 [0.013, 0.028] | +0.136 [0.108, 0.165] | +0.015 [0.008, 0.022] | +0.055 [0.033, 0.079] | 9.0% |
| Cellpose+B1 | 0.885 | +0.018 [0.011, 0.026] | +0.091 [0.068, 0.113] | +0.000 [−0.001, +0.002] | +0.016 [0.009, 0.024] | 1.0% |
| **B1** | 0.904 | +0.016 [0.009, 0.024] | +0.074 [0.051, 0.095] | −0.001 [−0.006, +0.003] | +0.023 [0.015, 0.034] | 0.6% |

**What the oracles say:**
1. **Separation/pairing costs B1 essentially nothing in detection.** O-sep's gain is −0.001, with a CI that includes 0 and excludes anything above +0.003. B1 makes 2 merges, 0 splits and 0 interstitial-gap pseudo-fibers among 1,619 GT fibers. The prior expectation was "a modest gain concentrated in touching fibers". It was **wrong for B1**: the 896×1152 patch (D-24), with touching neighbours in view, plus the −EDT watershed already separate touching fibers.
2. **Selection dominates.** 78% of B1's errors are unmatched predictions (`other_bg` + `unlabelled_real`).
   - Deleting just the confident unlabelled-real ones gains +0.016.
   - Deleting all background FPs gains +0.074, which brings F1 to 0.978, *above* the human ceiling.
   - The broad oracle is an upper bound: it also deletes true FPs that a perfect model would not produce. The FP gallery (§7) shows what those FPs are.
3. **Boundary placement is what decides IoU-0.75 matches and g-ratio.** O-bnd adds +0.023 at F1@0.75 (fibers matched at 0.5 but not 0.75) and, by construction, takes g-MAE to 0.
4. **But grouping *does* matter for g-ratio.** O-sep cuts B1's g-MAE from 0.0160 to 0.0093. Paired difference −0.0067 [−0.0083, −0.0050], i.e. −42%. For Cellpose+B1 the cut is 0.0114 → 0.0093 (−0.0021 [−0.0028, −0.0013]).
   - O-sep's g-MAE is the same (0.0093) for both, as it must be: it uses the same B1 pixels with GT grouping.
   - So with B1's pixels: **0.0093 is the floor for perfect grouping**. The watershed adds 0.0067 above it and Cellpose regions add 0.0021.
   - The remaining gap from 0.0093 to the human 0.0067 is pixel error.

**Why O-sep's FP count *rises* (271 → 292) while its TP rises 1,559 → 1,572:**
- O-sep regroups predicted pixels by GT fiber. Any GT fiber that contains *some* predicted foreground becomes a predicted fiber, even a tiny fragment (IoU < 0.5 → FP).
- A model fiber that was mostly outside GT fibers keeps its outside part as a separate fiber, so one prediction can become two objects.
- The net F1 change is ≈ 0. O-sep is therefore a slightly pessimistic oracle for detection. Its gain at IoU 0.75, where fragments matter less, is +0.006 (CI includes 0). Either way, the rule's 0.03 threshold is out of reach.

**Why O-sel-broad can exceed the human ceiling:**
- The ceiling counts *both* annotators' selection disagreements on the same tissue.
- The oracle deletes *every* background FP using the GT itself, so it assumes the single GT annotator is right about every fiber left unlabelled.
- It is a bound, not an achievable score.

---

## 6. Ring topology of the predicted fibers (new diagnostic; `results/day5/fiber_analysis_b1.json`; the per-fiber table `results/day5/fibers_b1.csv` is git-ignored, as a per-fiber derivative of the GT)

**Background:** a correct paired fiber is an axon *inside* a single closed ring. The checks (`evaluation/diagnostics.py::fiber_checks`, run on the raw prediction):
- **ring Euler ≠ 0:** the predicted ring is not exactly one closed annulus. It is broken into a "C", has extra holes, or is in several pieces.
- **axon touches background:** the ring has a gap through which the axon touches background.
- **axon in > 1 component.**

**B1's 1,559 matched fibers:**

| | Value |
|---|---|
| Matched fibers with ≥ 1 topology defect | **28.1%** (ring Euler ≠ 0: 26.2%, axon touches bg: 11.1%, multi-CC axon: 0%) |
| g-MAE, fibers with a defect | **0.032** |
| g-MAE, clean fibers | **0.0098** (human ceiling 0.0067) |
| Share of B1's total absolute g error carried by defective fibers | **56%** |
| Defect share by fiber diameter quartile (300–638 / 638–753 / 753–921 / 921–1,916 nm) | 16% / 22% / 30% / **45%** |
| g-MAE by diameter quartile | 0.017 / 0.016 / 0.016 / 0.014 |
| g bias by diameter quartile | −0.003 / −0.003 / +0.001 / +0.004 |
| Median fiber IoU of matched fibers | 0.963 |
| Fibers with \|g error\| > 0.05 | 7.0% |

**Readings:**
- Topologically broken rings, rather than mis-separated fibers, are the pairing-related weakness of B1.
- They are concentrated in **large** fibers, whose rings are long and thin relative to the patch, and they carry more than half of the g error.
- B1's *clean* fibers are already close to the human g floor: 0.0098 vs 0.0067.
- The g error is *not* concentrated in small fibers (MAE is flat across quartiles). This contradicts the common assumption that g error is a small-fiber problem; it holds here because fibers are large (median 755 nm = 351 px).

**Caveat (correlation, not cause):** defective rings may sit on hard fibers (oblique, abnormal, poorly contrasted myelin), where any method would err. This table does not show that fixing the topology would fix the g error. It shows *where* the error is.

**Why the violation rates in §4 look alarming but aren't errors:**
- The rate is computed over all scored predicted fibers, including FPs.
- The human ceiling's 0.15 comes from crops, which cut rings at crop edges.
- On the full GT the rate is 6.1% (R1, day 3), mostly border-cut and C-shaped annotations.
- Treat it as a topology *diagnostic*, compared between methods, not as an accuracy.
- Cellpose+B1 has the highest rate (0.50). Its regions cut rings along region borders that don't follow the ring, yet its g-MAE is better. So the violation rate and g error are not the same thing.

---

## 7. Visual check of B1's background FPs (`data/processed/day4/b1_other_bg_gallery.png`, local, 16 random examples)

Made with `scripts/run/day4_fp_gallery.py --pred-dir data/predictions/b1 --cls other_bg --n 16 --seed 0`.
- **Left** of each pair: the image crop.
- **Right:** the prediction outline (white), GT fibers (green), GT label 1 (purple).

| Judgement | n | Examples (sample, fiber k) |
|---|---|---|
| Real myelinated fiber left unlabelled; most have **thin** myelin | ~7 | 0073 k15, 0205 k28, 0215 k35, 0172 k11, 0172 k27, 0211 k45, 0007 k4 |
| Ragged or abnormal profile (irregular wrapping, large oblique or degenerating fiber); a judgement call | ~7 | 0215 k30, 0048 k21, 0037 k17, 0037 k1, 0211 k26, 0251 k30, 0251 k11 |
| True FP: a fragment or sliver | ~2 | 0073 k36, 0205 k31 |

The day-3 AxonPath check found the same split: ~7 real, ~3 FP, ~2 ambiguous out of 12.

**Consequences:**
- About half of the "background FPs" are fibers a human could defend labelling. Most of the rest are the **abnormal fibers that the annotation protocol deliberately leaves as background** (EDA [10](10-eda.md): ~25% of ring-like structures).
- So the selection error mixes two things:
  - (i) random annotation misses, which are not the model's fault;
  - (ii) a *learnable* protocol rule ("don't label thin-myelin or abnormal fibers") that B1 has not fully learned.
- The `unlabelled_real` detector (ring candidates, d ≥ 300 nm) misses many of the thin-myelin cases. That is why strict O-sel (+0.016) understates selection, and why the truth lies between strict and broad.

**Easy to miss:** the taxonomy's `other_bg` does **not** mean "the model hallucinated". Never report it as a hallucination rate.

---

## 8. Status

- [x] T-08: B1 trained (150 epochs, best epoch 134, $7.69).
- [x] T-12: B1, Cellpose+B1 and AxonPath scored with CIs.
- [x] T-13: oracles on all three. H1 flags are in each `oracles_<name>.json`; the decision is in [18](18-day5-go-no-go.md).
- [~] T-10: B3 remains zero-shot (D-26). Only $1.84 of the $10 is left after B1, so no fine-tune.

**Reproduce (all local CPU; ≈ 25 min in total on the M4):**

```bash
P=.venv/bin/python; V=data/nnunet/results_modal/nnUNet_results/Dataset501_TEMAxonMyelin/nnUNetTrainer_150epochs_ckpt10_es__nnUNetResEncUNetMPlans__2d_bigpatch/fold_0/validation
$P scripts/run/day4_postprocess.py --val-dir $V
for n in b1 cellpose_b1; do $P scripts/run/score_predictions.py --pred-dir data/predictions/$n --name $n; done
$P scripts/run/day4_oracles.py --pred-dir data/predictions/b1 --name b1 --sem-dir data/predictions/b1_sem
$P scripts/run/day4_oracles.py --pred-dir data/predictions/cellpose_b1 --name cellpose_b1 --sem-dir data/predictions/b1_sem
$P scripts/run/day5_analysis.py
```
