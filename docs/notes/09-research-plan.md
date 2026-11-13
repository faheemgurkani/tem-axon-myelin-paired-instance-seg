# 09 — Research Plan (v2, revised 12 Nov 2026)

**How to use this file.** Every decision has an ID (D-xx) with its reason, the evidence it rests on, the outcome we expect,
and what would make us revisit it. Every task (T-xx) has its steps, technical details, why it exists, expected
outcome, a "done when" check and a fallback. When an outcome differs from the expectation, record it in the **Log** at the bottom
and update the decision. Do not silently edit the expectation.

Sources referred to below:
- report A = `docs/reports/TEM axon myelin segmentation review.md` (11 Nov)
- report B = `docs/reports/Paired axon myelin gap validation.md` (12 Nov)
- [10](10-eda.md) and [11](11-deep-eda.md) = EDA notes

Time budget: **10 working days**. We are not bound by the challenge timeline.

---

## 1. Domain primer (what the task really is)

- **Myelinated fiber.** A nerve fiber (axon) wrapped by a myelin sheath: many tightly packed membrane layers made by an oligodendrocyte
  in the brain. In TEM cross-sections an axon is a light, granular disc. Its myelin is a **dark, electron-dense ring** around it.
  Between them sits a thin inner tongue / periaxonal space. In *this* dataset that space is counted as **axon** ([10](10-eda.md) §4).
- **g-ratio** = inner diameter / outer diameter = axon diameter / fiber diameter. It is the standard measure of how well myelinated a fiber is.
  - The challenge computes it from areas: g = √(A_axon / A_fiber), with A_fiber = A_axon + A_myelin, which is the equivalent-circle form.
  - Healthy mouse corpus callosum: g ≈ 0.7–0.8. Our GT median is 0.750 (IQR 0.711–0.790).
  - g rises with axon size (r = 0.65 here).
  - **Remyelinated** fibers have abnormally thin myelin (g > 0.85). **Degenerating** fibers show dark axoplasm, vacuoles and split myelin.
  - All of these occur in this dataset, consistent with a demyelination/remyelination model, possibly cuprizone ([11](11-deep-eda.md) §C).
- **Why pairing matters.** Every morphometric number (g, myelin thickness, axon diameter per fiber) needs to know *which* ring belongs to *which*
  axon. Pixel-level Dice can be near-perfect while fibers are merged or split. When two sheaths touch, a semantic map shows one continuous dark
  region, and the line between the two sheaths has to be inferred.
- **Why touching fibers are hard.** 71% of fibers touch a neighbour; 14% share more than a quarter of their outline ([10](10-eda.md) §2). The sheath–sheath
  contact has almost no intensity contrast. The light triangular gaps where three or more sheaths meet look like an axon inside a ring
  ([11](11-deep-eda.md) §B2).
- **Sensitivity of g.** One pixel of boundary shift on the axon changes g by about g/r_px.
  - Median axon radius is 131 px, so Δg ≈ 0.006 per px here.
  - It is ≈0.015 for a 0.2 µm axon, but only 2.4% of our axons are under 300 nm.
- **Metrics (challenge spec, [01](01-challenge-spec.md)):**
  - Dice for axon and myelin
  - HD95 in nm (95th-percentile boundary distance: how far off the worst 5% of boundary is)
  - g-MAE over fibers matched by IoU on X∪X+1
  - combined by aggregate rank, ties broken by HD95

  g-MAE ignores missed fibers, so it rewards predicting fewer, easier fibers (report A, "gaming").

## 2. What we now believe (evidence → consequence)

| # | Evidence (source) | Consequence |
|---|---|---|
| E1 | ASTIH's TEM1 fiber F1@0.5 = 0.791 measures fiber *detection* after watershed. It never checks axon↔ring assignment. The same pipeline scores 0.949 on TEM2 (report B §1) | "Pairing is the bottleneck" is a **hypothesis (H1)**, not a premise |
| E2 | AxonPath (Sep 2026; models on Zenodo 22982902, code paucabar/axonpath) predicts semantics + fibre-SDT + axon-SDT in one U-Net, then links axon→fibre by a ≥90% overlap rule (report B) | Joint prediction is not novel. AxonPath is a **required baseline** |
| E3 | Nested polar contours exist for single objects (Alblas 2022; NPS-Net 2026, r_cup = α·r_disc). Bi-Polar Mask (ICIP 2022) pairs cell + nucleus contours via a shared centre. Ratio losses exist (Meng MICCAI 2022) (report B) | Dual-radius head and g-ratio loss are **ablations, not claims** |
| E4 | GT pairs are clean: 1 axon in 1 ring, 99.5% enclosed, axon never touches another fiber; 3.7% of fibers have topology defects ([10](10-eda.md)) | The nested-by-construction output format matches GT exactly. Defects are label noise |
| E5 | ~4–8% of real fibers are unlabelled (≈1.1 large ones per image, randomly missed); interior label 1 marks large or oblique fibers ([11](11-deep-eda.md) §B) | **Selection** is a hidden error source. Background ≠ clean negatives |
| E6 | Overlapping tiles annotated twice: axon Dice 0.98–0.99, myelin 0.90–0.97, per-fiber Δg 0.004–0.010 ([11](11-deep-eda.md) §C) | Ceilings: myelin Dice ~0.93–0.96, g-MAE floor ~0.006 |
| E7 | g definitions differ by 0.065 on average (eq-circle 0.750, minor axis 0.725, Feret 0.790, thickness 0.763) | The evaluator must use exactly the equivalent-circle form |
| E8 | 22 overlapping tile pairs → 18 groups, 40 images (`configs/splits/overlap_groups.json`) | Group-aware splits are mandatory |
| E9 | Noise σ 7.5–22; illumination range up to 64%; black corners in 10 images; fiber density 0.11–1.47 /µm² ([11](11-deep-eda.md) §D) | Normalisation, augmentation and density-aware sampling |
| E10 | Challenge data licence unstated ([08](08-open-questions-and-risks.md)) | Headline results must be reproducible on public **ASTIH TEM1** |

## 3. Research question, contributions, hypotheses

**Question.** In paired axon–myelin instance segmentation of TEM, how much of the fiber-level error comes from:
- **(a) selection:** which fibers annotators chose to label
- **(b) separation/pairing:** grouping pixels into the right fibers and linking each axon to its ring
- **(c) boundary placement**

once annotation noise is accounted for? And does a decoder that makes each fiber's axon and ring nested *by construction* remove (b)?

| ID | Contribution | When delivered | Expected outcome |
|---|---|---|---|
| C1 | **Error-decomposition evaluation** for paired fibers. Comprises: pairing-correctness metric; FP/FN taxonomy; oracle experiments separating (a)/(b)/(c); annotation-noise ceilings from overlapping tiles; recall-vs-g-MAE gaming curve; g-definition sensitivity | Always | A table that says, per baseline, what share of fiber errors is selection vs separation vs boundary. No published benchmark has this (report B) |
| C2 | **Nested-by-construction decoder** (instance-first, part-second) | Only if H1 = GO | Zero pairing violations by construction; fewer merge/split and interstitial-gap FPs than nnU-Net+watershed and AxonPath, at equal Dice |
| C3 | **Learning with incomplete fiber annotation** (ignore-mask confident unlabelled rings during training) | Ablation if GO; main method if NO-GO | Higher recall of real-but-unlabelled fibers (overlap-disagreement set) with official metrics unchanged within CI |

**Hypotheses** (each with a decision rule):
- **H1, go/no-go on day 5:** on the B1 baseline, fixing separation/pairing with an oracle recovers ≥ 0.03 absolute fiber F1@0.5,
  *and* at least as much as fixing selection with an oracle, *and* separation errors (merge + split) are ≥ 30% of fiber-level errors.
  - **Why these numbers:**
    - 0.03 is about 2× the expected bootstrap CI half-width for ~50 validation images × ~30 fibers. This is an estimate; confirm with the actual CI from T-07.
    - 30% means pairing is a major, not marginal, error class.
  - **Expectation:** borderline. ASTIH suggests most loss is spurious fibers and mid-IoU delineation (report B), and our EDA suggests selection is large.
    So NO-GO is a real possibility, and the plan is built so NO-GO still yields a paper.
- **H2:** C2 reduces (merge + split + interstitial-gap FP) count by ≥ 30% relative to B1 and to AxonPath, with axon/myelin Dice within the B1 CI.
- **H3:** C3 raises recall on the overlap-disagreement fibers (fibers labelled in one tile, background in the other) by ≥ 10 points,
  and moves no official metric outside the B1 CI.
- **H4:** the ranking of methods (B1 vs C2/C3) on challenge data is the same on ASTIH TEM1.
- **H5 (added 13 Nov, post hoc, NOT pre-registered; see [18](18-day5-go-no-go.md) §3):** grouping/ring topology is a main source of *g-ratio* error even when it is negligible for detection.
  - Evidence on B1: O-sep cuts g-MAE 0.0160 → 0.0093 (−42%). Instance-first grouping with the same pixels beats the watershed on g by 0.0046 [0.0031, 0.0061]. Fibers with broken predicted rings carry 56% of the g error.
  - Status: a candidate for the next phase. It needs an owner decision and a fresh, pre-registered rule before it can drive any method choice.

## 4. Decision register

| ID | Decision | Reason / evidence | Expected outcome | Revisit if |
|---|---|---|---|---|
| D-01 | Evaluation-first; the method is conditional on H1 | E1–E3: the method's novelty is narrow, and the premise is untested | A publishable result in every branch | Never (structural) |
| D-02 | Primary data = challenge train set (266 images); external = ASTIH TEM1 only; AxonPath data deferred | E10. TEM1 has a published reference (Dice 0.974/0.924, F1 0.791/0.882) for validating our pipeline. AxonPath's ROI-only labels and Axon-vs-InnerCylinder convention cost conversion time | Pipeline validated against a published number; results publishable even if the challenge licence blocks us | Organisers forbid publication → TEM1 becomes primary |
| D-03 | Splits: dev split 80/20 by **overlap group**, stratified by fiber-density tercile; final numbers on 3-fold GroupKFold | E8 (leakage), E9 (13× density range). Singleton images are their own group | No tissue appears on both sides; the val set has dense and sparse images | — |
| D-04 | Score full 4096×3008 images, never tiles | Seam and border errors concentrate at tile edges (report A) | Honest HD95 and instance counts | — |
| D-05 | Label 1 = ignore everywhere (loss, Dice, HD95, matching) | Spec ([01](01-challenge-spec.md)); label 1 = border-cut fibers + deliberately excluded large/oblique fibers (E5) | Predictions on excluded fibers neither help nor hurt | — |
| D-06 | Equivalent-circle g only, from the areas of matched fibers; IoU threshold 0.5 one-to-one (Hungarian) on X∪X+1 | E7: definition spread 0.065 ≫ annotation noise 0.006. IoU>0.5 makes matching unique ([`configs/default.yaml`](../../configs/default.yaml)) | g-MAE comparable with the challenge definition | Organisers publish reference code → mirror it |
| D-07 | HD95 = max of the two directed 95th percentiles (MONAI-style), per image per compartment, in nm (× 2.152); if one side is empty, use the image diagonal 10,936 nm | The spec doesn't say which variant; this is the most common in medical segmentation (report A evaluator table) | Stable, documented HD95. Sensitivity reported for the other variants | Organiser code |
| D-08 | Always report fiber recall and matched count next to g-MAE | g-MAE ignores misses → gameable (report A) | Reviewers can see whether a low g-MAE was bought by skipping fibers | — |
| D-09 | B1 = nnU-Net v2, 2D, **ResEnc-L** plans, labels {0 bg, 1 axon, 2 myelin, 3 ignore}, then a shared watershed pairing | nnU-Net is the established bar (Isensee 2024; ASTIH baseline). nnU-Net's ignore label must be the highest integer and is excluded from the loss (nnU-Net docs) | Dice near ASTIH-level (axon ≥ 0.95, myelin ≥ 0.88 expected on our harder, noisier data) | Dice far below → check normalisation/patch size before anything else |
| D-10 | B1 trains 250 epochs (`-tr nnUNetTrainer_250epochs`, name verified in nnU-Net 2.8.1), one dev fold first. Plans: ResEnc-L if the GPU has ≥ 24 GB, else ResEnc-M | 1000 epochs × 5 folds doesn't fit 2 weeks. nnU-Net usually plateaus well before 1000 epochs on 2D data (to verify on our loss curve) | ≤ 1 day of GPU per run (estimate). **Revised 13 Nov after the timing run (142 s/epoch on L4, 97% GPU utilisation): `nnUNetTrainer_150epochs_ckpt10_es`.** The schedule is 150 epochs so the poly LR still anneals within the $10 budget (≈ 5.9 GPU-h), with a hard cap of 7 GPU-h. The original plan used `nnUNetTrainer_250epochs_ckpt10_es`: 250-epoch cap, checkpoint every 10 epochs (GPUs are preemptible), **early stopping** when the EMA pseudo-Dice gains ≤ 0.001 for 30 epochs (never before epoch 60), and final validation with `--val_best` ([13](13-modal.md) §3b). Log the stop epoch. Caveat: the poly LR schedule is defined over 250 epochs | Loss still falling at 250 → continue with `--c`. **Outcome (13 Nov):** B1 ran all 150 epochs; early stopping never fired (longest stale stretch 19 epochs under the min_delta rule, which has Keras semantics: the reference best only moves on a > 0.001 gain), best EMA at epoch 134, $7.69 ([17](17-day4-scoring-oracles.md) §1.1) |
| D-11 | Shared semantic→instance pairing for every semantic baseline: marker watershed. Seeds = connected components of predicted axon (holes filled, ≥ 50 px); landscape = −EDT of (axon∪myelin); mask = axon∪myelin; then keep fibers with exactly one axon CC and a non-empty ring, writing 2k/2k+1 | One pairing module means baselines differ only in representation. Seeds from components, not integer centroids, avoid ADS's centroid-outside-concave-axon failure (report A) | Pairing behaviour identical across baselines; isolates the representation effect | — |
| D-12 | B3 = Cellpose-SAM (`cpsam_v2`) trained on **filled fibers** (X∪X+1 as one instance) at **6× downsampling** (12.9 nm/px; was 4×, corrected 12 Nov because Cellpose-SAM's training range is 7.5–120 px, see [12](12-day0-5-resources.md)); axon/myelin split inside each fiber from B1's semantic probabilities | Fiber median diameter 755 nm = 351 px is far above Cellpose's training range (7.5–120 px); ×6 gives a median of 59 px and a 5–95% range of 37–99 px. Fibers are filled because a ring's centre lies in its hole, so flows to the centre fail on rings (report A). ASTIH Cellpose got the highest precision (0.990) | High precision, lower recall than B1, mirroring ASTIH (P 0.990 / R 0.797) | Small fibers lost at ×6 → try ×4 with `diameter` set |
| D-13 | AxonPath zero-shot: rescale our images 2.152→8 nm/px (×0.269), percentile-normalise (1–99%), run the bundled `em` weights (= `em_cns`), smoke-tested at ~2 s/image on MPS, upsample instances back with nearest neighbour; compare its `InnerCylinder` (axon + tongue) with our axon | Our GT counts the tongue as axon (E4/[10](10-eda.md)); AxonPath's model was trained at 8 nm/px (report B) | A strong, independent pairing-rule baseline; possibly weaker Dice from domain shift | Zero-shot unusable → fine-tune only if day 4 time allows; otherwise report as is |
| D-14 | Oracle experiments define the error decomposition (T-08) | The only clean way to attribute errors without new labels | Numbers for (a), (b), (c) per baseline | — |
| D-15 | GO/NO-GO strictly by the H1 rule on day 5 | Prevents sunk-cost drift; the 2-week budget is fixed | A decision in the log on day 5 | — |
| D-16 | C2 architecture: shared nnU-Net ResEnc-L 2D encoder-decoder; three heads (filled-fiber foreground, fiber SDT/boundary for separation, inner-vs-outer part) | Instance-first, part-second gives pairing by construction (part-aware panoptic recipe, report A/B). A fiber SDT is what AxonPath uses for fibres, so the difference from AxonPath is *only* the part head replacing the separate axon-instance branch and the 90% rule | Zero pairing violations; fewer gap FPs | Training unstable → freeze the B1 encoder and train only the heads |
| D-17 | g-ratio loss and TopoInteraction-style containment: **ablations only** | E3 (prior art), E6 (g noise floor 0.006), small Δg/px here. Containment already holds by construction in C2 | Effect within CI is the likely result | Big effect → promote in the write-up |
| D-18 | C3 ignore-masking: in training labels, set to ignore any pixels of high-confidence unlabelled ring candidates (dark band ≥ 85% closed, d ≥ 300 nm, not an interstitial gap, out-of-fold B1 fiber probability ≥ 0.9) | E5: about 1 real fiber per image is labelled as background; penalising the model for finding it teaches it to under-segment | Recall ↑ on disagreement fibers; official metrics ±CI | Masking also hides true negatives (organelles) → raise thresholds |
| D-19 | Augmentation: intensity normalisation per image (z-score); brightness/contrast/gamma; smooth illumination gradients (random 2nd-order surface, ±30%); Gaussian+Poisson noise σ 7–22; random black corner masks; flips/rot90 (TEM sections have no orientation) | E9 | Robustness across acquisition regimes; no regression on low-noise images | — |
| D-20 | No edge-snapping or active-contour post-processing | Labels sit ~2–4 px from the strongest edge, with ±1–2 px per-image variation ([11](11-deep-eda.md) §C) | Prevents systematic HD95/g bias | — |
| D-21 | Training-label cleaning: drop the 47 orphan IDs → ignore; the 296 topology-defect fibers → ignore (train only, never in eval) | E4 | Cleaner supervision; eval stays faithful to the official GT | — |
| D-22 | Statistics: bootstrap 1,000× over **overlap groups** (cluster bootstrap), paired between methods; 95% CI | Images in a group aren't independent (E8) | Honest CIs; differences inside the ceilings are flagged | — |
| D-23 | Deferred: dual-radius polar head, SSL pretraining, μSAM, AxonPath data conversion, test-set pseudo-labelling, any challenge submission | Time budget; E3 | — | Days left over after T-14 |
| D-24 (13 Nov) | B1 trains with configuration `2d_bigpatch`: patch 896×1152, batch 3, inheriting everything else from the planner's `2d` (448×576, batch 13) | The planner's patch (0.96×1.24 µm) is smaller than a median fiber (755 nm = 351 px; 95th percentile 595 px), so a patch rarely shows a fiber with its touching neighbours. Pixels per step are nearly equal (3.10 M vs 3.35 M px), so the cost per step is the same | The same or better semantic Dice; better separation on touching fibers | B1 Dice far below the ceiling → run the default patch as an ablation |
| D-25 (13 Nov) | Overlap groups v2: 8 more pairs found after the EDA registration bug was fixed (22 groups, 52 images); splits regenerated before any training | Five of the new pairs crossed the v1 dev split (leakage) ([15](15-day2-evaluator.md) §T-07) | No tissue on both sides of any split | An overlap found later that crosses a split → report it as a limitation; do not re-split after results |
| D-26 (13 Nov) | T-10 reduced to **Cellpose-SAM zero-shot** (local, MPS, ×6), with B1's semantics splitting each instance into axon/ring. Fine-tuned B3 deferred | $10 GPU budget: B1 takes ≈ $7–8.5 (timing run: 142 s/epoch on L4) | An instance-first comparison at no GPU cost | Money left after B1 → fine-tune |
| D-27 (13 Nov) | **H1 = NO-GO** → plan branch C1 + C3 (days 6–8). C2 is not built on the strength of H1 | B1: O-sep gain −0.0014 [−0.0058, +0.0030] vs ≥ 0.03 needed; O-sel +0.016 (strict) / +0.074 (broad); separation share 0.6% vs ≥ 30% ([18](18-day5-go-no-go.md)) | C1 is delivered as the main contribution; C3 targets selection | Owner adopts H5 with a pre-registered rule → C2 returns as a *morphometry* method (first test: GPU-free ring closing + instance-first regrouping on B1) |

## 5. Task list (day by day)

Status markers: `[ ]` open, `[x]` done, `[~]` partial. Each task: **Do / Why / Expect / Done when / If not**.

### Day 0 — prerequisites (before day 1)

- [x] **T-00 Compute and permissions**
  - **Status 12 Nov:** the local Mac has no CUDA GPU. Resources are collected in [12](12-day0-5-resources.md).
  - **Update 12 Nov (evening):** compute is now **Modal**, with an L40S 48 GB GPU and $80 of credit ([13](13-modal.md)). The smoke check passed. The repo has also moved to a 2 TB external disk with about 1 TB free. Data uploaded to the Volume (266 + 266 files, verified).
  - **Do:**
    - Confirm one GPU (≥ 24 GB preferred for ResEnc-L 2D at nnU-Net's default batch) and ≥ 100 GB disk for nnU-Net preprocessed data.
    - Email the organisers asking whether results on their data may be published, and under what licence.
    - Request the Bi-Polar Mask PDF via Sir Faizan.
  - **Why:** D-09/D-10 need a GPU; D-02 needs the licence answer; the C2 novelty statement needs Bi-Polar Mask.
  - **Expect:** a GPU is available; the licence answer may take days, which doesn't block us because of TEM1.
  - **If not (no GPU):** run B1 at 2× downsampling (4.3 nm/px) on CPU/MPS, and drop C2 to a light U-Net. State this in the write-up.

### Day 1 — data plumbing and splits

- [x] **T-01 Loader** (`src/axon_myelin_seg/data/io.py`)
  - **Do:** read uint8 images and uint16 masks with tifffile; validate the shape is 3008×4096; return (image, mask, name). Add the helpers from `encoding.py`.
  - **Why:** single entry point; EDA confirmed there is no metadata to parse ([10](10-eda.md) §1).
  - **Expect:** 266/266 load; 0 shape mismatches.
  - **Done when:** a unit test loads 3 samples and checks dtype and shape.
- [x] **T-02 Training-label cleaner** (`data/cleaning.py`)
  - **Do:** produce a copy of the masks where orphan IDs and topology-defect fibers become ignore (D-21). Log the counts.
  - **Why:** E4.
  - **Expect:** 47 orphan IDs and ~296 fibers converted, matching [10](10-eda.md).
  - **Done when:** the counts match the EDA within ±1.
- [x] **T-03 Splits** (`data/splits.py`, `configs/splits/dev.json`, `cv3.json`)
  - **Do:**
    - Groups are the overlap groups; every other image is its own group.
    - Stratify by fiber-density tercile.
    - Dev split is 80/20, seed 0; plus a 3-fold GroupKFold.
    - Commit the JSONs.
  - **Why:** D-03.
  - **Expect:** about 213/53 images; no group crosses a split.
  - **Done when:** an assertion test passes on both files.

### Day 2 — evaluator (C1 core)

- [x] **T-04 Official replica** (`evaluation/official.py`)
  - **Do:**
    - Mask GT label 1 in both maps.
    - Axon = even ≥ 2; myelin = odd ≥ 3.
    - Per-image Dice for axon and myelin.
    - HD95 per D-07 using surface distances (scipy EDT on boundaries).
    - Fiber matching on X∪X+1 via an IoU matrix and Hungarian assignment at IoU > 0.5.
    - Eq-circle g per matched pair.
    - Return g-MAE, n_matched, recall and precision.
  - **Why:** D-05–D-08.
  - **Expect:** GT vs GT gives Dice 1, HD95 0 and g-MAE 0. Shifting GT by 1 px changes g-MAE by about 0.005 (§1 sensitivity).
  - **Done when:** both tests pass.
- [x] **T-05 Pairing & instance diagnostics** (`evaluation/diagnostics.py`)
  - **Do:**
    - Fiber F1@0.5/0.75, PQ, merges (a pred fiber overlapping ≥ 2 GT fibers each by ≥ 20% of GT area) and splits (the reverse).
    - **Pairing correctness** per matched fiber: pred axon is one connected component, fully enclosed by its own ring, and IoU with the GT axon ≥ 0.5.
    - **Pairing-violation rate** on all predictions: axon without ring, ring without axon, axon touching background, ring with Euler number ≠ 0.
  - **Why:** C1. Nobody measures pairing correctness (report B).
  - **Expect:** GT vs GT gives 0 violations except the known 3.7% topology defects. Report those as the "GT violation rate" baseline.
- [x] **T-06 FP/FN taxonomy** (`evaluation/taxonomy.py`)
  - **Do:** every unmatched predicted fiber gets one class:
    - (a) **unlabelled real fiber**: ≥ 50% overlap with a ring candidate from `scripts/eda/unl_loc.py` that has d ≥ 300 nm and is not a gap
    - (b) **interstitial gap**: ≥ 75% of its outer boundary adjacent to myelin of ≥ 2 different GT fibers
    - (c) **on label 1**: ≥ 50% overlap with label 1. Free under the official metric, but counted
    - (d) **merge/split debris**: overlaps a GT fiber that is matched to another prediction
    - (e) **other**

    Every unmatched GT fiber gets one class: missed (no overlap), merged, split, or low-IoU.
  - **Why:** E5 and E6. This is what separates selection from separation.
  - **Expect:** on GT-derived perturbations, the classes behave as constructed.
  - **Done when:** a hand-made toy example covering each class passes.
- [x] **T-07 Ceiling check**
  - **Do:** run T-04/T-05 on the aligned overlap pairs (0055/0212, 0024/0141, 0091/0201, 0054/0152, 0236/0257), treating one tile as the "prediction".
  - **Why:** validates the evaluator against [11](11-deep-eda.md) and fixes the noise ceilings we compare against.
  - **Expect:** axon Dice ≈ 0.98–0.99, myelin ≈ 0.90–0.97, g-MAE ≈ 0.006. Record bootstrap CI widths for H1's 0.03 threshold.
  - **If not:** a mismatch means an evaluator bug or misregistration. Check registration first (NCC > 0.8).

### Day 3 — baselines start

- [x] **T-08 nnU-Net dataset + B1 training** (done 13 Nov: 150/150 epochs, best EMA epoch 134, 6.34 L4-hours, $7.69; val Dice axon 0.882 / myelin 0.861, **below the expectation below**: the shortfall is selection, see [17](17-day4-scoring-oracles.md) §1, §4)
  - **Do:**
    - Convert to nnU-Net format (Dataset501_TEMAxonMyelin, 2D, `.png` or `.tif` with `_0000` channel suffix).
    - Labels: {background 0, axon 1, myelin 2, ignore 3}. Label 1 → 3; cleaned defects → 3.
    - `nnUNetv2_plan_and_preprocess -d 501 -c 2d -pl nnUNetPlannerResEncL`
    - `nnUNetv2_train 501 2d <fold> -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs --npz`, with the dev split injected as a custom `splits_final.json`.
  - **Why:** D-09, D-10, D-03. `--npz` keeps softmax for the oracle and C3 experiments.
  - **Expect:** a 2D patch on the order of 1024² chosen by the planner; training ≤ ~1 day (estimate); val Dice axon ≥ 0.95, myelin ≥ 0.88.
  - **If not:** Dice < 0.85 myelin → check intensity normalisation (nnU-Net z-score is fine for 2D grey) and that ignore really is the highest label.
- [x] **T-09 Shared pairing module** (`postprocessing/pairing.py`), per D-11
  - **Expect:** when run on **GT semantics**, it reproduces GT instances with fiber F1@0.5 high (≥ 0.95, expected) but lower *per-fiber myelin IoU on touching fibers*. The watershed line between touching sheaths is a guess.
  - **This number is itself result R1.**
- [~] **T-10 B3 Cellpose** (reduced to zero-shot, D-26)
  - **Do:**
    - Filled-fiber instance masks at ×4 downsampling, with label 1 set to 0 and excluded via masks.
    - Fine-tune `cpsam` for about 100 epochs on dev-train.
    - Predict, upsample, and split each fiber with B1's axon probability: axon = largest CC of p_axon > 0.5 inside the fiber, holes filled, eroded 1 px from the fiber edge. The rest is myelin.
  - **Why:** D-12.
  - **Expect:** precision > B1's, recall < B1's (the ASTIH pattern).
- [x] **T-11 AxonPath zero-shot**, per D-13
  - **Expect:** reasonable fiber recall; Dice lower than B1 because of the domain and scale shift. Possibly a lower pairing-violation rate than B1, since it has explicit fibre instances.

### Day 4 — scoring and oracles

- [x] **T-12 Score B1/B3/AxonPath** on dev-val with T-04–T-06 and cluster-bootstrap CIs (D-22). Done 13 Nov ([17](17-day4-scoring-oracles.md) §4): F1@0.5 B1 0.904 [0.881, 0.928], Cellpose+B1 0.885, AxonPath 0.784; g-MAE 0.016 / 0.0114 / 0.060.
- [x] **T-13 Oracle experiments** (on B1, then on the others). Done 13 Nov ([17](17-day4-scoring-oracles.md) §5). **Expectation partly wrong:** O-sel was the largest gain, as expected; but O-sep gave **no** F1 gain on B1 (−0.001), not a modest one; and O-sep cut g-MAE by 42%, which was not anticipated.
  - **O-sel (oracle selection):** drop predicted fibers of taxonomy classes (a) and (c) → Δ precision and F1 = cost of selection.
  - **O-sep (oracle separation):** take the GT fiber instances (X∪X+1 regions) and fill the axon/myelin *compartments from B1's predicted semantics* inside each GT fiber → ΔF1, Δg-MAE = cost of grouping/pairing.
  - **O-pair-GT:** GT semantics → T-09 watershed (= R1). This is the error of the pairing *rule* with perfect pixels.
  - **O-bnd (oracle boundaries):** predicted fiber instances, but every matched fiber's pixels replaced by GT → remaining error = selection + detection only. Its complement is the boundary cost.
  - **Why:** D-14. Each oracle removes one error source.
  - **Expect (prior guess, to be tested):** O-sel gives the largest F1 gain (E5), O-sep gives a modest gain concentrated in touching fibers, O-bnd mainly moves HD95 and g-MAE.

### Day 5 — decision

- [x] **T-14 GO/NO-GO** → **NO-GO** (13 Nov, [18](18-day5-go-no-go.md)). All three H1 conditions failed on B1; the same holds on AxonPath and Cellpose+B1.
  - **Do:** apply the H1 rule to the T-13 numbers with CIs. Write the decision, the numbers and a one-paragraph interpretation into the Log.
  - **Expect:** a decision either way. Both branches below are fully planned.

### Days 6–8 — branch GO (C2)

- [ ] **T-15 Nested decoder** (`models/nested.py`) as an nnU-Net custom trainer/network, so B1's preprocessing, augmentation and sampling are shared.
  - **Heads:**
    - (1) filled-fiber foreground: sigmoid, BCE + Dice, ignore-masked
    - (2) fiber SDT: normalised per fiber to [0,1], MSE, ignore-masked; optional boundary channel
    - (3) part head p_inner: sigmoid, BCE + Dice, loss only inside GT fibers
  - **Decoding:**
    - Seeds = h-maxima of the SDT (h tuned on dev-train).
    - Watershed on −SDT restricted to foreground > 0.5.
    - Per fiber: axon = largest CC of p_inner > 0.5, holes filled, then intersected with the fiber eroded by 1 px. This guarantees containment and that the axon never touches background.
    - Myelin = fiber − axon. Drop fibers with an empty axon or an empty ring. Write 2k/2k+1.
  - **Why:** D-16. Pairing violations become impossible by construction, and a gap between three sheaths only becomes a fiber if head (1) calls the gap foreground. Head (1) is trained with gaps as background.
  - **Expect:** violation rate 0 (structural); merge+split+gap FP −30% vs B1 (H2); Dice within B1's CI.
  - **If not:** if separation is worse than the watershed, swap head (2) for Cellpose-style flows on filled fibers (D-12 logic), at ×4 for flows only.
- [ ] **T-16 Ablations (one run each, dev fold):**
  - + C3 ignore-masking (D-18)
  - + g-ratio loss (λ = 0.1, per GT fiber: |√(Σp_axon / Σp_fiber) − g_gt|, small fibers up-weighted ×2) (D-17)
  - SDT-only vs SDT+boundary channel
  - **Expect:** the C3 effect shows in diagnostics; the g-loss effect is within CI.

### Days 6–8 — branch NO-GO (C1 + C3)

- [ ] **T-15' C3 properly**
  - **Do:**
    - Out-of-fold B1 predictions on all 266 images (3-fold).
    - Build ignore masks per D-18.
    - Retrain B1 with them.
    - Evaluate on (i) the official metrics and (ii) the overlap-disagreement fibers (fibers labelled in one tile and background in the other: the only place we know a "missed" label is real).
  - **Expect:** H3.
- [ ] **T-16' Full C1 analyses**
  - **Do:**
    - Decomposition for B1/B3/AxonPath.
    - Recall-vs-g-MAE gaming curve: drop predicted fibers by confidence or size and plot g-MAE against recall.
    - g-definition sensitivity (E7).
    - HD95-variant sensitivity.
    - Rank stability under these switches.
  - **Expect:** g-MAE can be lowered by dropping small or uncertain fibers while recall falls. Rankings flip under some evaluator switches.

### Day 9 — external replication (both branches)

- [ ] **T-17 ASTIH TEM1**
  - **Do:**
    - Download DANDI 001436 with `ASTIH/scripts/get_data.py --make-splits`. The test subject is `sub-nyuMouse26`.
    - Apply the released ASTIH model (`get_models.py`, `apply_models.py`) and score it with **our** evaluator → must reproduce Dice 0.974/0.924 and F1@0.5 0.791 (±0.01 Dice, ±0.03 F1).
    - Then run our best method or analysis (train on TEM1-train, rescale 2.36→2.152 or train natively).
    - Instance GT for TEM1 is built the ASTIH way (ADS watershed on GT). Note that this makes the TEM1 pairing GT rule-derived, which is a known limitation for pairing claims.
  - **Why:** D-02 and E10. It validates the evaluator against a published number and checks H4.
  - **If not reproduced:** fix the evaluator before trusting any day-4 result.

### Day 10 — write-up

- [ ] **T-18 Short paper (4 pages, MIDL short-paper format; check the current call for dates)**
  - **Do:** method or analysis, decomposition table, ceilings, gaming curve, limitations (selective annotation, rule-derived TEM1 pairing GT, no animal IDs).
  - **Update:** README, notes 05/06/08, this log.

## 6. Rules carried everywhere (from the EDA)

- Splits use `configs/splits/overlap_groups.json` (D-03). Score full images (D-04).
- Label 1 = ignore (D-05). Never assume consecutive IDs. Orphans are skipped in eval and ignored in training (D-21).
- Eq-circle g only, with recall next to g-MAE (D-06, D-08). No edge snapping (D-20).
- Read every difference against the ceilings: myelin Dice ~0.93–0.96, g-MAE ~0.006, fiber selection disagreement ~8% (E6, D-22).

## 7. Risks specific to this plan

| Risk | Signal | Response |
|---|---|---|
| nnU-Net training slower than budgeted | < 50% of epochs done by the end of day 3 | Use 2× downsampled data; 150 epochs |
| Evaluator disagrees with the TEM1 reference | T-17 numbers off by more than the tolerance | Stop and debug the evaluator. All conclusions depend on it |
| H1 result is ambiguous (inside the CIs) | CIs overlap 0.03 | Treat as NO-GO (conservative); report the decomposition |
| Organisers forbid publication | Email reply | TEM1 becomes the primary dataset; the challenge data is internal validation only |
| Bi-Polar Mask turns out to be nested by construction | PDF read | C2 framed as "multi-instance, touching annuli, dense" (report B novelty statement) |

## 7a. Open items

- Organiser reply on publication and licence ([08](08-open-questions-and-risks.md)).
- Bi-Polar Mask full text.
- AxonPath preprint (still pending; re-check before submission).

## Log

- 2026-11-10: repo scaffolded; challenge site crawled; literature review preserved.
- 2026-11-11: deep-research pass done (5 research streams plus a final report in `docs/reports/`). Notes 03–08 and `configs/default.yaml` updated. **Next:** inspect the 266 training masks (inner tongue, sheath splitting, scale bar, filename metadata), then implement `evaluation/official.py` and validate it against ASTIH TEM1 (B0).
- 2026-11-11: feasibility check for a 2-week budget. The gap still holds: a fresh search found no paper on nested/paired axon–myelin instances or g-ratio-trained segmentation. **Biggest risk:** access to the challenge data, since we are not registered. **Fallback:** run the whole study on public CC-BY data (ASTIH TEM1 + AxonPath). Two-week scope: evaluator + B1/B3 + oracle-pairing (week 1), then the nested decoder + g-ratio loss vs B1 (week 2). The dual-radius head, SSL pretraining and μSAM are deferred.
- 2026-11-12: gap-validation pass (`docs/reports/Paired axon myelin gap validation.md`). The gap is open but narrower than first stated. AxonPath (Sep 2026) is a multi-head joint model, but it pairs with a 90% overlap rule. NPS-Net, Alblas 2022 and Bi-Polar Mask already cover nested polar contours, so the dual-radius head is no longer the novelty. Meng 2022 is prior art for ratio losses. Novelty now rests on dense multi-instance nesting by construction, a per-fiber g-ratio loss and a pairing-correctness metric. Plan changes: add AxonPath as a week-1 baseline; oracle pairing becomes the go/no-go experiment; still need the Bi-Polar Mask full text.
- 2026-11-12: training data received (266 images + masks) and EDA done ([10](10-eda.md)). Conventions are now settled: border-cut fibers are label 1, the inner tongue is part of the axon, IDs are not consecutive, and 47 orphan labels exist. **New risk:** selective annotation. Abnormal myelinated fibers are background (~25% of ring-like structures), so fiber *selection* becomes a hidden task, and the go/no-go experiment must separate selection errors from pairing errors. Fibers are large (median axon 562 nm), so g is less sensitive than the report assumed.
- 2026-11-12: deep EDA ([11](11-deep-eda.md)). Found **leakage**: 22 overlapping tile pairs in 18 groups (40 images), saved in `configs/splits/overlap_groups.json` and to be used as split groups. **FP patterns:** ~4–8% of real fibers are unlabelled (random misses, not a rule), interstitial gaps between touching fibers act as pseudo-fibers, and interior label 1 marks large or oblique fibers. **Ceilings from re-annotated overlaps:** myelin Dice ~0.93–0.96, g-MAE floor ~0.006. The g definition spread (0.065) is 10× annotation noise. Next: an evaluator with an FP taxonomy (unlabelled fiber / interstitial gap / label 1 / other).
- 2026-11-12: **plan v2.** The project becomes evaluation-first (C1 error decomposition, always delivered), method-second (C2 nested decoder, only if the H1 go/no-go passes on day 5), with C3 incomplete-annotation handling. Dual-radius head, SSL, μSAM and the g-ratio-loss headline are dropped.
- 2026-11-12: plan v2 expanded into a decision register (D-01–D-23) and a task list (T-00–T-18), each with reason, expected outcome and fallback. Hypotheses H1–H4 now have numeric decision rules.
- 2026-11-12: day 0–5 resources collected ([12](12-day0-5-resources.md)): nnU-Net, Cellpose, AxonPath and ASTIH code; 9 new papers; `.venv` (torch 2.15, nnunetv2 2.8.1, cellpose 4.2.1.1, axonpath 0.1.0). Corrections: D-12 Cellpose ×6 (not ×4); D-09/D-10 plan ResEnc-M vs L chosen by VRAM, trainer name verified; D-13 AxonPath weights bundled and runnable locally; ASTIH's CLI matching default is 0.3, so always set the threshold. No local GPU: training waits for the GPU machine (13 Nov). The MPS timing benchmark was cancelled, so epoch time is still unmeasured.
- 2026-11-12: **compute = Modal** (workspace `muhammadfaheem-mansha`, $80 credit) ([13](13-modal.md)).
  - Added `infra/modal/tem_modal.py`: an image pinned to the local versions, one private Volume, and CPU, GPU and resumable nnU-Net train functions.
  - Smoke check on an L40S passed: torch 2.15.0+cu132, CUDA available, 44.4 GiB.
  - GPU containers can be preempted, so training retries with `--c`. New trainer `nnUNetTrainer_250epochs_ckpt10` (D-10): the schedule is unchanged; it only checkpoints more often.
  - Budget: about 30 L40S-hours. Epoch time is still unmeasured; the next step measures it with the 5-epoch trainer.
- 2026-11-12: **budget for days 0–5 = $10.** Default GPU changed to **L4** and plans to **ResEnc-M** (D-09/D-10 fallback branch; reasons in [13](13-modal.md) §3a): about 6.9 L4-hours. Measure the epoch time first; if 250 epochs would cost more than $8, cut epochs and log it. Data uploaded to the private Volume and verified: 266 images + 266 masks at `/data/raw/train-dataset/`.
- 2026-11-13: **compute split** (owner): all CPU work runs on the local M4 (conversion, planning/preprocessing, pairing, evaluation, AxonPath); Modal runs GPU work only. Mechanisms to avoid paying for unneeded GPU seconds ([13](13-modal.md) §3b):
  - early-stopping trainer (D-10)
  - a hard per-run GPU-hour cap with a ledger that retries can't extend
  - single-use containers
  - local-disk staging of the preprocessed data
  - a GPU-utilisation log
  - the 5-epoch timing run first

  The `shell_cpu` function was removed. The GPU container now requests 6 cores and 16 GiB (~$1.21/h on L4, ~8 h for $10). The resume check now matches both `.pth` (2.8.1) and `.safetensors` (master) checkpoints. L4 smoke check passed.
- 2026-11-13 (day 1, [14](14-day1-data-plumbing.md)): T-01–T-03 done.
  - 266/266 load; 47 orphans; 296 defect fibers, as expected.
  - Splits are frozen (dev 213/53, cv3).
  - Fixed a stale editable install left over from the repo move.
- 2026-11-13 (day 2, [15](15-day2-evaluator.md)): evaluator done (T-04–T-06), with tests.
  - **T-07 found a bug in the EDA overlap registration (inverted ECC direction).** Fixed, 18/22 pairs register (was 5), and 8 new overlapping pairs turned up, 5 of which crossed the dev split. Groups v2 and splits regenerated before any training (D-25).
  - **Ceiling:** human vs human fiber F1@0.5 = 0.929, with **0 merges/splits**: all disagreement is selection. Axon/myelin Dice 0.987/0.951, g-MAE 0.0067.
- 2026-11-13 (day 3, [16](16-day3-baselines.md)): nnU-Net Dataset501 converted and preprocessed locally.
  - D-24 big patch. Uploaded to Modal; the first upload failed silently and a chunked uploader was added.
  - **R1:** the pairing rule on GT semantics gives F1 0.99994 and g-MAE 0.0017, so the rule is not an error source.
  - **AxonPath zero-shot:** F1 0.784, Dice 0.71/0.62, g-MAE 0.060. Separation is only 9% of its errors.
  - **Timing run:** 142 s/epoch, 97% GPU utilisation (compute-bound). D-10 revised to 150 epochs with early stopping.
  - B1 launched on an L4 with a 7 GPU-h cap. T-10 reduced to Cellpose zero-shot (D-26).
- 2026-11-13 (day 4, [17](17-day4-scoring-oracles.md)): **B1 done**.
  - 150/150 epochs; early stopping never fired (the min_delta rule's reference only moves on a > 0.001 gain; longest stale stretch 19). Best EMA 0.8743 at epoch 134; 6.34 L4-hours, no preemption, GPU utilisation 94%; **$7.69**.
  - **Environment incident #2:** 109 launchers in `.venv/bin` (activate, pytest, modal, nnUNetv2_*) still executed the old repo copy's venv. Fixed by rewriting the paths. Earlier results are unaffected: the old copy only has stubs, so a wrong-venv run would have crashed.
  - **T-12:** F1@0.5 B1 0.904 [0.881, 0.928] (ceiling 0.929), Cellpose+B1 0.885, AxonPath 0.784. g-MAE 0.016 / 0.0114 / 0.060. Our Dice equals nnU-Net's own (axon 0.8822 vs 0.8823), an independent evaluator check.
  - **Deviations from expectations:**
    - B1 val Dice 0.882 / 0.861 vs the expected ≥ 0.95 / ≥ 0.88. The cause is selection (O-sel-broad → 0.948 / 0.908), so D-24's fallback is not triggered.
    - Cellpose+B1 has *higher recall, lower precision* than B1; the opposite of the ASTIH pattern expected in T-10, because ours is zero-shot and partitions the whole image.
  - **T-13:** separation is 0.6% of B1's 331 errors (2 merges, 0 splits, 0 gap FPs). O-sep gain −0.001; O-sel +0.016 / broad +0.074; O-bnd +0.023 at F1@0.75.
    - **Unexpected:** O-sep cuts g-MAE by 42% (0.016 → 0.0093). 28% of matched B1 fibers have broken predicted rings, carrying 56% of the g error.
    - The B1 FP gallery shows ~7/16 real (thin-myelin) fibers, ~7 abnormal profiles and ~2 fragments.
- 2026-11-13 (day 5, [18](18-day5-go-no-go.md)): **T-14: H1 = NO-GO (D-27).** No condition met, none borderline. The plan proceeds to the NO-GO branch (C1 + C3).
  - **H5** recorded (post hoc, not pre-registered): grouping/ring topology drives g-ratio error. It is the owner's call whether to pre-register it.
  - **Budget:** $8.16 of the $10 spent (Modal billing summary); $1.84 left. T-15′ (3-fold out-of-fold B1 ≈ $23 + retrain ≈ $8) does **not** fit the cap. Options are in [18](18-day5-go-no-go.md) §4.
  - Days 6–8 were not started (out of scope for this run).
