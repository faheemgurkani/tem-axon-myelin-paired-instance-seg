# 07 — Research plan: hypotheses, decisions, next tasks, log (plan v2, updated 13 Nov 2026)

**Rules for this file:**
- Every decision (D-xx) and task (T-xx) has an ID.
- When an outcome differs from its expectation, write it in the **Log** and update the decision. Never silently edit an expectation.
- Cite the IDs in commit messages.
- Time budget: 10 working days. Days 0–5 are done.

## 1. Question and contributions

**Question:** in paired axon–myelin instance segmentation of TEM, how much fiber-level error comes from each source, once annotation noise is accounted for?
- **(a) selection:** which fibers count;
- **(b) separation/pairing;**
- **(c) boundary placement.**

And does a decoder that is nested by construction remove (b)?

| ID | Contribution | Status |
|---|---|---|
| **C1** | Error-decomposition evaluation: pairing-correctness metric, FP/FN taxonomy, oracles, human ceilings from overlapping tiles, recall-vs-g-MAE gaming curve, g-definition sensitivity | Core built and run (days 2–5). Remaining analyses: T-16′ |
| C2 | Nested-by-construction decoder (instance-first, part-second) | **Not built:** H1 = NO-GO. Could return via H5 |
| **C3** | Learning with incomplete fiber annotation (ignore confident unlabelled rings in training) | **Next main method** (NO-GO branch); blocked on budget |

## 2. Hypotheses

| ID | Statement and decision rule | Status |
|---|---|---|
| **H1** | On B1, the separation oracle gains ≥ 0.03 F1@0.5, **and** ≥ the selection-oracle gain, **and** separation is ≥ 30% of errors → GO (build C2) | **NO-GO** (13 Nov): −0.001 / +0.016 / 0.6% |
| H2 | C2 cuts (merge + split + gap FP) by ≥ 30% vs B1 and AxonPath, with Dice within B1's CI | Moot: B1 already has 2 such errors in 1,619 fibers |
| **H3** | C3 raises recall on overlap-disagreement fibers (labelled in one tile, background in the other) by ≥ 10 points, with no official metric outside B1's CI | Open (T-15′) |
| H4 | The method ranking is the same on ASTIH TEM1 | Open (T-17) |
| **H5** *(post hoc, 13 Nov, NOT pre-registered)* | Grouping and ring topology drive g-ratio error. Evidence: O-sep −42% g-MAE; instance-first grouping beats the watershed by 0.0046 g; broken rings carry 56% of g error | **Owner to decide** whether to pre-register it with a numeric rule *before* testing |

## 3. Decision register (compact)

| ID | Decision | Status / outcome |
|---|---|---|
| D-01 | Evaluation-first; the method is conditional on H1 | Applied |
| D-02 | Primary data = challenge train set; external = ASTIH TEM1 (published reference); AxonPath data deferred | Applied; TEM1 pending (T-17) |
| D-03 | Splits by overlap group, stratified by density; dev = fold 0 of 5; final numbers on 3-fold | Done; frozen |
| D-04 | Score full images, never tiles | Applied |
| D-05 | Label 1 = ignore everywhere; predictions ≥ 50% on label 1 are ignored | Applied |
| D-06 | Equivalent-circle g; one-to-one matching at IoU > 0.5 | Applied |
| D-07 | HD95 = max of directed P95, per image and compartment, in nm; empty side → 10,936 nm | Applied |
| D-08 | Always report recall next to g-MAE | Applied |
| D-09 | B1 = nnU-Net 2D, labels {0, 1 axon, 2 myelin, 3 ignore}, shared watershed. Plans ResEnc-L if ≥ 24 GB VRAM, else ResEnc-M | ResEnc-M (L4, $10 budget) |
| D-10 | Training length | Revised to **150 epochs + early stopping, 7 GPU-h cap** after the timing run (250 would cost ≈ $12). Outcome: ran 150/150, ES never fired, best epoch 134, $7.69 |
| D-11 | One shared semantic→instance pairing (component-seeded −EDT watershed) | Done; R1 shows it is lossless on perfect pixels |
| D-12 | B3 = Cellpose-SAM on filled fibers at ×1/6 | Fine-tune not run (D-26) |
| D-13 | AxonPath zero-shot at 8 nm/px; inner cylinder → axon | Done; with axon fallback |
| D-14 | Oracles define the error decomposition | Done |
| D-15 | GO/NO-GO strictly by the H1 rule on day 5 | Applied |
| D-16 | C2 architecture: shared encoder; heads for filled-fiber foreground, fiber SDT, inner/outer part | Not built |
| D-17 | g-ratio loss and containment loss are ablations only | Unchanged; may matter under H5 |
| D-18 | C3: in training labels, ignore high-confidence unlabelled ring candidates (closed dark band ≥ 85%, d ≥ 300 nm, not a gap, out-of-fold B1 fiber probability ≥ 0.9) | **Check first:** the candidate detector misses thin-myelin fibers ([05](05-results.md) §5) |
| D-19 | Augmentation: per-image z-score, brightness/contrast/gamma, illumination gradients, noise σ 7–22, black corners, flips/rot90 | nnU-Net defaults used for B1; custom augmentation not yet added |
| D-20 | No edge snapping | Applied |
| D-21 | Training-label cleaning: 47 orphans + 296 defect fibers → ignore (train only) | Done |
| D-22 | Cluster bootstrap over overlap groups; paired differences | Applied |
| D-23 | Deferred: dual-radius head, SSL pretraining, μSAM, AxonPath data conversion, test pseudo-labels, any submission | Deferred |
| D-24 | B1 trains on 896×1152 patches, batch 3 (`2d_bigpatch`) | Done. Fallback (default-patch ablation) not triggered: the Dice shortfall is selection |
| D-25 | Overlap groups v2 (22 groups / 52 images); splits regenerated before training | Done. An overlap found later → report it, don't re-split |
| D-26 | T-10 reduced to Cellpose-SAM zero-shot + B1 split ($10 budget) | Done |
| **D-27** | **H1 = NO-GO → branch C1 + C3**; no C2 on the strength of H1 | 13 Nov. Revisit only if the owner pre-registers H5 |

## 4. Tasks

**Done (days 0–5):**

| Tasks | What |
|---|---|
| T-00 | Compute: Modal setup |
| T-01 – T-03 | Loader, cleaner, splits |
| T-04 – T-07 | Evaluator, diagnostics, taxonomy, ceilings |
| T-08 | nnU-Net data + B1 |
| T-09 | Pairing + R1 |
| T-11 | AxonPath |
| T-12, T-13 | Scoring, oracles |
| T-14 | **NO-GO** |
| T-10 | Partial: zero-shot only |

**Open (NO-GO branch, days 6–8; needs a budget decision):**

- **T-15′ C3, learning with incomplete annotation** (≈ $31 at full size):
  1. Out-of-fold B1 predictions on all 266 images: 3 folds × ~6.3 L4-h ≈ $23.
  2. Build ignore masks per D-18, after checking the candidate detector's recall on thin-myelin fibers.
  3. Retrain B1 with them (≈ $8).
  4. Evaluate on (i) the official metrics and (ii) the overlap-disagreement fibers. These are the only fibers where a "miss" is known to be a real fiber. Expect H3.
  - **Cheaper variants:** 60-epoch folds (~2.4 h each); or use the dev-fold B1 only and skip out-of-fold labels for train images.
- **T-16′ Full C1 analyses** (local CPU, no GPU):
  - decomposition for every baseline (done);
  - **recall-vs-g-MAE gaming curve:** drop fibers by size or confidence and plot g-MAE against recall;
  - g-definition sensitivity;
  - HD95-variant sensitivity;
  - per-fiber morphometry agreement (Bland–Altman of g, axon diameter, myelin thickness; errors by size bin);
  - rank stability under these switches.
- **H5 first test** (only if the owner adopts H5; local, no GPU): close broken rings and regroup B1's pixels instance-first. Does simple post-processing capture most of O-sep's g gain? If yes, no new decoder is needed.

**Both branches:**
- **T-17 ASTIH TEM1 (day 9):**
  1. Download DANDI 001436 (`ASTIH/scripts/get_data.py --make-splits`; test subject `sub-nyuMouse26`).
  2. Apply ASTIH's released model and score it with **our** evaluator. It must reproduce Dice 0.974/0.924 and F1@0.5 0.791 (±0.01 Dice, ±0.03 F1).
  3. **Always set the matching threshold explicitly:** ASTIH's CLI default is 0.3, although its help text says 0.5.
  4. If it doesn't reproduce, fix the evaluator before trusting any result.
  5. Note that TEM1's instance GT is itself watershed-derived, which limits pairing claims there.
  6. Run our taxonomy on TEM1: is ASTIH's low precision at IoU 0.5 (0.747, against Dice 0.974) selection (our main finding, replicated) or low_iou? Use TEM1's 20 subject IDs for an animal-level bootstrap. No independent group has published on TEM1 yet ([report](../reports/ASTIH%20TEM1%20citation%20analysis.md)).
- **T-18 Short paper** (4 pages, MIDL short format): decomposition table, ceilings, gaming curve, limitations (selective annotation, rule-derived TEM1 pairing, no animal IDs).
  - Later venues: a method paper at MICCAI / ISBI 2027 if C3 or H5 work; a journal extension (MedIA / TMI) with external datasets.

## 5. Risks and open questions

| Risk | Response |
|---|---|
| Our metric replica diverges from the organisers' | Config switches + sensitivity table; T-17 checks against a published number |
| Data licence blocks publication | Ask the organisers; keep TEM1 as a publishable fallback |
| Unknown overlaps cross the frozen split | Report it as a limitation; don't re-split |
| Single lab, no animal IDs | Image-level claims only; external validation on TEM1 |
| g-MAE rewards dropping hard fibers | Always report recall; the gaming curve (T-16′) |
| Budget | Raise the cap or use cheaper T-15′ variants |

**Questions for the organisers** (gabriel.jimenez@icm-institute.org; not yet sent; in the owner's TODO):
- Can we publish on the data, and under what licence?
- Will the test GT or an evaluation server be released?
- Is there official evaluation code (Dice aggregation, HD95 variant, IoU threshold)?
- Is there per-image animal or condition metadata?
- What is the annotation protocol for abnormal and remyelinated fibers (why are some left as background)?
- Is transductive use of the test images allowed?

**Pending reading:** Bi-Polar Mask full text (paywalled), the AxonPath preprint (not out).

## 6. Log (condensed; the full history is in git)

- **10 Nov:**
  - Challenge site crawled and verified.
  - Dataset = 266 + 134; mask format = uint16 pairs.
- **11 Nov:**
  - Deep-research review (report A): the gap is post-hoc pairing, with no g-trained segmentation.
  - Biggest early risk: data access (not registered).
- **12 Nov:**
  - Gap validation (report B): AxonPath already does joint prediction; ASTIH's 0.79 measures detection. So "pairing is the bottleneck" becomes hypothesis H1, not a premise.
  - Data received. EDA: selective annotation, unlabelled fibers, label 1 = fibers, overlapping tiles (leakage), human ceilings.
  - **Plan v2:** evaluation-first, with C1 / C2 (conditional) / C3.
  - Modal set up; budget for days 0–5 is $10, so L4 + ResEnc-M.
- **13 Nov, day 1:**
  - Loader, cleaner and splits done.
  - Fixed stale editable installs after the repo move.
- **13 Nov, day 2:**
  - Evaluator done.
  - EDA registration bug found, which revealed 8 new overlaps, 5 of them leaking. Groups v2 and splits regenerated before training (D-25).
  - Human F1 0.929; all disagreement is selection.
- **13 Nov, day 3:**
  - R1: the pairing rule is lossless on GT pixels.
  - D-24 big patch.
  - Silent upload failure fixed.
  - AxonPath F1 0.784.
  - Timing run 142 s/epoch → D-10 revised to 150 epochs.
  - Cellpose reduced to zero-shot (D-26).
  - B1 launched.
- **13 Nov, day 4:**
  - B1 done: 150/150 epochs, $7.69. F1 0.904; Dice 0.882 / 0.861, below expectation because of selection.
  - Separation is 0.6% of errors.
  - **Unexpected:** O-sep −42% g-MAE; broken rings carry 56% of g error.
  - Fixed stale venv launchers.
- **13 Nov, day 5:**
  - **H1 = NO-GO (D-27).** H5 recorded.
  - Spend $8.16 of $10. Days 6–8 wait on the owner's budget decision.
- **13 Nov:** notes consolidated from 19 files into 7 (this structure).
