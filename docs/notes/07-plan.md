# 07 — Research plan: hypotheses, decisions, next tasks, log (plan v2, updated 15 Nov 2026)

**Rules for this file:**
- Every decision (D-xx) and task (T-xx) has an ID.
- When an outcome differs from its expectation, write it in the **Log** and update the decision. Never silently edit an expectation.
- Cite the IDs in commit messages.
- Time budget: 10 working days. Days 0–5 are done.
- From 15 Nov, the order of work, the gates and the to-do list live in the owner's local `PLAN.md` (not committed). This file keeps the record: hypotheses, decisions, task definitions, risks and the log.

## 1. Question and contributions

**Question:** in paired axon–myelin instance segmentation of TEM, how much fiber-level error comes from each source, once annotation noise is accounted for?
- **(a) selection:** which fibers count;
- **(b) separation/pairing;**
- **(c) boundary placement.**

And does a decoder that is nested by construction remove (b)?

| ID | Contribution | Status |
|---|---|---|
| **C1** | Error-decomposition evaluation: pairing-correctness metric, FP/FN taxonomy, oracles, repeat-annotation agreement from overlapping tiles, recall-vs-g-MAE gaming curve and a PQ-style g-quality metric, g-definition sensitivity; plus the protocol's effect on the endpoint (H7) and on models (H8) | Core built and run (days 2–5). Remaining: T-16′, T-20, T-23 (H7), T-22 (H8), H6. Status 15 Nov: stronger; selection is protocol-specific (ours ≈ 4.2% skipped vs ≤ 0.3% elsewhere) |
| C2 | Nested-by-construction decoder (instance-first, part-second) | **Not built:** H1 = NO-GO. Could return via H5 |
| **C3** | Learning with incomplete fiber annotation. **Proposed redesign (15 Nov):** a fiber-level SAR selection model (missed normal / protocol-excluded / not a fiber) gating a soft ignore in the B1 recipe | The methods opening: no method anywhere models *why* objects are unlabelled. Mechanism is prior art (nnU-Net ignore, LNQ, SAR-PU). Claimable: the instance-level selection model, overlap-tile capture–recapture, the first appearance-dependent omission benchmark in EM. Primary test: TEM1 benchmark (T-24/H9, proposed); H3 (18 fibers) secondary. ≈ 1 in 3 for a methods paper |

## 2. Hypotheses

| ID | Statement and decision rule | Status |
|---|---|---|
| **H1** | On B1, the separation oracle gains ≥ 0.03 F1@0.5, **and** ≥ the selection-oracle gain, **and** separation is ≥ 30% of errors → GO (build C2) | **NO-GO** (13 Nov): −0.001 / +0.016 / 0.6% |
| H2 | C2 cuts (merge + split + gap FP) by ≥ 30% vs B1 and AxonPath, with Dice within B1's CI | Moot: B1 already has 2 such errors in 1,619 fibers |
| **H3** | C3 raises recall on overlap-disagreement fibers (labelled in one tile, background in the other) by ≥ 10 points, with no official metric outside B1's CI | Open (T-15′) Evaluation rule added 15 Nov (D-34): judge C3 only on fibers known to be real (overlap disagreements; thin fibers on outside sets, H8), never on official precision alone ("observed-label illusion", Parikh 2026) **Power (15 Nov audit):** the overlap-disagreement set is only **18 fibers** (FP 7 + FN 11, [05](05-results.md) §2), so H3 is reported as secondary; the proposed TEM1 controlled benchmark (T-24) is the primary known-positive test if adopted |
| H4 | The method ranking is the same on ASTIH TEM1 | Open (T-17) |
| **H5** *(post hoc, 13 Nov, NOT pre-registered)* | Grouping and ring topology drive g-ratio error. Evidence: O-sep −42% g-MAE; instance-first grouping beats the watershed by 0.0046 g; broken rings carry 56% of g error | **Pre-registered 15 Nov (D-29)** as H5a (O-sep cuts g-MAE ≥ 25%), H5b (topology carries it) and H5c (frozen post-processing captures ≥ 50% of O-sep's gain within F1/recall guards). Confirmed **only** on the 213 non-dev images via T-21; dev values are not evidence. Full rules in §7.2 |
| **H6** *(pre-registered 15 Nov, D-31, before any TEM1 prediction exists)* | Selection errors track label incompleteness. On TEM1, which the completeness audit shows is near-complete (u 0.6% vs ours 5.1%), the selection share of errors is small. **H6a:** on `sub-nyuMouse26` with ASTIH's released TEM1 model + ADS post-processing, scored at IoU 0.5, strict selection share (unlabelled_real) ≤ 5% **and** broad share (unlabelled_real + other_bg) ≤ 39% (half of B1's 78.9% on ours). **H6b:** the modal TEM1 FP class is not unlabelled_real. **H6c (TEM2, added 15 Nov after its audit, before any TEM2 score):** the same two thresholds on TEM2's external test set. Prerequisite: T-17 reproduces ASTIH (±0.01 Dice, ±0.03 F1) on each dataset | Open. Full rule in §7.4. **Limitation (15 Nov):** the leakage-free TEM1 test mouse looks control-like (proxy), so a TEM1 pass speaks mainly for healthy tissue. The earlier informal H6 ("high on TEM1, low on TEM2") was **withdrawn before any data**: the audit reversed its premise |
| **H7** *(added 15 Nov, D-34; C1)* | The labelling protocol biases the morphometric endpoint. **Two-sided.** H7a: including the expert-confirmed skipped fibers shifts mean per-fiber g by \|Δ\| ≥ 0.03. H7b: the protocol under-counts myelinated fibers by ≥ 5%. H7c: it changes the share of fibers with g > 0.85 by ≥ 2 points (absolute) | Planned (T-23). **Feasibility caveat written before any data:** Δ mean g ≈ p·(g_skipped − g_labelled); at p ≈ 5% H7a needs a g difference ≈ 0.6 and will almost surely fail. It is kept to report honestly; H7b/H7c are the better-powered readouts. "Masks disease" needs per-image condition (A2). Full rule §7.5 |
| **H8** *(added 15 Nov, D-34; C1, + C3 outside test)* | Training on selective labels creates a hidden stratification. On near-complete outside data, B1's recall gap (thin-myelin minus normal fibers) is worse than a reference model trained on complete labels. Holds if (gap_B1 − gap_ref) ≤ −0.10 with a mouse-bootstrap CI below 0 | Planned (T-22). Tests **thin-myelin** fibers only (outside sets also exclude degenerating rings). The reference must be leakage-free and see thin fibers, so it needs a TEM1 fold holding out the remyelinating-like mice (≈ $8, reserve). Full rule §7.5 |
| H9 *(proposed 15 Nov, not adopted; C3 methods route)* | On the TEM1 controlled two-part omission benchmark (T-24), the fiber-level selection-gated soft ignore beats **both** hard ignore and the random-ignore control on normal-fiber recall against protocol truth (paired mouse-bootstrap CI > 0), **and** its protocol-violation rate is ≤ the naive arm's + 0.05. Official metrics never decide | Draft; fixed on adoption, before any run. Design: [C3 methodology report](../reports/C3%20methodology%20design.md); audit findings in the log (15 Nov) |

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
| **D-28** | GPU budget: $60 ceiling including the $8.16 spent, released in tranches. A ≈ $23 (cap $26) T-21; B ≈ $15.4 (cap $17) C3 pilot with 2 arms; C ≈ $15.4 (cap $17) C3 full; ~$12 reserve. Each tranche opens on a gate; C2 is outside it | 15 Nov. The earlier "pilot ≈ $8" costed one arm only. T-15′'s dev-only cheaper variant is ruled out: B1's dev model has seen the training images, so its probabilities there would bias the ignore masks |
| **D-29** | H5 adopted and pre-registered. Metric: equivalent-circle g, matching IoU > 0.5, paired cluster bootstrap. H5a: O-sep Δg-MAE ≥ 25% relative, CI excludes 0. H5b: defective fibers ≤ 35% of fibers but ≥ 40% of \|g err\|, and ≥ 50% of O-sep's reduction. H5c: post-processing tuned on dev and frozen captures ≥ 50% of O-sep's gain, ΔF1@0.5 and Δrecall lower CI ≥ −0.005. Gate 3a: H5a fails → reject; H5a+H5c → measurement finding, no C2; H5a only → C2 candidate (new budget) | 15 Nov. Amendments only before any T-21 prediction exists, with a logged reason |
| **D-30** | AxonPath CC TEM approved as an external dataset (T-17 item 7): download and check now, score only after Gate 0. AxonPath's own model is **never** scored on it (tile-level training split) | 15 Nov (engineer, delegated) |
| **D-31** | H6 rewritten from the completeness audit (TEM1 near-complete, ours selective) and pre-registered as H6a/H6b. TEM2 rule deferred until full-image TEM2 is audited | 15 Nov. Premise reversal logged in §6 and §7.4 |
| **D-32** | B4: primary per-fiber g = equivalent-circle D_A/D_F (challenge = ADS default). Also report D_A and D_F errors; population summaries use the g' cline (D_A-on-D_F slope with intercept), never mean g or a regression of g on diameter (Gow 2026); bin by fiber diameter; with/without oblique fibers. Yardstick (B5): smallest meaningful group shift ≈ 0.03, disease 0.05–0.10 | 15 Nov. Evidence: `docs/research_notes/Phase 0b evidence/02_gow_g_ratio.md`, `06_g_effect_sizes_B5.md` |
| **D-33** | Blind expert review: T-20 (61 `unlabelled_real` + 40 `other_bg`) and the completeness-audit candidates (159), 260 items in one shuffled packet. Pass rules fixed before rating: T-20 ≥ 70% real (expert); κ ≥ 0.6 between expert and owner; ours ≥ 70% real; each external ≤ 25% real (else that dataset's H6 run waits for a re-check) | 15 Nov. Protocol: `docs/research_notes/Phase 0b evidence/10_expert_review_protocol.md` |
| **D-34** | Integrate the validated gap findings into the plan (owner instruction 15 Nov). **Adopted:** H7 (protocol bias on g and counts, two-sided, T-23), H8 (hidden stratification on outside data with a complete-label reference arm, T-22; also C3's outside test), C3 evaluation on known positives only. **C1 add-ons:** PQ-style g-quality metric (cite PQ), "repeat-annotation agreement" wording for the overlap ceiling, and the lower-bound note on the selection share. **Gate 1 option only:** an abnormal-fiber class (prior art Zepf 2023, Parikh 2026, MicroStructFormer). **Dropped:** the SIPAIM-winners test. Corrections: TEM1 is near-complete, not selective; Parikh is an arXiv preprint (NeurIPS unconfirmed); Chuter is a journal paper | 15 Nov. `docs/reports/Overlooked gaps validation.md` + §7.5 |

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
  - **Novelty check (14 Nov, [report](../reports/C3%20incomplete%20label%20learning%20novelty.md)):** the ignore mechanism is nnU-Net's own ignore label (Gotkowski 2024), and the closest prior work is LNQ 2023 (atlas-weighted loss for selectively labelled lymph nodes; recall +3 points). Add baselines: soft down-weighting, a PU loss, and a random-ignore control with the same pixel budget. H3's ≥ 10 points may be optimistic; record the outcome against it, without editing H3.
- **T-16′ Full C1 analyses** (local CPU, no GPU):
  - decomposition for every baseline (done);
  - **recall-vs-g-MAE gaming curve:** drop fibers by size or confidence and plot g-MAE against recall;
  - g-definition sensitivity;
  - HD95-variant sensitivity;
  - per-fiber morphometry agreement (Bland–Altman of g, axon diameter, myelin thickness; errors by size bin);
  - rank stability under these switches.
- **H5 first test** (only if the owner adopts H5; local, no GPU): close broken rings and regroup B1's pixels instance-first. Does simple post-processing capture most of O-sep's g gain? If yes, no new decoder is needed.

- **T-21 cv3 out-of-fold B1** (GPU tranche A, D-28): train the B1 recipe on the 3 frozen cv3 folds; predict each held-out fold. Gives D-03's 3-fold C1 numbers, H5's unseen confirmation set, and C3's out-of-fold ignore masks. No oracle or H5 analysis on non-dev images until H5c's post-processing is frozen.

- **T-22 Hidden stratification on outside data (H8; Phase 2; adopted 15 Nov, D-34):**
  1. B1 inference on TEM1 (all 20 mice; rescale 2.36 → 2.152 nm/px), TEM2 (DANDI + test ROIs) and AxonPath CC TEM, with our D-11 pairing.
  2. **Reference arm:** nnU-Net on TEM1 with one fold holding out the remyelinating-like mice (cluster 2 of the condition proxy, plus 2 control-like mice). B1 recipe, ≈ $8 from the D-28 reserve; cost stated before launch.
  3. Recall by GT fiber type (thin g > 0.85 against normal; also by fiber diameter), mouse bootstrap.
  4. Later, C3 against B1 on the same images (C3's outside test).
- **T-23 Protocol bias on the endpoint (H7; Phase 1, CPU):**
  1. outer-ring outlines for the expert-confirmed skipped fibers: semi-automatic dark-band growth from the candidate hole, with the expert checking a subset;
  2. correct the skip rate for detector recall (thin rings missed);
  3. compute Δ mean g, Δ count, Δ share g > 0.85, two-sided, bootstrap over overlap groups;
  4. per-cluster on TEM1 as context.

**Both branches:**
- **T-17 ASTIH TEM1 (day 9):**
  1. Download DANDI 001436 (`ASTIH/scripts/get_data.py --make-splits`; test subject `sub-nyuMouse26`).
  2. Apply ASTIH's released model and score it with **our** evaluator. It must reproduce Dice 0.974/0.924 and F1@0.5 0.791 (±0.01 Dice, ±0.03 F1).
  3. **Always set the matching threshold explicitly:** ASTIH's CLI default is 0.3, although its help text says 0.5.
  4. If it doesn't reproduce, fix the evaluator before trusting any result.
  5. Note that TEM1's instance GT is itself watershed-derived, which limits pairing claims there.
  6. Run our taxonomy on TEM1: is ASTIH's low precision at IoU 0.5 (0.747, against Dice 0.974) selection (our main finding, replicated) or low_iou? Use TEM1's 20 subject IDs for an animal-level bootstrap. No independent group has published on TEM1 yet ([report](../reports/ASTIH%20TEM1%20citation%20analysis.md)).
  7. *Proposed (owner to approve):* also use **AxonPath CC TEM** (Zenodo 23084690, CC-BY, 20 images, 2,074 fibers with annotated fiber ⊃ inner cylinder ⊃ axon). Unlike TEM1, its pairs are annotated, not watershed-derived, so it can test the pairing metric and H5. See [similar-datasets report](../reports/Similar%20datasets%202023%202026%20citation%20analysis.md).
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
| Selection share is a **lower bound**: a model trained on biased labels partly learns the bias (Parikh 2026) | State it with C1; T-20 expert audit and H8 give independent estimates |
| C3 looks worse on official labels even when it works ("observed-label illusion") | Judge C3 only on known-positive fibers (H3 set, H8 thin fibers) |
| H8's reference arm needs thin fibers it never saw | TEM1 fold holding out the remyelinating-like mice (T-22) |

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
- **15 Nov, Phase 0 (T-19 and the organiser questions deferred):**
  - **D-28** budget in tranches (ceiling $60).
  - **D-29** H5 pre-registered, with T-21 as its unseen confirmation set.
  - **C4 checked:** TEM2's completeness is plausible but not established, with three confounds (phenotype, model-seeded GT, in-animal split; [report](../reports/TEM2%20label%20completeness%20check.md), evidence in `docs/research_notes/TEM2 label completeness check/`). A completeness audit on TEM1/TEM2 now precedes H6.
  - New task **T-21** (cv3 out-of-fold B1 = tranche A). The C3 pilot was re-costed from ≈ $8 (one arm) to ≈ $15.4 (two arms).
  - Full rationale in §7. No compute was used and nothing was spent (total still $8.16).
  - Commits: `c829031` (decisions, report); the follow-up commit adds §7, the evidence note and the status updates in 01/02/06 and the README.
  - **Owner action open:** set a $60 Modal workspace spend limit.
- **15 Nov, Phase 0b** (evidence in `docs/research_notes/Phase 0b evidence/` 01–08):
  - **Read in full:** Gow 2026 (B4 → D-32), Chuter 2026 (now published in *Bioengineering*, not a bioRxiv preprint; corrects the 13 Nov note), Bukas 2026 (boundary-style batch effects; hard ignore is safer than pseudo-labels for C3), BMVC 2021 (an oracle ignore helps a lot, pseudo-labels help more), Vădineanu 2024 (random omission only).
  - **Read in part:** ReCo-I2P (abstract and reviews; size-based simulated omission).
  - **Bi-Polar Mask** is still paywalled; no institutional access in the browser.
  - **2026 sweep:** the MICCAI open-access index lists only 313 papers; no hit there. New must-cite: Parikh et al. 2026 (arXiv 2605.06891; names NeurIPS 2026, acceptance unconfirmed on OpenReview), a Confident-Learning audit of label bias in segmentation.
  - **AxonDeepSeg** reviewed (D2): centroid pairing, g = ADS default.
  - **Released models:** TEM1 model is leakage-free on `sub-nyuMouse26`; TEM2 is in-animal; AxonPath is tile-level (D-30).
  - **Data:** TEM1 complete (158 images, 7,841 fibers); TEM2 test ROIs (21); TEM2 DANDI and AxonPath still downloading.
  - **Label mapping** passed the by-eye test.
  - **Completeness audit:** ours u 5.1% (~83% real fibers by eye, ~60% of them abnormal) vs TEM1 u 0.6% (~88% interstitial gaps). **The C4 premise reverses:** TEM1 is near-complete, and ours is the selective set. The 13 Nov "F1 follows completeness" claim is contradicted → **D-31** H6 rewritten and pre-registered.
  - **New idea (B5):** protocol selection can bias group-mean g more than segmentation does; to test in T-16′.
  - **C3 caveat:** report FPs on protocol-excluded abnormal fibers in the pilot.
  - No compute was spent.
- **15 Nov, Phase 0b completed:**
  - **AxonPath CC TEM** downloaded (MD5 matches Zenodo), converted from GeoJSON and checked by eye. Its 8 AimSeg images are identified by name. Audit: u 0.7%, and the only real unlabelled fibers are 2 degenerating figures (protocol exclusions); no normal-looking misses.
  - **Real unlabelled per labelled fiber:** ours ≈ 4.2% ≫ TEM2 ≈ 0.3% > AxonPath ≈ 0.1% > TEM1 ≈ 0.05%.
  - **Gate 0: PASSED** (engineer, delegated):
    1. No paper does C1's error decomposition or C3's selective-omission framing. The sweep covered the MICCAI 2026 OA subset; Parikh et al. (arXiv 2026) is must-cite.
    2. External data and models are usable, with the leakage limits in note 05.
    3. H6 is written (H6a, H6b, H6c).
  - **Carried forward, not blocking:**
    - Bi-Polar Mask (paywalled; affects only the C2/H5 containment wording, Phase 3a);
    - the by-eye audit counts need a domain expert (with T-20);
    - TEM1's per-mouse condition is missing (ask the ASTIH authors, or use the GT-g proxy).
  - **Owner actions:** the $60 Modal spend limit; the Bi-Polar Mask PDF via Sir Faizan.
- **15 Nov, follow-ups:**
  - **Modal workspace usage limit set to $60** in the dashboard (Settings → Usage & billing → Usage limit; Modal's default was $105; $8.07 used). The separate card "spend limit" was left untouched.
  - **Expert review packet built (D-33):** 260 blind items, `scripts/run/expert_review_packet.py` → `data/processed/expert_review/` (local only; contains challenge crops).
  - **TEM1 groups:** the Jelescu 2016 design was read (33 EM mice: 9 Ctl, 12 Cpz6, 12 Cpz12). TEM1 holds 20, mapping unpublished. A GT proxy (3 clusters) puts the test mouse `sub-nyuMouse26` in the **control-like** cluster, so H6 on TEM1 mostly tests healthy tissue (a limitation, reported with H6). An email to the ASTIH authors is drafted, not sent.
  - **Assessment of C1/C3/H5 standings:** `docs/research_notes/Phase 0b evidence/09_standings_after_phase0.md`.
  - **Overlooked gaps (owner question), proposals awaiting a decision** (`11_overlooked_gaps.md`):
    1. run our B1 on the near-complete outside sets and measure recall by fiber type, which also gives C3 a large known-positive test set;
    2. protocol bias on mean g and fiber counts as a named C1 result;
    3. abnormal fibers as an explicit third class (a C3 design option).
  - **Overlooked-gaps validation** ([report](../reports/Overlooked%20gaps%20validation.md)): item 1 holds with a required reference arm (a complete-label model on the same images) and covers thin-myelin fibers only (cite hidden stratification, Oakden-Rayner 2020). Item 2 holds, but the skipped fibers are mixed (onion/thick and thin myelin) and no candidate has a measured g, so any rule must be two-sided; "masks a disease effect" needs per-image condition metadata (A2). Item 3 is partly pre-empted (Zepf ICLR 2023; Parikh et al. 2026, arXiv; MicroStructFormer's abnormality class), so it is a Gate 1 option only. The SIPAIM-winners C1 test is infeasible (no test GT promised; leakage through the dev fold). New: disagreement-based selection estimates are a lower bound once a model learns the labels (Parikh et al.). Still proposals; no IDs assigned.
- **15 Nov, D-34: validated gap findings integrated into the plan** (owner instruction to reflect them in current and future phases):
  - **Adopted:** H7 (T-23), H8 (T-22), the C3 known-positive evaluation rule, and the C1 add-ons. Abnormal-fiber class = Gate 1 option. SIPAIM-winners test dropped.
  - **Two design problems found while writing the rules** (§7.5):
    - H7a is arithmetically bounded (Δ mean g ≈ p·Δg; at p ≈ 5% it would need Δg ≈ 0.6), so H7b/H7c were added as the powered readouts;
    - H8's leakage-free reference (ASTIH TEM1 model, test mouse 26) holds only ~14 thin-myelin fibers, so the reference arm needs a TEM1 fold that holds out the remyelinating-like mice (≈ $8 from the reserve).
  - **Corrections propagated across the docs:** TEM1 near-complete (not selective); C3 *does* have an outside test (thin fibers, H8); Parikh = arXiv preprint; Chuter = journal; the protocol-bias premise is two-sided; the overlap ceiling = repeat-annotation agreement.
- **15 Nov, publication assessment (owner question):**
  - **Paper 1** = analysis paper (C1 + H6 + H7 + H8). Viable for a MIDL short paper, a MICCAI 2027 workshop or ISBI 2027 if T-20, T-17 and A1 pass; the MICCAI main track only with clear H7b/c and H8 effects.
  - **Paper 2, methods: possible but risky (~1 in 3).** The only open route is C3 as **MNAR-aware training**: learn the probability that a real fiber is labelled, given its appearance. No segmentation MNAR method was found.
    - It needs a **controlled benchmark on TEM1** (delete fibers the way our protocol does, then score on the full labels), because our known-positive set is too small.
    - Baselines: naive, hard ignore, LNQ, nnPU, pseudo-labels, random-ignore control.
    - About 6–10 runs: this exceeds the remaining tranches unless B and C are redirected (budget decision needed).
  - **Proposal only** (would get D-35, T-24, H9); the full text is in the owner's local plan, Phase 3b. Order: C3 pilot first, then grow it only on a pre-set win over the random-ignore control.
- **15 Nov, C3 methodology deep research** ([report](../reports/C3%20methodology%20design.md)):
  - **No in-domain or out-of-domain method conditions on *why* objects are unlabelled.** EM methods (SPOCO, WDA-Net, SomaNet, LSD masking) and detection, cell, lesion and remote-sensing methods all assume random or size-based omission.
  - **The right formal family is propensity-weighted PU under SAR**, not class-dependent MNAR. Our omission is two mechanisms: random misses (identifiable) and protocol exclusions (propensity ≈ 0; negatives for the protocol output).
  - **Recommended:** fiber-level, selection-gated soft ignore in the unchanged B1 recipe, with no pseudo-positives (Bukas).
  - **Primary evaluation:** a controlled TEM1 benchmark with two-part simulated omission, because our data has only ~18 known positives.
  - **Overlap tiles as capture–recapture** give a random-miss recall r ≈ 0.93 (independence unconfirmed).
  - **Budget:** a 4-run pilot gate + 3 runs, ≈ $33.
  - Not adopted; it feeds the proposed D-35/T-24/H9.

## 7. Decision records (full rationale; Phase 0, 15 Nov 2026)

These are the complete records behind D-28, D-29 and the C4 check, so the reasoning survives outside the owner's local `PLAN.md`. The decisions were made by the research engineer (Claude Code), with authority delegated by the owner on 15 Nov. T-19 and the organiser questions (A1–A3) were deferred by the owner.

### 7.1 D-28: GPU budget, released in tranches

**State:** $8.16 spent of the $80 Modal credit. The days 0–5 cap of $10 is retired.
**Decision:** a total ceiling of **$60 including what is already spent**, released one tranche at a time. Each tranche opens only when its condition is met, and each launch still follows the owner's planning rule (`PLAN.md` §1, rule 6): state the cost, run a 5-epoch timing run if the configuration changed, and cap with `--max-gpu-hours`.

| Tranche | What | Estimate | Hard cap | Opens when |
|---|---|---|---|---|
| A | **T-21** cv3 out-of-fold B1: 3 folds × ~6.3 L4-h | ≈ $23 | $26 | Gate 0 passed and the H5c post-processing frozen (no oracle use of non-dev images before that) |
| B | C3 pilot: hard ignore + random-ignore control, 2 × B1 recipe | ≈ $15.4 | $17 | Gate 1 passed, the power check shows H3 is testable, and the D-18 detector's recall is checked |
| C | C3 full: soft down-weighting + PU loss | ≈ $15.4 | $17 | Pilot: hard ignore beats the control on overlap-disagreement recall (paired CI > 0) |
| Reserve | Timing runs, preemption reruns, external inference if the M4 is too slow | — | ≈ $12 | Case by case, logged. **Earmarked 15 Nov (D-34):** ≈ $8 for T-22's TEM1 reference fold |

> **Correction (15 Nov, C3-design audit; the original table above is left as written).** The table did not add up: spent $8.16 + A ≈ $23 + B ≈ $15.4 + C ≈ $15.4 ≈ **$62**, already over the $60 ceiling, so the "≈ $12 reserve" never existed inside it. The C3 design report's full plan (T-21 + shared TEM1 reference + 7 C3 runs at 80 epochs) totals **≈ $72**. That is within the $80 credit but over both the ceiling and the Modal usage limit ($60), which would stop runs mid-way. **Pending owner decision:** raise the ceiling and the Modal limit to ≈ $75 (recommended), or cut runs (options in `PLAN.md` Phase 3b).

**Why this shape:**
- **Tranche A comes first because it pays three times.** D-03 promised 3-fold final numbers for C1, and so far we only have dev. H5 needs images nobody has looked at. C3's ignore masks need out-of-fold probabilities: B1's dev model has seen the 213 training images, so its probabilities there would mark almost no unlabelled fiber as confident, and the masks would be biased. So the "cheaper variant" in §4 T-15′ (dev-fold B1 only) is **ruled out** for C3.
- **The C3 pilot was under-costed:** "≈ $8" covered one arm. A pilot without the random-ignore control can't separate "targeting skipped fibers helps" from "less supervision helps", so both arms are needed.
- **C2 is outside this budget.** If Gate 3a makes C2 a candidate, that needs a new decision.
- **$60 leaves ~$20 of credit untouched** for whatever the reviewers ask for.
- **Owner action:** set a Modal workspace spend limit of $60 (Settings → Usage & Billing), so the ceiling is enforced by Modal and not only by us.

### 7.2 D-29: H5 adopted and pre-registered

**Why adopt it:** it costs no extra GPU (it rides on tranche A), it is the only route by which C2 could return, and a g-ratio finding matters to biologists more than F1 does. **Why the rule is strict:** H5 came from the dev images, and one regrouping of dev images (Cellpose+B1) has already captured ~69% of O-sep's g gain there. **No dev image can confirm anything.**

**Common rules**
- **Confirmation set:** the 213 non-dev images, predicted by the T-21 fold models (each image by the model that didn't train on it). The 53 dev images are for development only.
- **Metric:** fiber-weighted g-MAE with the challenge's equivalent-circle g, matching at IoU > 0.5 (D-06). H5 was observed on this definition. The B4 primary definition, once chosen, is reported as sensitivity only and does not decide H5.
- **CIs:** 95% paired cluster bootstrap over overlap groups, 1,000 resamples (D-22).
- **"Defective" fiber:** as in `scripts/run/day5_analysis.py`: on the raw predicted fiber, ring Euler ≠ 0, or axon touches background, or axon in more than one component.

| ID | Claim | Rule (holds if all parts hold) | Dev value (for reference, not evidence) |
|---|---|---|---|
| **H5a** (primary) | Grouping drives g error | O-sep cuts base g-MAE by **≥ 25% relative**, and the CI of Δg-MAE excludes 0 | −42% |
| **H5b** | Ring topology carries the grouping effect | Defective fibers are ≤ 35% of matched fibers but carry **≥ 40%** of total \|g error\|, **and ≥ 50% of O-sep's total \|g error\| reduction** comes from fibers defective in the base prediction | 28% of fibers / 56% of error; the reduction share was never computed |
| **H5c** (actionable) | Simple post-processing recovers it | A rule-based post-processing (ring closing + instance-first regrouping of B1's pixels), **tuned on dev only and frozen before scoring**, captures **≥ 50% of O-sep's g-MAE gain** (point estimate, with a Δg-MAE CI vs base excluding 0), **and** costs ΔF1@0.5 ≥ −0.005 and Δrecall ≥ −0.005 (lower CI bounds) | Cellpose+B1 ≈ 69%, but with −0.019 F1, which would fail the guard |

**Why these thresholds:**
- H5a's 25% is about half the dev effect. Dev-selected effects shrink on new data, and a smaller effect would not justify a method paper.
- H5c's guards stop it winning by dropping hard fibers (g-MAE ignores misses).
- Whether a given g-MAE change *matters biologically* is deferred to B5. It doesn't decide H5.

**Outcomes (Gate 3a):**

| H5a | H5c | Conclusion |
|---|---|---|
| fails | — | H5 rejected; the dev finding doesn't generalise. Report it as a negative result |
| holds | holds | **Measurement finding:** grouping matters for g, and a cheap rule fixes most of it. No new decoder |
| holds | fails | **C2 for morphometry becomes a candidate** (new budget decision) |

H5b is interpretive. It decides whether the paper may say "ring topology", or only "grouping".

**Secondary, exploratory (not confirmatory):** the same three measures on AxonPath CC TEM, the only external set with hand-annotated pairs. A model trained on those images would leak, so these results only add context.

**Amendments:** allowed only during Phase 0b, with a logged reason, and only before any T-21 prediction exists.

### 7.3 C4: TEM2 label completeness, checked

**Plausible, not established** ([report](../reports/TEM2%20label%20completeness%20check.md)):
- The labels started from AxonDeepSeg pseudo-labels that were corrected by hand. The whole scene was labelled, including unmyelinated axons and OL classes.
- But no completeness audit is reported, and the correction emphasis was fiber *separation*.
- **Three confounds** could each explain TEM2's 0.949 vs TEM1's 0.791 without any difference in completeness:
  1. phenotype: hypomyelination with normal thickness, so few abnormal profiles;
  2. model-seeded ground truth that favours model-like predictions;
  3. an in-animal test split.
- **Change:** a completeness audit on both datasets (Phase 0b) replaces the assumption, and H6 is written from its result.

### 7.4 Phase 0b decisions (15 Nov 2026)

**D-30, AxonPath CC TEM.** Approved for download and checks. Its 8 AimSeg-derived images are flagged.
- AxonPath's released `em_cns` model trained on tiles from every image, so it is **never** scored there.
- The dataset serves GT-only analyses (completeness, pairing metric on GT, H5 checks) and scoring of *our* models.

**D-31, H6 rewritten from the completeness audit, then pre-registered.**

*Why rewritten.* The plan's H6 ("selection share high on TEM1, low on TEM2") assumed TEM1 was selectively labelled ("full rings only"). Same-detector audit (`07_external_datasets.md` §4):

| | u | Candidates that are real fibers (by eye, 48) | Real unlabelled per labelled fiber |
|---|---|---|---|
| Ours | 5.1% | ~83% | ≈ 4.2% |
| TEM1 | 0.6% | ~8% | ≈ 0.05% |

The premise is reversed. Nothing had been scored on TEM1, so rewriting is legitimate. This entry is the log of it.

*H6 (C1's mechanism outside our data).* If selection errors track label incompleteness, a near-complete dataset shows a small selection share of errors.
- **H6a (primary).** On TEM1's leakage-free test subject `sub-nyuMouse26` (8 images, 503 fibers), using ASTIH's released TEM1 model (`default-TEM-model` r20240403) with **ADS's own post-processing**, scored by our evaluator at IoU > 0.5 with ASTIH's convention (border fibers scored):
  - **strict selection share** = unlabelled_real / (FP + FN) **≤ 5%** (ours, B1: 18.4%), **and**
  - **broad selection share** = (unlabelled_real + other_bg) / (FP + FN) **≤ 39%** (half of B1's 78.9%).
- **H6b (secondary).** The most frequent TEM1 FP class is **not** unlabelled_real. All classes are reported, plus the share of errors on border-touching fibers (33% of TEM1 fibers touch the border).
- **Prerequisite (gate).** T-17 first reproduces ASTIH's TEM1 numbers with that model and pipeline (±0.01 Dice, ±0.03 F1@0.5, IoU threshold set explicitly). If it doesn't reproduce, H6 is **not run** until the evaluator or pipeline difference is found.
- **Statistics.** One test animal, so there is no animal-level inference. Image-level bootstrap over 8 images, reported as a limitation. Sensitivity: the ring-candidate detector at ds 3 and 5.
- **Outcomes:**
  - H6a holds → C1's mechanism replicates: selection errors scale with how incomplete the labels are, and TEM1's low precision has another cause, named by H6b's class table.
  - H6a fails → errors on unlabelled structures arise even with near-complete labels. Part of what we call "selection" on our data may then be model over-prediction. Revisit C1's wording, and weight T-20 (by-eye audit) as decisive.
- **H6c (TEM2), added the same day after the full-image audit, before any TEM2 score.**
  - The audit: u 1.4%; by eye ~19% of candidates are real, mostly small thin-myelin fibers, so ≈ 0.3% real unlabelled per labelled fiber. TEM2 is near-complete, like TEM1.
  - Rule: on TEM2's external test set (21 ROIs, 10 mice, 292 fibers), with the released `model_seg_unmyelinated_tem` r20240708-stanford (**never** v3.0.0) and ADS post-processing, **strict share ≤ 5% and broad share ≤ 39%**.
  - Prerequisite: reproduce ASTIH's TEM2 numbers (Dice 0.971 / 0.929, F1@0.5 0.949, same tolerances).
  - Caveats, reported with the result: the test is **in-animal**, and **64% of the ROI fibers touch the frame**. Report it also with border fibers ignored (`border_to_ignore=True`).
- **Conversion detail found in the audit:** TEM2's touching rings sometimes lack the 2-px gap. The converter splits them by their myelin holes (`07_external_datasets.md` §2), checked by eye.
- **Amendments** only before the first TEM1 prediction exists, with a logged reason.

**D-32, primary g (B4) and yardstick (B5).** See the register row and `02_gow_g_ratio.md` / `06_g_effect_sizes_B5.md`.
- Gow's critique targets population-level g estimation, not per-fiber D_A/D_F. So the challenge's equivalent-circle g stays primary, and the g' cline is used for group summaries.
- Group bias matters for biology. B1's bias (−0.0002) is far below the ~0.03 smallest meaningful shift, so H5's relevance must be argued per fiber (share of fibers with |g error| > 0.03).

### 7.5 D-34: H7 and H8 rules, and the design checks behind them (15 Nov 2026)

Source of the proposals: `11_overlooked_gaps.md`, validated in `docs/reports/Overlooked gaps validation.md`. Both rules are written **before** any of their data is measured.

**H7: protocol bias on the morphometric endpoint (C1; T-23; CPU).**
- *Arithmetic check, done first.* If a share p of real fibers is skipped and their mean g differs from the labelled fibers' by Δg, including them moves the dataset mean by p/(1+p)·Δg.
  - With p ≈ 4–5% (our audit's lower bound), reaching 0.03 needs Δg ≈ 0.6–0.75, outside the range of g.
  - With p ≈ 25–33% (the EDA's estimate of ring-like structures left as background, which includes rings the detector misses), Δg ≈ 0.12 suffices.
  - So the mean-g readout is weak unless the true skip rate is high. It is kept for honesty and joined by two better-powered readouts.
- **H7a:** |Δ mean g| ≥ 0.03, two-sided; 95% bootstrap CI over overlap groups excludes 0.
- **H7b:** the protocol under-counts myelinated fibers by ≥ 5% (detector-recall-corrected skip rate; CI lower bound ≥ 5%).
- **H7c:** the share of fibers with g > 0.85 changes by ≥ 2 percentage points, two-sided.
- **Inputs:** expert-confirmed skipped fibers (D-33 categories 1–2) with traced outer rings; a detector-recall correction from a hand-traced subset.
- **Wording:** "the protocol shifts g / counts by X". "Masks a disease effect" only with per-image condition metadata (A2).
- **Prior art to cite:** stereology and nerve-morphometry inclusion bias, Fields & Dutta 2025. What we claim is the *measurement* for DL annotation protocols, not the principle.

**H8: hidden stratification induced by the protocol (C1; T-22; plus C3's outside test).**
- **Claim:** B1, trained on selective labels, misses thin-myelin fibers on near-complete outside data more than a model trained on complete labels does.
- **Rule:** with gap = recall(thin, GT g > 0.85) − recall(normal), H8 holds if gap_B1 − gap_ref ≤ −0.10 and the mouse-bootstrap 95% CI is below 0. Two-sided reporting.
- **Reference arm, design check:** ASTIH's TEM1 model is leakage-free only on `sub-nyuMouse26`, which is control-like with **~14 thin fibers** (of 324). The thin fibers are in the remyelinating-like mice (cluster 2: 261 of 1,349), which that model trained on.
  - **So the reference is our own nnU-Net on TEM1, with a fold holding out cluster-2 mice plus 2 control-like mice.** ≈ $8, B1 recipe, from the D-28 reserve; cost stated before launch.
  - B1 is scored on the same held-out mice.
- **Scope:** thin-myelin fibers only. TEM1 ("full rings") and AxonPath (AimSeg protocol) also exclude degenerating or broken rings.
- **Cite:** hidden stratification (Oakden-Rayner 2020; Bissoto, MICCAI 2025).
- **C3 outside test:** the same readout for C3 against B1, paired on the same images. It counts as "recovers thin fibers / does no harm", not recovery of deliberate exclusions.

**Other D-34 items:**
- **C3 evaluation rule:** judge C3 on known-positive fibers only (Parikh 2026 "observed-label illusion": correction looks like failure on the biased labels).
- **C1 add-ons:**
  - g-quality = F1 × (1 − scaled g error), an adaptation of PQ (Kirillov 2019); no prior missed-fiber-aware g metric was found;
  - "repeat-annotation agreement" instead of "human ceiling / inter-rater" until the organisers confirm two annotators;
  - the selection share is a lower bound.
- **Gate 1 option:** an abnormal-fiber output class (cite Zepf ICLR 2023, Parikh 2026, MicroStructFormer 2026).
- **Dropped:** scoring SIPAIM winners (no test GT; leakage through the dev fold). Kept: the novelty recheck after 18–20 Nov.
- **15 Nov, plan audit after the publication assessment** (owner asked that nothing is missed and no irrational decision is taken). Four corrections, all written into `PLAN.md`:
  1. **Venues:** ISBI 2027 four-page papers closed on 26 Oct 2026 (only one-page abstracts by 1 Feb 2027 remain). MIDL 2027 (Porto, 14–16 Jul) and MICCAI 2027 (Auckland, late Sep) deadlines are TBA. Plan backwards once they are announced.
  2. **Methods-route novelty:** the proposed "learn P(labelled | fiber appearance)" is **propensity-weighted PU learning under the SAR assumption** (Bekker & Davis 2018; Bekker, Robberechts & Davis 2019; a 2022 survey of instance-dependent PU learning). It is SAR, not MNAR. The entry above saying "no segmentation MNAR method was found" stands only as "no *segmentation application* found". The claim becomes an application + controlled benchmark + known-positive evaluation. A targeted novelty search is required before adoption.
  3. **H3 power:** known positives on our data = 18 fibers, so the C3 pilot should run on the TEM1 controlled benchmark (T-24), not on our dev fold.
  4. **Budget:**
     - a run costs ≈ $7.7 (B1 recipe), not $4–8;
     - the H8 reference run and T-24's complete-label arm are the **same run**;
     - minimal pilot (4 TEM1 runs) ≈ $62 total incl. T-21, so it needs shorter runs or a ceiling of ≈ $65;
     - the full methods route ≈ $93, above the $80 credit.
     - Decisions pending (owner): methods route, budget option, T-19, expert.
- **15 Nov, audit of the C3 methodology report** (owner: "this finding is major; update the plan"). The report's core design is accepted as the proposal (`PLAN.md` Phase 3b box rebuilt; H9 draft in §2; C3 row in §1). Four findings must be resolved before adoption:
  1. **Leakage in t_k on TEM1.** The selection-gated and hard-ignore arms need fiber probabilities on TEM1 *training* images.
     - The complete-label reference saw the deleted fibers' true labels, so it would leak.
     - The naive arm trained on those images, so it is in-sample.
     - **Use B1 zero-shot on TEM1** (already T-22's inference). T-22's inference is therefore a prerequisite of T-24, and the domain shift in t_k is a stated limitation. Alternative: 2-fold out-of-fold models (+≈ $8).
  2. **Budget:** the full plan is ≈ $72 against a $60 ceiling and a $60 Modal limit. D-28's own table already summed to ≈ $62 (corrected under §7.1). Owner decision; recommended ceiling ≈ $75.
  3. **T-20 is the anchor for the protocol score a(x)** on our data, not a deferred extra (the report calls it deferred; the packet is ready, waiting for an expert). Without it, C3 is mostly a benchmark paper.
  4. **r ≈ 0.93 (18/136 overlap discordance) mixes random misses with inconsistent protocol judgements**, so the 7% is an upper bound on pure random misses. Proposed: the expert rates the 18 disagreement fibers normal vs abnormal (an addendum to D-33); sweep r over 0.88–0.96.
  - Unchanged: ≈ 1 in 3 for a methods paper; official metrics will barely move; H3 likely fails. Order: SIPAIM novelty recheck → T-20 (+ addendum) → T-22 B1 inference on TEM1 → shared TEM1 reference → 4-run pilot → R5–R7 only on an H9 pass.
