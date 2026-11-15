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
| **H5** *(post hoc, 13 Nov, NOT pre-registered)* | Grouping and ring topology drive g-ratio error. Evidence: O-sep −42% g-MAE; instance-first grouping beats the watershed by 0.0046 g; broken rings carry 56% of g error | **Pre-registered 15 Nov (D-29)** as H5a (O-sep cuts g-MAE ≥ 25%), H5b (topology carries it) and H5c (frozen post-processing captures ≥ 50% of O-sep's gain within F1/recall guards). Confirmed **only** on the 213 non-dev images via T-21; dev values are not evidence. Full rules in §7.2 |
| **H6** *(pre-registered 15 Nov, D-31, before any TEM1 prediction exists)* | Selection errors track label incompleteness. On TEM1, which the completeness audit shows is near-complete (u 0.6% vs ours 5.1%), the selection share of errors is small. **H6a:** on `sub-nyuMouse26` with ASTIH's released TEM1 model + ADS post-processing, scored at IoU 0.5, strict selection share (unlabelled_real) ≤ 5% **and** broad share (unlabelled_real + other_bg) ≤ 39% (half of B1's 78.9% on ours). **H6b:** the modal TEM1 FP class is not unlabelled_real. Prerequisite: T-17 reproduces ASTIH (±0.01 Dice, ±0.03 F1) | Open. Full rule in §7.4. The earlier informal H6 ("high on TEM1, low on TEM2") was **withdrawn before any data**: the audit reversed its premise |

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
  - **2026 sweep:** the MICCAI open-access index lists only 313 papers; no hit there. New must-cite: Parikh et al., NeurIPS 2026 (Confident-Learning audit of label bias in segmentation).
  - **AxonDeepSeg** reviewed (D2): centroid pairing, g = ADS default.
  - **Released models:** TEM1 model is leakage-free on `sub-nyuMouse26`; TEM2 is in-animal; AxonPath is tile-level (D-30).
  - **Data:** TEM1 complete (158 images, 7,841 fibers); TEM2 test ROIs (21); TEM2 DANDI and AxonPath still downloading.
  - **Label mapping** passed the by-eye test.
  - **Completeness audit:** ours u 5.1% (~83% real fibers by eye, ~60% of them abnormal) vs TEM1 u 0.6% (~88% interstitial gaps). **The C4 premise reverses:** TEM1 is near-complete, and ours is the selective set. The 13 Nov "F1 follows completeness" claim is contradicted → **D-31** H6 rewritten and pre-registered.
  - **New idea (B5):** protocol selection can bias group-mean g more than segmentation does; to test in T-16′.
  - **C3 caveat:** report FPs on protocol-excluded abnormal fibers in the pilot.
  - No compute was spent.

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
| Reserve | Timing runs, preemption reruns, external inference if the M4 is too slow | — | ≈ $12 | Case by case, logged |

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
- **TEM2.** Deferred. The 600×600 test ROIs are too small for the detector. The TEM2 rule is written after auditing the full DANDI images, before any TEM2 score.
- **Amendments** only before the first TEM1 prediction exists, with a logged reason.

**D-32, primary g (B4) and yardstick (B5).** See the register row and `02_gow_g_ratio.md` / `06_g_effect_sizes_B5.md`.
- Gow's critique targets population-level g estimation, not per-fiber D_A/D_F. So the challenge's equivalent-circle g stays primary, and the g' cline is used for group summaries.
- Group bias matters for biology. B1's bias (−0.0002) is far below the ~0.03 smallest meaningful shift, so H5's relevance must be argued per fiber (share of fibers with |g error| > 0.03).
