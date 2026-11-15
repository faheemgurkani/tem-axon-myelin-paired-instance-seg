# Can AlphaGenome 2 ideas help C3? Claim-by-claim check (16 Nov 2026)

**Source:**
- Gayoso, Mottram, Karollus et al. (Google DeepMind), "AlphaGenome 2 predicts regulatory activity and variant effects across cell states".
- Local PDF: `docs/references/papers/alphagenome-2.pdf`, read with `pdftotext`.
- Venue not stated in the PDF; treat it as a preprint or technical report.

**What was checked:** an assessment, pasted by the owner, of whether AG2's methods transfer to C3. It came with a description of AG2.

**Context:** the C3 design is in `docs/reports/C3 methodology design.md` §3.

## 1. Is the description of AG2 accurate? Yes

Verified against the PDF text (Results p. 2–3; Methods "Stage 3: Distillation", "Post-training"):
- a U-Net-style backbone on 1-Mb DNA;
- **adaptive normalization layers** modulated by a condition embedding, either from a learned dictionary of reference conditions or from a cell encoder;
- four training stages: pre-training, condition refinement, distillation, post-training;
- **eight teachers**, distilled into one student "to reproduce the ensemble-averaged predictions of the teachers on randomly mutated input sequences";
- post-training freezes the student; a trainable **adapter** maps embeddings from a separate **scVI-inspired autoencoder** into the condition space;
- unseen states are conditioned on the **mean embedding of k cells**.

## 2. How AG2 itself handles missing data: plain masking

- "Loss functions across all heads were evaluated exclusively over observed entries by masking both predictions and targets with M" (Methods, training losses).
- Unmeasured CpG sites were "masked during training".

These are entries missing **by design** (assays never run). They are not entries missing because of what they look like. That is the same mechanism as nnU-Net's ignore label: AG2 has **no selection model**. Its conditioning answers "predict in a new setting", not "why is this label missing". **The pasted verdict ("main idea doesn't fit") is confirmed.**

## 3. Component claims

| Claim (pasted) | Check | Verdict |
|---|---|---|
| Teacher ensemble → better-calibrated t_k; "~75% usable, nearly free, you already have fold models from T-21" | **On disk:** only the dev `fold_0` B1 model exists (`data/nnunet/results_modal/…/fold_0`). T-21 is not run (tranche A, ≈ $23). **Leakage:** in cv3, each training image is in the training set of 2 of the 3 fold models, so only its single out-of-fold model is clean. Averaging fold models gives a valid t_k only on images no fold saw (TEM1, the challenge test set). On the TEM1 benchmark an ensemble means extra paid runs. Distillation (AG2's real purpose: one cheap network) brings C3 nothing. Free calibration that does exist: nnU-Net's test-time mirroring (default on; `tem_modal.py` doesn't disable it), and temperature scaling on dev. B1's calibration itself is **untested**: the probabilities are on the Modal Volume | **Overstated.** Not free and not leakage-safe on our training images. Useful only on outside images. Usability ≈ 30–40%, not 75% |
| Mean labelled-fiber embedding → a_k by distance, CPU, "~70%", ≈ SAMPD gate | **It is already in the C3 design** (report §3, step 3: "a calibrated novelty score against the labelled-fiber feature distribution, with SAMPD-style thresholds"), so it is not a new contribution from AG2. **CPU feasibility test (16 Nov, below):** on the 433 audit candidates (≥ 300 nm), **24%** fall inside the 95% region of 7,704 labelled, enclosed, non-border fibers (Mahalanobis on log inner diameter, eccentricity, solidity). The by-eye audit expects ≈ 33% real normal-looking (83% real × ~40% normal), so the order of magnitude is plausible. **Two limits found:** (1) the distance is **one axis**: protocol-excluded abnormal fibers and non-fibers (gaps) both fall "far", so it cannot do the three-way split alone; it needs the fiber probability t_k. (2) **Measurement mismatch:** candidates come from the threshold detector and labelled fibers from GT masks (solidity 0.87 vs 0.99), so a fair gate must compute features the same way for both, or use deep features. The threshold needs labelled validation (T-20 or overlap fibers) | **Feasible as one input, not validated.** Usability ≈ 60%; success unproven until T-20 |
| FiLM / adaptive-norm "protocol code", ~20% | Agree: it needs T-20 abnormal labels, a new trainer and runs beyond $34, and it mixes an architecture change into C3. Attribution fix: Parikh 2026 uses FiLM; Pionono uses rater codes; MultiTalent uses per-dataset heads (not FiLM); MicroStructFormer has an abnormality *class*, not conditioning | **Agree (low)** |
| Domain conditioning for TEM1 / outside data, ~35% | Agree: this is domain shift (relevant to H8 and generalisation), not label omission. Out of scope for C3 | **Agree (out of scope)** |
| AG2 as the C3 method: 10–15%; overall methods-paper odds stay ~1 in 3 | Judgement; not testable now | **Agree** |

## 4. Test details (reproducible, CPU, seconds)
- **Inputs:** `data/processed/eda/fibers.csv` (filters: not border, axon_enclosed > 0.99; n = 7,704) and `data/processed/eda/unl_cands.csv` (filter: d_nm ≥ 300; n = 433, the same count as the Phase 0b completeness audit).
- **Features:** log inner/axon diameter, eccentricity, solidity.
- **Method:** Mahalanobis distance from the labelled mean; threshold = the labelled 95th percentile (2.94).
- **Results:**
  - share of candidates inside: 24.2%;
  - medians, candidate vs labelled: inner diameter 579 vs 561 nm; eccentricity 0.74 vs 0.73; solidity 0.87 vs 0.99.
- **Without the ≥ 300 nm filter** (all 2,062 EDA holes, median 148 nm), only 5.3% fall inside. Mostly small holes and gaps, as expected.

## 5. Bottom line for the plan
- Keep the C3 design unchanged.
- **Calibration:** use mirroring TTA plus temperature scaling on dev. Use a fold ensemble only on images no fold saw.
- **The novelty-gate input:** compute the features the same way for labelled fibers and candidates, and tune its threshold on expert or overlap labels.
- Don't adopt AG2's conditioning or distillation.

## 6. Closer look at the three conditioning components (16 Nov, owner request)

**Description check:**
- **Learned dictionary of reference conditions:** correct ("selected from a learned dictionary of ontology-defined reference conditions").
- **Adaptive normalization:** correct (scale and shift of intermediate features, ref. 26).
- **Cell encoder, trainable adapter, mean of k cells, no retraining for a new dataset:** correct at inference. The adapter and head *are* trained once, in post-training, with the backbone frozen.

**The C3 question each component would answer:** *does the chance that a real fiber is skipped depend on the image (its condition), and not only on the fiber?* If so, the selection model needs a condition term.

### Test: per-image heterogeneity of the unlabelled-candidate rate (ours, 266 images, CPU)
- **Inputs:** `results/phase0b/completeness_ours.csv` (candidates ≥ 300 nm, labelled fibers, background area) and the per-image median g from `fibers.csv`.
- **Naive view (misleading):**
  - the per-image share u = c/(c+n) is strongly over-dispersed (binomial χ²/df = 2.77, p ≈ 1e-45; median 3.6%, IQR 0–7.7%, 34% of images have none);
  - u correlates with fiber density (ρ = −0.43) and with median g (ρ = +0.17).
- **Controlled view (correct):** candidates arise in background, so model counts with a background-area offset (quasi-Poisson).
  - **density:** rate ratio 0.88 per SD (CI 0.75–1.04);
  - **median g:** 0.95 per SD (CI 0.81–1.11).
  - Neither is significant. **The naive correlations were an area artefact.**
- **But heterogeneity remains:** dispersion is **2.2×** Poisson after the covariates. Some images have far more unlabelled candidates than others, for reasons that neither density nor tissue g explains: annotator or session, image contrast, or detector behaviour. These can't be separated without the expert labels.

### Verdict per component
1. **Condition embedding (lookup per known condition) → useful, in a cheap form.**
   - The selection model should carry a **per-image (or per-overlap-group) random effect** on the skip rate. The fitted dispersion justifies it.
   - Implementation: a hierarchical logistic or beta-binomial term in the CPU selection model (step 3 of the C3 design). No network change.
   - Caveat: part of the dispersion may come from the detector, not the annotators. Re-fit on expert-confirmed fibers after T-20.
2. **Adaptive normalization inside the segmentation network → don't.**
   - The selection model is deliberately separate from B1, so conditioning B1 would mix an architecture change into the C3 test.
   - Conditioning on an image term of unknown cause risks letting the network learn "skip in this image", which is exactly the bias we want to avoid.
   - No budget for it.
3. **Encoder + averaged embeddings for unseen data → two uses, one fails.**
   - **(a) Works as an analysis tool:** averaging fiber or patch features per image gives an *image fingerprint*. Clustering it is a proxy for the missing condition metadata (A2), the same idea as the TEM1 mouse proxy (note 07 §6). Useful for H7 and H8 stratification.
   - **(b) Fails for zero-shot protocol transfer:** AG2 learned its condition space from thousands of fully measured conditions. We have 4 datasets and only 1 selective protocol, so there is no way to learn a "fingerprint → labelling protocol" map. The TEM1 benchmark must simulate the protocol explicitly.

### Consequence for H7 (proxy only)
- Under the area-controlled model, the candidate rate does **not** rise with image median g. So this proxy gives **no support** for "the skip rate grows with pathology", the differential-bias wording.
- Keep H7's two-sided rule. The "masks disease" wording still needs A2.
