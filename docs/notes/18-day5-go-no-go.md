# 18 — Day 5: GO/NO-GO on H1 (T-14), decided 13 Nov 2026

Inputs:
- [17](17-day4-scoring-oracles.md): B1 scores and oracles;
- `results/day4/oracles_*.json`;
- `results/day5/*.json` from `scripts/run/day5_analysis.py`.

The rule was fixed in advance (plan [09](09-research-plan.md) §3, D-15) and is applied here **as written**. Evidence that arrived later is recorded separately (§3) and does not change the decision.

## 1. Why this decision exists (for a reader new to the project)

The project's original idea was a **nested-by-construction decoder (C2)**: a network that predicts each fiber as one object with its axon inside, so that axon↔ring pairing can never go wrong. It only makes sense if pairing/separation is a real error source. The literature suggested it is: ASTIH reports Dice ≈ 0.97 / 0.92 but fiber F1 ≈ 0.79 after watershed grouping.

Our EDA suggested a rival explanation: much of the fiber-level error is **selection**, because annotators skip real fibers and deliberately leave abnormal fibers as background. Hypothesis H1 makes the choice measurable on our own strong baseline (B1). If the separation oracle doesn't pay, building C2 would optimise the wrong thing.

## 2. The rule and the result

**H1** holds (GO) only if all three conditions hold on B1. All numbers are fiber F1@0.5 on 53 dev-val images, with paired cluster-bootstrap CIs.

| Condition | Required | B1 measured | Met? |
|---|---|---|---|
| (1) O-sep gain | ≥ +0.03 | **−0.0014** [−0.0058, +0.0030] | **No.** Even the upper CI bound is 10× below the threshold |
| (2) O-sep gain ≥ O-sel gain | O-sep ≥ O-sel | O-sep −0.001 vs O-sel **+0.016** [0.009, 0.024] (strict) / **+0.074** [0.051, 0.095] (broad) | **No**, under both O-sel variants |
| (3) Separation share of errors | ≥ 30% | **0.6%** (2 of 331: 2 merged GT fibers, 0 splits, 0 gap pseudo-fibers) | **No** |

### **Decision: H1 = NO-GO.** It is unambiguous: no condition is met, and none is borderline. The risk-table rule "ambiguous → NO-GO" (09 §7) wasn't needed.

The same rule applied to the other baselines also gives NO-GO. That is supporting evidence; the rule is defined on B1.
- **AxonPath zero-shot:** O-sep +0.015 [0.008, 0.022], O-sel +0.019 / broad +0.136, separation share 9.0%.
- **Cellpose+B1:** O-sep +0.000 [−0.001, +0.002], O-sel +0.018 / broad +0.091, separation share 1.0%.

**Interpretation:**
- On this dataset, with a properly sized nnU-Net (896×1152 patches, D-24) and the shared −EDT watershed, separating touching fibers and pairing axons with rings is **solved for detection purposes**: 2 merges among 1,619 fibers.
- The detection gap that remains (F1 0.904 vs a human-vs-human 0.929) is almost entirely **which fibers count**. 78% of errors are predictions on unlabelled structures. The visual check shows about half are real (mostly thin-myelin) fibers and most of the rest are abnormal profiles the protocol excludes.
- The weaker zero-shot AxonPath does make separation errors (9%), but still far fewer than selection errors. So "pairing is the bottleneck" (claim E1 / ASTIH's reading) is not supported here.
- Two caveats on generality:
  - our B1 is not ASTIH's model;
  - TEM1's pairing GT is rule-derived (T-17 will check H4 there).

## 3. Evidence that arrived with the result (recorded, not used to change the decision)

These findings are *new* relative to the plan. They are recorded as candidates for revisiting the plan after T-14, per the plan's rule "when a result differs from its expectation, write it in the Log and update the decision; never silently rewrite the expectation".

1. **Grouping matters for g-ratio even though it doesn't matter for detection.**
   - Fixing grouping (O-sep) reduces B1's g-MAE by 42%: 0.0160 → 0.0093, paired Δ −0.0067 [−0.0083, −0.0050].
   - An instance-first grouping with *the same pixels* (Cellpose+B1) is already significantly better than the watershed on g: Δ 0.0046 [0.0031, 0.0061] in Cellpose+B1's favour.
   - It is worse on detection: −0.019 F1 [−0.028, −0.012].
2. **Broken rings carry the g error.**
   - 28% of B1's matched fibers have a topologically broken ring or an axon touching background.
   - Their g-MAE is 0.032 vs 0.0098 for clean fibers, and they carry 56% of the total g error.
   - Defects rise with fiber size (16% → 45% across diameter quartiles).
   - This is a correlation; the cause is not established ([17](17-day4-scoring-oracles.md) §6).
3. **Clean B1 fibers are near the human g floor:** 0.0098 vs 0.0067.

**What this means:**
- The *morphometric* case for a nested-by-construction decoder (closed rings, the axon strictly inside, instance-first grouping) is stronger than the *detection* case. H1 was written in terms of F1, so it could not see this.
- A g-ratio version of H1 would read: "O-sep reduces g-MAE by ≥ X and grouping explains ≥ Y% of the g error". It would have passed for B1, but it was **not pre-registered**.
- Adopting it now would be moving the goalposts after seeing the data. So it is **not** used for this decision. It is recorded as hypothesis **H5 (new, post hoc)**, for the owner to decide whether to pre-register for the next phase.

## 4. Consequences for days 6–8 (plan branch NO-GO: C1 + C3)

Per the plan:
- **T-15′ C3 properly:** learning with incomplete annotation. It needs out-of-fold B1 predictions on all 266 images (3 folds × ~6.3 L4-hours ≈ **$23**) and a retrain (≈ $7.7).
- **T-16′ Full C1 analyses:** mostly local CPU.

**Not executed:** the owner scoped this run to days 1–5.

**Budget check:** $8.16 of the $10 is spent and $1.84 is left. **T-15′ does not fit the $10 cap.**

Options for the owner:
- (a) Raise the cap. The $80 credit holds about $72.
- (b) Cut T-15′ to cheaper variants:
  - 3-fold training at a lower epoch count (e.g. 60 epochs ≈ 2.4 h each), or
  - use the dev-fold B1 alone for the dev-val part of C3 and skip out-of-fold labels for the train images.
- (c) Do the CPU-only T-16′ analyses first: the gaming curve, g-definition sensitivity and rank stability.

**Selection-related findings that feed C3:**
- The `unlabelled_real` candidate detector undercounts thin-myelin fibers (gallery, [17](17-day4-scoring-oracles.md) §7). D-18's thresholds (dark band ≥ 85% closed, d ≥ 300 nm) will likely miss them as well. Check this before building the ignore masks.
- The strict and broad O-sel results (+0.016 vs +0.074) bracket the achievable selection gain. C3's effect on official metrics should be read against this bracket.

**If H5 is adopted:** C2 comes back as a *morphometry* method. The success criterion becomes g-MAE (and ring topology) at equal F1, not separation counts. The cheapest first test costs no GPU: a ring-closing post-process on B1, plus instance-first regrouping. If simple post-processing captures most of the O-sep g gain, a new decoder is not needed.

## 5. Spend (Modal, workspace `muhammadfaheem-mansha`)

| Item | Cost |
|---|---|
| Smoke checks (3 apps, L40S + L4) | $0.02 |
| Timing run (5-epoch trainer, 18 min) | $0.37 |
| **B1** (150 epochs + validation, 6.34 h on L4) | **$7.69** |
| **Total compute, `modal billing summary` (Nov 2026)** | **$8.16 of the $10 cap**; all covered by credits, billed $0.00 |
| Volume storage (metered $6.81 this month) | $0, covered by Modal's free storage allowance |

Free storage covers the Volume today. It holds **21.4 GB** (measured 13 Nov with `Volume.listdir(recursive=True)`):
- `data/` 3.3 GB;
- `nnUNet_preprocessed/` 7.7 GB;
- `nnUNet_results/` 10.4 GB, of which 6.8 GB is B1's validation npz;
- `runs/` < 0.01 GB.

If the allowance changes, delete the npz files first; they are re-creatable with `predict`.

## 6. Status

- [x] T-14: **NO-GO**, recorded in the plan's Log with numbers, together with new hypothesis H5.
