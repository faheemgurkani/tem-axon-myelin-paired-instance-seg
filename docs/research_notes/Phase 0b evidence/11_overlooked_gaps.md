# Overlooked contribution gaps: research engineer's honest assessment (15 Nov 2026)

**Question (owner):** after all the reading, datasets and runs, is there a contribution-specific gap we missed?

**Basis:** everything in `docs/` (reports, research notes, references), `external/` and `data/`. A grep of the plan confirms none of items 1–3 is an existing task (H4 is only a ranking claim).

**Status:** **proposals, not decisions.** The owner decides; IDs are assigned only on adoption.

## 1. Measure the harm of selective labels: our B1 on near-complete outside data (highest value)

- **Idea:** run the existing B1 (inference only) on TEM1, TEM2 and AxonPath CC TEM, which label the thin-myelin and abnormal fibers our protocol skips. Measure **recall by fiber type** (binned by GT g, e.g. g > 0.85 against normal, and by fiber diameter).
- **Prediction:** B1 misses thin-myelin fibers disproportionately, because its training labels taught it they are background. That is the concrete harm behind C1.
- **Also fixes C3's test problem.** Note 09 said C3 is testable only on our data. That is too pessimistic: these sets are a large known-positive test set for exactly the omitted kind of fiber. C3 against B1, *paired* on the same outside images, is a stronger test than the 129-fiber overlap set.
- **Leakage:** B1 never saw any outside image, so all TEM1/TEM2/AxonPath images are usable. The `sub-nyuMouse26` restriction applies only to ASTIH's model.
- **Confounds:** pixel size (rescale to 2.152 nm/px), contrast and lab. Interpret only paired contrasts: thin against normal fibers within a dataset, and C3 against B1.
- **Cost:** inference on about 190 images. MPS locally (ask first: heavy) or Modal (< $2).

## 2. The labelling protocol biases the biological endpoint (elevate the B5 idea)

- Skipped fibers are mostly abnormal or thin-myelin (high g). So the labels shift mean g, and fiber counts, in a way that grows with pathology. That is a differential bias that can mask a disease effect.
- **Measurable without models:** GT + candidate detector + expert categories (D-33).
- **Four protocols to compare:** ours, TEM1 "full rings", TEM2 model-seeded, AimSeg/AxonPath "exclude degenerating". No one has compared them.
- **Yardstick:** ≈ 0.03 is a meaningful group shift (note 06).
- Audience: biologists. It connects C1 to Spalińska 2026, Gow 2026 and Fields & Dutta.

## 3. Abnormal fibers as a target, not noise (alternative C3 design)

- Every dataset reviewed excludes abnormal or degenerating fibers in some way, and in disease they are the most informative.
- **Option:** a 3-way output (normal fiber / abnormal fiber / background). The official map keeps the protocol (abnormal → 0), so precision is not hurt (the note 03 caveat), and abnormal fibers are counted and measured separately.
- **Labels:** the expert packet already asks normal against abnormal; the ring-candidate detector proposes locations.
- **Limit:** a few hundred expert labels is a pilot, not a training set. Decide at Gate 1 together with C3.

## 4. Covered but under-framed
- **A recall-aware g metric** (count missed fibers, or a g-error-at-recall curve): a concrete metric contribution inside C1's gaming curve.
- **Human ceiling from overlapping tiles:** a reusable inter-rater method without a second annotator. State it as a contribution.
- **Not gaps:** pairing (H1 settled), nested representations and ratio losses (prior art), semi-supervised use of the 134 test images (generic).

## Risks underlined
- **Licence (T-19):** the selection effect lives only in our dataset. Items 1–2 reduce the dependence by producing outside-data results.
- **SIPAIM 2026 entries (18–20 Nov):** check before any "first on this data" claim.

## Recommendation
- Item 1 → an explicit Phase 2 task (inference only).
- Item 2 → a named C1 analysis in T-16′, with a pre-registered rule before running.
- Item 3 → a C3 design option for Gate 1.

Each gets an ID (next free: D-34, T-22, H7) only when the owner adopts it.
