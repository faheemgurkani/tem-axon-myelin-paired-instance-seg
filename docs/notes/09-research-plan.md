# 09 — Research Plan (v2, revised 12 Nov 2026)

v1 (10–11 Nov) was "beat the challenge with a nested decoder and a g-ratio loss". Three things changed it:

- **The gap-validation report** (`docs/reports/Paired axon myelin gap validation.md`) narrowed the novelty claim.
- **We now have the challenge training data.**
- **The EDA** ([10](10-eda.md), [11](11-deep-eda.md)) showed that labels and data, not only the model, shape the error.

We are not bound by the challenge timeline. The time budget is **2 weeks (10 working days)** of focused work.

## 1. What we now believe (evidence → consequence)

| Evidence | Consequence for the plan |
|---|---|
| ASTIH's 0.79 fiber F1 measures fiber *detection*, not axon↔ring pairing. Nobody measures pairing correctness | "Pairing is the bottleneck" is a **hypothesis to test**, not a premise |
| AxonPath (Sep 2026) already predicts fiber + axon jointly, then pairs them with a 90% overlap rule | AxonPath is a required baseline; "joint prediction" alone is not novel |
| Nested polar contours (Alblas 2022, NPS-Net 2026, Bi-Polar Mask 2022) and ratio losses (Meng 2022) exist | The dual-radius head and the g-ratio loss **cannot be headline claims** |
| GT pairs are clean (1 axon in 1 ring, 99.5% enclosed) but 71% of fibers touch a neighbour | Errors will come from **separating touching fibers**, not mismatching axons |
| Interstitial gaps between touching fibers look like axon-in-ring | A concrete failure mode that by-construction decoding should fix, and a measurable one |
| ~4–8% of real fibers are unlabelled. Interior label 1 marks large or oblique fibers | **Fiber selection** is a hidden error source. Background is not clean negatives |
| Re-annotated overlaps: myelin Dice ~0.93–0.96, g noise ~0.006. g-definition spread 0.065 | Hard ceilings: gains smaller than these are noise, and the evaluator definition must be exact |
| Fibers are large (median axon 562 nm), so 1 px moves g by only ~0.006 | The g-ratio loss is a minor ablation, not a contribution, on this data |
| 40 images form 18 overlapping-tile groups | Group-aware splits are mandatory |
| Challenge data licence is unstated | Headline numbers must also be reproducible on public data (ASTIH TEM1, AxonPath) |

## 2. Revised intention

**Research question.** In paired axon–myelin instance segmentation of TEM, how much of the fiber-level error comes from
(a) **fiber selection** (what annotators chose to label), (b) **separating touching fibers / pairing**, and (c) **boundary
placement**, once annotation noise is accounted for? And does a decoder that makes each fiber's axon and ring nested by construction
remove the (b) part?

The project is now **evaluation-first, method-second**. The evaluation contribution stands on its own whatever the method result.

| Contribution | Status | Risk |
|---|---|---|
| **C1. Error-decomposition evaluation for paired fibers.** Pairing-correctness metric, FP taxonomy (unlabelled real fiber / interstitial gap / label 1 / other), selection vs separation vs boundary decomposition via oracle experiments, annotation-noise ceilings from overlapping tiles, g-MAE gaming curve (recall vs g-MAE) and g-definition sensitivity | **Core, always delivered** | Low: no comparable benchmark analysis exists (gap report §C) |
| **C2. Nested-by-construction decoder** (instance-first, part-second: filled-fiber + fiber separation + inner/outer part head). Judged on touching-fiber and interstitial-gap errors, not just Dice | Delivered only if the go/no-go passes | Medium: must beat nnU-Net + watershed *and* AxonPath beyond the noise ceiling |
| **C3. Learning with incomplete fiber annotation.** Ignore-masking or down-weighting confident unlabelled rings in training (a partial-label view) | Cheap add-on, mainly if the go/no-go fails | Medium: the official GT has the same misses, so gains may show only in our diagnostics |

**Dropped or deferred:**
- dual-radius polar head
- self-supervised pretraining
- μSAM and the full 5-baseline set
- H3 "small fibers" on challenge data (only ~190 fibers under 300 nm)
- any challenge-style submission

## 3. Hypotheses (falsifiable)

- **H1 (go/no-go).** On the nnU-Net + watershed baseline, oracle *pairing/separation* recovers at least as much fiber F1@0.5 as oracle *selection*,
  and the recovered amount exceeds the re-annotation noise (fiber-level disagreement ~8%).
  - **Rejected** → pivot: C1 + C3. Pairing is not the bottleneck on this data, and that is itself the finding.
- **H2.** The nested decoder cuts interstitial-gap FPs and merge/split errors versus nnU-Net + watershed and AxonPath, without losing compartment Dice (within CIs).
- **H3.** Ignore-masking unlabelled confident rings raises recall on overlapping tiles' disagreement fibers without lowering official metrics beyond CI.
- **H4.** The ordering of methods on challenge data is reproduced on ASTIH TEM1 (and AxonPath where labels allow).

## 4. Two-week schedule

| Day | Work | Output / exit check |
|---|---|---|
| 1 | Data loader; group-aware splits: one dev split (80/20 by overlap group) plus 3-fold GroupKFold for final numbers, frozen in `configs/splits/` | Split JSONs committed; no overlap group crosses a split |
| 2 | Evaluator: official replica (Dice, HD95 nm, g-MAE eq-circle with IoU matching), fiber F1/PQ, **pairing-correctness**, **FP taxonomy**, recall next to g-MAE | Unit tests; GT-vs-GT perfect; overlap pairs reproduce the ceilings in [11](11-deep-eda.md) |
| 3 | Start baselines: **B1** nnU-Net 2D (semantic, dev fold) + shared watershed pairing; **B3** Cellpose fibers + semantic split; **AxonPath** zero-shot (rescale 2.152→8 nm/px) | Training running; AxonPath predictions scored |
| 4 | Score B1/B3/AxonPath; **oracle experiments**: GT selection (drop FPs on unlabelled/label 1), GT separation (GT fiber outlines + predicted semantics), GT boundaries | Error-decomposition table |
| 5 | **GO / NO-GO on H1.** Write the decision and numbers into the log | Decision recorded |
| 6–8 | **GO:** nested decoder on the nnU-Net backbone (C2), plus ignore-masking (C3) as one ablation. **NO-GO:** C3 properly + full decomposition across all baselines + g-MAE gaming and definition analyses | Model or analysis results with bootstrap CIs over images |
| 9 | ASTIH TEM1 replication (B0 check: Dice 0.974/0.924, fiber F1 0.791), then the best method / analysis on TEM1 | H4 table |
| 10 | Write-up: 4-page short paper draft (MIDL short-paper format) + figures; update notes | `paper/` draft |

**Compute.** This assumes access to one GPU for nnU-Net 2D. Without one, B1 runs on downsampled data and C2 shrinks to a
lighter backbone. Settle this before day 3.

## 5. Rules that come from the EDA (apply everywhere)

- Splits use `configs/splits/overlap_groups.json`. Score full images, never tiles.
- Ignore label 1 in loss and metrics. Never assume consecutive IDs. Skip orphan IDs.
- g uses the equivalent-circle definition only. Report recall next to g-MAE.
- Use no edge-snapping post-processing, because the labels follow a smooth convention.
- Augment brightness, contrast and gamma, illumination gradients, noise (σ 7–22) and black corners.
- Report bootstrap CIs. Read every difference against the ceilings (myelin Dice ~0.93–0.96; g ~0.006).

## 6. Open items

- Ask the organisers whether results on their data may be published, and under what licence ([08](08-open-questions-and-risks.md)).
- Bi-Polar Mask full text, for the C2 novelty statement.
- AxonPath preprint, which is still pending; re-check it before submission.

## Log

- 2026-11-10: repo scaffolded; challenge site crawled; literature review preserved.
- 2026-11-11: deep-research pass done (5 research streams plus a final report in `docs/reports/`). Notes 03–08 and `configs/default.yaml` updated. **Next:** inspect the 266 training masks (inner tongue, sheath splitting, scale bar, filename metadata), then implement `evaluation/official.py` and validate it against ASTIH TEM1 (B0).
- 2026-11-11: feasibility check for a 2-week budget. The gap still holds: a fresh search found no paper on nested/paired axon–myelin instances or g-ratio-trained segmentation. **Biggest risk:** access to the challenge data, since we are not registered. **Fallback:** run the whole study on public CC-BY data (ASTIH TEM1 + AxonPath). Two-week scope: evaluator + B1/B3 + oracle-pairing (week 1), then the nested decoder + g-ratio loss vs B1 (week 2). The dual-radius head, SSL pretraining and μSAM are deferred.
- 2026-11-12: gap-validation pass (`docs/reports/Paired axon myelin gap validation.md`). The gap is open but narrower than first stated. AxonPath (Sep 2026) is a multi-head joint model, but it pairs with a 90% overlap rule. NPS-Net, Alblas 2022 and Bi-Polar Mask already cover nested polar contours, so the dual-radius head is no longer the novelty. Meng 2022 is prior art for ratio losses. Novelty now rests on dense multi-instance nesting by construction, a per-fiber g-ratio loss and a pairing-correctness metric. Plan changes: add AxonPath as a week-1 baseline; oracle pairing becomes the go/no-go experiment; still need the Bi-Polar Mask full text.
- 2026-11-12: training data received (266 images + masks) and EDA done ([10](10-eda.md)). Conventions are now settled: border-cut fibers are label 1, the inner tongue is part of the axon, IDs are not consecutive, and 47 orphan labels exist. **New risk:** selective annotation. Abnormal myelinated fibers are background (~25% of ring-like structures), so fiber *selection* becomes a hidden task, and the go/no-go experiment must separate selection errors from pairing errors. Fibers are large (median axon 562 nm), so g is less sensitive than the report assumed.
- 2026-11-12: deep EDA ([11](11-deep-eda.md)). Found **leakage**: 22 overlapping tile pairs in 18 groups (40 images), saved in `configs/splits/overlap_groups.json` and to be used as split groups. **FP patterns:** ~4–8% of real fibers are unlabelled (random misses, not a rule), interstitial gaps between touching fibers act as pseudo-fibers, and interior label 1 marks large or oblique fibers. **Ceilings from re-annotated overlaps:** myelin Dice ~0.93–0.96, g-MAE floor ~0.006. The g definition spread (0.065) is 10× annotation noise. Next: an evaluator with an FP taxonomy (unlabelled fiber / interstitial gap / label 1 / other).
- 2026-11-12: **plan v2.** The project becomes evaluation-first (C1 error decomposition, always delivered), method-second (C2 nested decoder, only if the H1 go/no-go passes on day 5), with C3 incomplete-annotation handling. Dual-radius head, SSL, μSAM and the g-ratio-loss headline are dropped.
