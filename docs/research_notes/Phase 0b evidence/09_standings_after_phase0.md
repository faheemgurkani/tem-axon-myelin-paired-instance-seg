# Where C1, C3 and H5 stand after Phase 0 and Phase 0b: research engineer's assessment (15 Nov 2026)

> **Corrections, 15 Nov (after `docs/reports/Overlooked gaps validation.md`; D-34). Read these first; the text below is kept as written.**
> 1. **C3 is not limited to our data.** The outside sets label thin-myelin fibers our protocol skips, so H8 gives C3 an outside known-positive test (thin fibers only).
> 2. **C3 evaluation:** judge it only on known-positive fibers ("observed-label illusion", Parikh 2026).
> 3. **Parikh et al.** is an **arXiv preprint** (2605.06891). It names NeurIPS 2026, but OpenReview has no acceptance record.
> 4. **New C1 readouts:** H7 (protocol bias on g and counts, two-sided) and H8 (hidden stratification: B1 vs a complete-label reference). Add-ons: PQ-style g-quality, repeat-annotation agreement, lower-bound note.
> 5. The "biased low" g premise is withdrawn (two-sided now). The abnormal-fiber class is a Gate 1 option only. The SIPAIM-winners test is dropped.

Scope: everything known after Phase 0 (decisions D-28, D-29, C4) and Phase 0b (notes 01–08 and 10 here, decisions D-30…D-33).
- No new model has been trained or scored since day 5. The *evidence about* the claims changed; the claims' own numbers did not.
- Day 1–5 numbers are in [05](../../notes/05-results.md).

## Summary

| | Claim | Before Phase 0 | After Phase 0b | Confidence | What decides it next |
|---|---|---|---|---|---|
| **C1** | Errors here come from *which fibers were labelled* (selection), not from linking axon to ring | Strong on our data; novelty plausible | **Stronger and sharper.** Novelty holds; selection is now shown to be specific to our labelling protocol; an outside test (H6) is written in advance; a new biological angle (selection biases mean g) | Medium-high, pending checks | T-20 expert audit (D-33), T-17 reproduction, H6 |
| **C3** | Training that tolerates deliberately skipped fibers recovers them | Plausible; budget missing | **Viable but modest and narrower.** Budget approved in tranches; the framing is still new; the mechanism is borrowed; only our data can test it | Medium-low for a large gain | Gate 1, power check, pilot vs random-ignore control |
| **H5** | Bad grouping and broken rings drive the g-ratio error | Post-hoc observation | **Pre-registered, untested.** Its practical importance is narrower than hoped: it matters per fiber, not for group averages | Low until tested | T-21 predictions, frozen post-processing, H5a–c |

## C1: error decomposition (the main contribution)

**What strengthened it:**
- **Novelty re-checked.**
  - No 2025–26 paper decomposes fiber errors by cause or measures pairing.
  - Chuter 2026 (published) shows tool scores collapse on new data while correlations hide it. That argues for per-object error analysis.
  - Spalińska 2026 already called for this kind of benchmark.
  - Closest new work: Parikh et al., NeurIPS 2026. It audits *label bias* by comparing labels with a model's confident predictions. That is related to how we measure selection, but it doesn't decompose errors by cause. **Must cite.**
- **Selection is protocol-specific, now measured.** Real unlabelled fibers per labelled fiber, same detector, checked by eye:

  | Ours | TEM2 | AxonPath | TEM1 |
  |---|---|---|---|
  | **≈ 4.2%** | ≈ 0.3% | ≈ 0.1% | ≈ 0.05% |

  - About 60% of ours look abnormal, the rest normal. This matches the EDA (abnormal fibers skipped on purpose, plus random misses).
  - It turns C1 from "selection dominates here" into a testable mechanism: **selection errors appear where labels are incomplete.**
- **An outside test is pre-registered (H6, D-31)** before any external number exists. On near-complete TEM1 and TEM2, selection should be a small share of errors (strict ≤ 5%, broad ≤ 39%, against 18% and 79% on ours).
- **New biological angle (B5).** Skipped fibers are mostly abnormal, with thin myelin and high g. The protocol can therefore shift a group's average g by more than segmentation error does: B1's group bias is only −0.0002, against the ~0.03 smallest meaningful shift. Planned as a T-16′ analysis. If it is ≥ 0.03, C1 has a direct biological consequence.

**What weakened or narrowed it:**
- **The 13 Nov argument "fiber F1 follows label completeness" is wrong.** TEM1 is the most complete set yet has the lowest published F1 (0.791). C1 must not lean on cross-dataset F1 comparisons.
- **"Selection dominates" may hold only for selective protocols like ours.** That is fine, but the claim must be worded as protocol-conditional.
- **TEM1's leakage-free test mouse looks like a control** (proxy, note 07 §6). H6 on TEM1 mostly tests healthy tissue, which limits how general a pass is.

**Still unverified, and decisive:**
1. **T-20:** are B1's "unlabelled_real" predictions really fibers? Claude's 16-image gallery said about half of `other_bg` were real. The packet for a domain expert is ready (D-33). The pass rule is ≥ 70% real.
2. **T-17:** our evaluator must reproduce ASTIH's published TEM1 numbers with their pipeline before any external claim.
3. **Licence (T-19, deferred by the owner):** without it the paper must lean on TEM1/TEM2/AxonPath, and those have little selection to show. **This is now the largest strategic risk to C1**, because the effect lives only in our dataset.

**Verdict:** C1 is the paper. It is novel, has a mechanism and has an outside test, provided T-20 passes and the licence is obtained.

## C3: learning with selectively omitted labels

**What holds:**
- **Novelty of the framing, re-checked.** Vădineanu 2024 and ReCo-I2P 2025 (and BMVC 2021) simulate omission at random or by size. No method found targets *deliberate, protocol-driven* omission, and none tests against a set of fibers known to be real (our overlap-disagreement fibers).
- **Budget is approved in tranches (D-28).** The pilot (~$15) runs only after Gate 1 and a power check.

**What narrowed it:**
- **Only our dataset has this kind of omission**, so C3 can't be shown on TEM1, TEM2 or AxonPath beyond "does no harm".
- **The mechanism is not ours** (nnU-Net's ignore label). Expected gains in the literature are +2–3 points (LNQ). H3 (≥ 10 points) is probably optimistic and is recorded as is.
- **Two kinds of unlabelled fiber** (note 03):
  - random misses of normal fibers, which should be learned;
  - deliberate exclusions of abnormal fibers, which the protocol calls negatives.

  Ignoring both can teach the model to predict abnormal fibers, lowering official precision. The pilot must report this.
- **Design guidance from the reading:**
  - Bukas 2026: mixing model-made and human masks creates shape batch effects, so *hard ignore* is the safer primary arm.
  - BMVC 2021: pseudo-labels can beat even perfect ignoring, so they are a stronger but riskier comparison arm.

**Verdict:** a reasonable second contribution or a short method study. It is unlikely to be a headline unless the pilot clearly beats the random-ignore control.

## H5: grouping and ring topology drive g-ratio error

**Status:** pre-registered (D-29) as H5a/H5b/H5c. Confirmation is allowed only on the 213 non-dev images from T-21, after the post-processing is frozen. **Not tested yet.**

**What Phase 0b changed:**
- **Gow 2026 does not undermine the g we use.** His critique targets population fits, not per-fiber D_A/D_F. Primary g is unchanged (D-32).
- **Practical importance is narrower.** For group comparisons the average error matters, and B1's is tiny. H5 matters for **per-fiber** analyses (distributions, outliers). Its report must include the share of fibers with |g error| > 0.03.
- **AxonPath CC TEM** (hand-drawn pairs) gives H5 an outside, exploratory check on GT and our models. AxonPath's own model is excluded for leakage.
- **Bi-Polar Mask** is still unread. It matters only if H5 leads back to the nested decoder (C2).

**Verdict:** a cheap, honest test is set up. If H5a fails, we report a negative result. If H5a and H5c hold, it's a measurement finding without a new model. Only H5a alone reopens C2.

## Readiness: what still bars paper-ready results

| Item | Blocks | Owner / action |
|---|---|---|
| Domain-expert ratings (T-20 + audit), D-33 | Gate 1, H6 premise | **Owner:** find an expert (via Sir Faizan); send the local packet |
| Licence for the challenge data (T-19) | Publishing anything on our data | **Owner** (deferred by choice) |
| T-17 reproduction | H6, every external claim | Engineer, Phase 2 (CPU/MPS) |
| TEM1 per-mouse groups | Interpreting H6 on TEM1 | Owner may send the drafted email (note 07 §6); proxy meanwhile |
| Bi-Polar Mask | C2/H5 wording only | Owner, via Sir Faizan |
| Modal $60 usage limit | Budget enforcement | **Done 15 Nov** (set in the Modal dashboard) |
