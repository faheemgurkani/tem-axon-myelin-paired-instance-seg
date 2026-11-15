# Overlooked contribution gaps: validation of two assessments (15 Nov 2026)

**Question (owner):** validate two answers to "is there a contribution gap we missed?", using the same search criteria as before, and finalise.
- **Assessment A:** a short answer given in chat, with three points: protocol bias on biology, a "selection head" for C3, and SIPAIM winners as C1 test subjects.
- **Assessment B:** `docs/research_notes/Phase 0b evidence/11_overlooked_gaps.md`, with four points: B1 recall by fiber type on outside data, protocol bias on g, abnormal fibers as a class, and minor framing items.

**Criteria:** top venues, 2025 to 13 Nov 2026, preprints marked; older work cited only where it is the origin of an idea. Downloads and CAPTCHAs: wait ≥ 15 s, never solve.

**Local check:** every report, the Phase 0b notes 01–11, `05-results.md` §4–5, `07-plan.md`, the self-study notes, `results/phase0b/`, `data/processed/eda/` tables, and `data/external/` inventories.

Sources and verbatim evidence: `docs/research_notes/Overlooked gaps validation/00_sources.md`.

## Verdict summary

| Point | Source | Verdict | What changes |
|---|---|---|---|
| Protocol bias on mean g and counts | A1 = B2 | **Holds, novel in this domain; premise corrected** | The *direction* of the g shift is unknown: our skipped fibers are mixed (thick "onion" myelin = low g; thin myelin = high g). The *differential* claim (bias grows with disease) can't be tested on our data without per-image condition metadata (A2). Cite stereology and nerve-morphometry bias literature: the principle is old; the measurement for DL annotation protocols is new |
| B1 recall by fiber type on near-complete outside sets | B1 | **Holds; strongest new task; needs a reference arm** | It is a hidden-stratification analysis (Oakden-Rayner 2020); cite it. The cause (annotation protocol) and the EM setting are new. Without a model trained on *complete* labels as a reference, "B1 misses thin fibers" can't be blamed on the protocol (thin fibers are harder anyway, and there's domain shift). It tests **thin-myelin** fibers only: the outside sets also exclude degenerating or broken-ring fibers |
| Abnormal fibers as a class / user picks the protocol | A2 = B3 | **Partly pre-empted; downgrade to a design option** | Style-conditioned decoders already exist (Zepf et al., ICLR 2023; Parikh et al. 2026, arXiv). MicroStructFormer (bioRxiv May 2026, preprint) already segments an EM "abnormality" class and reports per-fiber inclusive and exclusive measures. What remains new is narrow: a whole-fiber, protocol-excluded class in paired instance output, with the official map unchanged |
| SIPAIM winners as C1 test subjects | A3 | **Not feasible as stated; drop** | Test GT is unannotated, and the site promises no release of GT, predictions or code. Winners are not announced. Running their code on our dev images would leak: they trained on all 266. Keep only "check the SIPAIM entries before claiming 'first on this data'" |
| Recall-aware g metric | B4 | **Plausible minor contribution; not searched in depth** | Keep inside C1's gaming curve, not as a headline |
| Human ceiling from overlapping tiles | B4 | **No prior method found; reword** | We don't know whether both tiles were labelled by the same annotator (ICM.Quant and PUCP both annotated). Call it *repeat-annotation agreement*, not inter-rater, unless the organisers confirm. Only 129 fibers |

**Overall:** assessment B is the more accurate of the two. Assessment A's point 3 fails, and its point 1 rests on a premise the local data does not support. Both missed the prior art for the "choose a protocol" idea.

## 1. Protocol bias on the biological endpoint

**Novelty (searched):**
- No 2025–26 paper quantifies how an annotation *inclusion protocol* shifts g-ratio, fiber counts or any morphometric endpoint.
- **Closest prior:**
  - Bukas et al. (MICCAI-W 2026): model-assisted labels bias downstream analysis, through mixing, not inclusion;
  - Parikh et al. (arXiv 2605.06891, 2026; the NeurIPS label is unconfirmed on the paper itself): boundary bias only, simulated, fairness framing, no omission and no morphometry;
  - a cell-segmentation study noting annotators skipped dying, abnormal-looking cells (bioRxiv 2022 / PMC9981753; qualitative, outside the window);
  - Komura et al. (*Patterns* 2023): pathologists miss atypical cells.
- **The underlying principle is old:** inclusion and exclusion rules for fiber profiles are a named level of bias in nerve morphometry, and design-based stereology exists to avoid it. Fields & Dutta 2025 already call EM g sampling "inadequate and inappropriate". **Cite these. Claim the measurement across four DL annotation protocols, not the idea.**

**Premise correction (local data):**
- **Assessment A said the skipped fibers are mostly thin myelin (high g).** The evidence is mixed and small:
  - the completeness audit (48 candidates, one non-expert rater): ~60% abnormal, including very thick or redundant "onion" myelin (**low g**), large axons with thin myelin (**high g**), and oblique profiles;
  - B1's `other_bg` gallery (16 items): real unlabelled fibers are "mostly thin myelin".
- **No unlabelled candidate has a measured g.** `data/processed/eda/unl_cands.csv` holds only the inner diameter.
- The detector misses thin-myelin fibers, so the candidate sample is itself biased toward closed, thicker rings.
- **So the rule must be two-sided:** |Δ mean g| ≥ 0.03. Measuring it needs outer ring outlines for a sample of candidates, which an expert or a semi-automatic tracing can provide.

**Differential bias:**
- "Bias grows with pathology" needs per-image condition labels. We have none for our data (A2, not yet asked). TEM1 has only a 3-cluster proxy.
- Until A2 is answered, the claim is "the protocol shifts mean g by X", not "it masks a disease effect".

## 2. B1 recall by fiber type on near-complete outside data

**Novelty:**
- **Hidden stratification** (Oakden-Rayner et al. 2020; Bissoto et al., MICCAI 2025) is the established name for "good on average, fails on a clinically important subset". Cite it.
- No paper found evaluates a model trained on *selectively* labelled data against a fully labelled external set, broken down by subtype. A 2025 imaging-mass-cytometry study simulated random omission only. **The framing is new.**

**Corrections:**
1. **A reference arm is required.** On the same images, score a model trained on that dataset's own complete labels (ASTIH's TEM1 model on `sub-nyuMouse26`; its TEM2 model on the TEM2 test set), binned by the same fiber types. Then compare the difference in recall (thin minus normal) between B1 and the reference. Without it, intrinsic difficulty and domain shift explain any gap.
2. **Which fiber types it can test:**
   - TEM1 labels thin-myelin fibers: its remyelinating-like cluster has 19% of fibers with g > 0.85.
   - AxonPath labels thin remyelinating rings.
   - Both **exclude degenerating or broken rings** (TEM1 "full rings"; the AimSeg protocol). Assessment B's wording "they label the abnormal fibers our dataset leaves out" is true only for thin myelin.
3. **The outcome is genuinely open.** On our data, B1 *already predicts* many unlabelled thin-myelin fibers (about half of its 200 `other_bg` FPs look real; `05-results.md` §5). It has not fully learned to skip them. That's good: the test can fail.
4. **Usable size:**
   - TEM1 (158 images, 20 mice) carries it.
   - TEM2 DANDI has 10 annotated images at 4.93 nm, which needs 2.3× rescaling.
   - TEM2 test ROIs are 600×600 with 64% border-cut fibers: weak.
   - AxonPath has 20 ROIs.
   - Bootstrap by mouse.
5. **C3's outside test (assessment B) is accepted, with a narrower scope:** recovery of thin-myelin fibers and "does no harm", not the deliberately excluded kinds.

**Cost:** inference only. B1's checkpoints are on the Modal Volume (< $2 on Modal, or MPS locally after asking).

## 3. Abnormal fibers as a class, or a protocol the user picks

**Prior art found:**
- **Zepf et al., ICLR 2023**, "That Label's Got Style": segmentation conditioned on a discrete label-style variable, chosen at inference.
- **Parikh et al. 2026** (arXiv; the venue is unconfirmed on the paper): a subgroup-conditioned decoder that outputs masks in a chosen *reference annotation style*. Styles are limited to those present in training, and the bias is boundary-only.
- **MicroStructFormer** (Deng … Khadra, bioRxiv May 2026, preprint; read in full 15 Nov):
  - Mask2Former + AxonDeepSeg on Mbp-mutant spinal cord;
  - an **"abnormality" class** (uncompacted membrane), assigned per fiber by geodesic growth;
  - "exclusive" and "inclusive" fiber measures;
  - reduced sensitivity for the abnormality class in the most severe genotypes;
  - labels were made by **iterative model-assisted refinement** (the Bukas risk).
- The optic-nerve paper (*Ophthalmol Sci* 2026) excludes degenerated profiles rather than classifying them.

**What remains new:** a **whole-fiber** class for fibers the protocol excludes, inside a *paired* instance output, with the official map left unchanged.
- This is narrow and data-limited: a few hundred expert ratings make a pilot, not a training set.
- **Verdict:** a C3 design option for Gate 1, framed as *applying* style or protocol conditioning, never as a new mechanism.

## 4. SIPAIM winners as C1 test subjects (assessment A point 3): rejected

The challenge site was read on 15 Nov; the subpages return 404.
- **Required deliverables:** test predictions (PNG, labels 0/1/2), a 2–4-page method PDF and code. Weights are not required.
- **What the site does not promise:** release of test GT, an evaluation server, or publication of code, predictions or papers.
- **Winners:** "will be announced at SIPAIM 2026". The page also lists three named entries with Dice values while saying submissions are "under independent validation"; treat these as unverified.
- **Leakage:** any released code was trained on all 266 train images, which include our dev fold. Scoring it on our dev split would leak; retraining on our split costs GPU.
- **What survives:** recheck the SIPAIM entries after 18–20 Nov for novelty, and score their predictions only if GT ever appears.
- A minor discrepancy: the site banner says "40 new unlabeled images"; our count of the test set is 134.

## 5. New angles found during validation

1. **Our selection estimates are a lower bound for a second reason.** Parikh et al. note that disagreement-based audits *underestimate* bias once a model has learned the biased labels. B1 was trained on selective labels, so its `unlabelled_real` share understates selection. This strengthens C1's direction, but must be stated.
2. **The labels of TEM2 and MicroStructFormer are model-seeded.** Bukas' bias applies to H6c's interpretation (already noted for TEM2 in C4). Report it with H6c.
3. **The differential-bias claim depends on A2** (per-image condition). This is now a second reason, beyond the licence, to send the organiser questions.

## Recommendation (for the owner's decision; no IDs assigned)

1. **Adopt B1 as a Phase 2 task, with the reference arm** (a model trained on complete labels, on the same images), and pre-register a two-sided rule on the difference in recall between thin and normal fibers.
2. **Adopt protocol bias on g as a named T-16′ analysis**, with a **two-sided** rule (|Δ mean g| ≥ 0.03). First trace outer rings for a candidate sample, through the expert packet or semi-automatically. Word it as "shifts mean g", not "masks disease", until A2 is answered.
3. **Keep the abnormal-fiber class as a Gate 1 option only**, citing Zepf 2023, Parikh 2026 and MicroStructFormer.
4. **Drop the SIPAIM-winners test.** Keep the novelty recheck after 20 Nov.
5. **Reword** the overlap ceiling as repeat-annotation agreement, and add the lower-bound note (§5.1) to C1.
