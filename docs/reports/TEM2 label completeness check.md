# Are ASTIH TEM2's labels near-complete? (readiness gap C4, 15 Nov 2026)

**Why it matters:** H6 (Phase 2) predicts a *low* selection share on TEM2 and a *high* one on TEM1. That prediction rests on the assumption that TEM2's labels are near-complete and TEM1's are selective. If the assumption is wrong, the TEM1-vs-TEM2 contrast can't be used as support for C1.

**Sources:** the ASTIH paper (Collin, Boudreau, Cohen-Adad, MIDL 2026), local full text `docs/references/papers/astih_collin26a.txt` (§2.1.2; App. A.1–A.2; split text around line 265), and our earlier notes on Iram et al., *PNAS* 2024 (`docs/research_notes/Similar datasets 2023 2026 citation analysis/02_tem2_axoncallosumem_shank3b.md`).

## Verdict: plausible, not established

**What supports it:**
- TEM2's myelinated-axon labels started from AxonDeepSeg pseudo-labels that were "manually corrected" (A.2). A model proposes a fiber wherever it sees one, so this route tends towards exhaustive labels. TEM1 instead started from a threshold + connected-component macro that kept only full myelin rings (A.1).
- For every ASTIH dataset, stage 1 of curation corrected masks "to ensure accurate myelin thickness for all axons" (§2.1.2).
- TEM2 is labelled as a whole scene: unmyelinated axons, OL nuclei and OL processes are separate classes (26,629 axons, 4,372 of them myelinated). That points to an intent to label everything visible.

**What is missing:**
- **No completeness audit is reported.** Neither paper counts unlabelled myelinated fibers or states an inclusion rule for abnormal profiles.
- **The correction emphasis was separation**, not completeness: "with an emphasis on separation of adjacent fibers by a minimum 2-pixel boundary" (A.2). The stage-2 QC script checks topology (one axon per myelin region), not omissions.

**Three confounds**, each of which could produce TEM2's higher fiber F1 (0.949 vs 0.791) without any difference in label completeness:
1. **Phenotype.** TEM2 is SRF-cKO hypomyelination. There are fewer myelinated axons, and myelin *thickness is unchanged*, so there are few thin, split or degenerating profiles of the kind our protocol leaves out. TEM1 contains cuprizone demyelination. Fewer abnormal fibers means fewer selection decisions, however careful the labeller.
2. **Model-seeded ground truth.** The labels began as AxonDeepSeg predictions. A U-Net scored against corrected U-Net output is likely to agree with it on what counts as a fiber, which inflates precision. This is the bias Bukas et al. (MICCAI-W 2026) warn about.
3. **Split.** The TEM2 test set takes images from every mouse (the split text around line 265), while TEM1 is split by subject. TEM2's score is therefore an in-animal number and is expected to be higher.

Pixel size also differs (4.93 vs 2.36 nm/px).

## What this changes

- **Don't use "TEM2 is near-complete" as a premise.** Before H6 is pre-registered, *measure* completeness on both datasets with the same CPU-only audit:
  1. run our ring-candidate miner (`taxonomy.ring_candidates`) on GT background and report unlabelled candidates per 100 µm²;
  2. check ~50 random candidates per dataset by eye (real fiber / abnormal / not a fiber).

  H6's prediction is then conditional on the measured completeness, not on the paper's description.
- **Report the three confounds as limitations** of any TEM1-vs-TEM2 comparison. The phenotype confound can be partly separated by also scoring TEM1's control animals only.
- TEM2 stays the negative control for C1, but its weight in the argument now depends on the audit.
