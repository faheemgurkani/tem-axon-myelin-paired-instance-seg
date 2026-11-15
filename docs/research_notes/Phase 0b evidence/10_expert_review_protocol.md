# Blind expert review: T-20 + validation of the completeness audit (D-33, fixed 15 Nov 2026, before any rating)

**Why:**
- The "78% of B1's errors are selection" claim (C1) and the completeness ranking behind H6 both rest on judgements that so far come from **one non-expert rater** (Claude).
- A domain expert must confirm them, as `PLAN.md` §1.7 and readiness gap B1 require.
- The two checks are combined into one blind session so the rater can't tell which question an item belongs to.

## Packet
- **Built by** `scripts/run/expert_review_packet.py` (seed 0, ~4 min on CPU).
- **Output:** `data/processed/expert_review/`. Git-ignored: it contains challenge-image crops, and the data licence is unstated. **Never upload it anywhere public.** Share it with the expert only as a local folder (zip by hand).

| Part | Items | What the yellow outline shows |
|---|---|---|
| **T-20** | **all 61** B1 dev-val predictions classed `unlabelled_real` + **40 random** of 200 `other_bg` | B1's predicted fiber |
| **Audit, ours** | 48 | an unlabelled ring candidate |
| **Audit, TEM1** | 48 | " |
| **Audit, TEM2 (DANDI)** | 48 | " |
| **Audit, AxonPath** | 15 (all) | " |
| **Total** | **260**, shuffled | |

- **Presentation:** every crop is the same physical field of view (1.8 µm) at 256 px, the raw image plus one yellow outline. There is no GT tint and no dataset or class name.
- **Blinding limit:** image style (contrast, white padding at image edges, TEM2's coarser pixels) can hint at the source dataset. Report this as a limitation.
- **Rating page:** `index.html`, opened locally in any browser.
  - Five categories, keyboard 1–5, optional comment.
  - Answers autosave in the browser; **Export CSV** writes `fiber_review_<rater>.csv`.
- **Categories:**
  1. real myelinated fiber, normal-looking;
  2. real myelinated fiber, abnormal (thin, redundant/onion, degenerating, oblique);
  3. not a fiber (gap between fibers, organelle, debris, artefact);
  4. unmyelinated axon;
  5. can't tell.
- **Answer key:** `key.csv` (item → part, dataset, class, image, position). **The rater must not open it.**

## Raters
- **Rater A:** a domain expert in myelin EM (required). Ask Sir Faizan for a contact.
- **Rater B:** the owner (second rater, for agreement).
- Both rate independently, without discussing.
- Claude's earlier gallery counts were aggregate, not per item. They are kept as the provisional estimate and are **not** used as a rater.

## Pre-set decision rules ("real" = categories 1 + 2; "can't tell" excluded from denominators and reported)

| Readout | Rule | Consequence |
|---|---|---|
| **T-20, primary (gap B1, Gate 1)** | Rater A judges **≥ 70%** of the 61 `unlabelled_real` items real | Holds → the selection claim of C1 stands. Fails → C1's "78% selection" is rewritten before Gate 1 |
| Agreement | Cohen's κ (real vs not real, A vs B) on all 260 items. **κ ≥ 0.6** acceptable | κ < 0.6 → a third rater or adjudication of disagreements before using the result |
| `other_bg` | Report Rater A's real share (Claude's 13 Nov gallery: about half) | Sets how far O-sel-broad over-counts; descriptive |
| **Audit, ours** | Real share **≥ 70%** (provisional 83%) | Fails → our "selective dataset" estimate (≈ 4.2%) is revised downward |
| **Audit, external** | Real share **≤ 25%** in each of TEM1, TEM2, AxonPath (provisional 8%, 19%, 13%) | Fails for a dataset → its completeness is re-examined **before** H6 is run on it (H6 premise) |
| Abnormal share | Category 2 / (1 + 2) for ours, T-20 and audit separately (provisional ~60%) | Descriptive; feeds C3's two-kinds-of-unlabelled caveat |

These thresholds were written on 15 Nov, before any expert rating exists. Change them only with a logged reason, and only before the first rating CSV is opened.

## Scoring (to do when ratings arrive)
Join `fiber_review_*.csv` to `key.csv` by item, then compute the table above with 95% bootstrap CIs over items. Record the numbers in `docs/notes/05-results.md` and the outcomes in the `07-plan.md` log.
