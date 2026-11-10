# Project context for Claude

Research project: paired axon/myelin instance segmentation in TEM (SIPAIM 2026 challenge used as the problem statement).

- **Read `docs/notes/00-index.md` first.** `01-challenge-spec.md` is verified from the site source and beats
  `03-literature-review.md` (secondary) where they disagree.
- Record new facts and answers in `docs/notes/08-open-questions-and-risks.md` and progress in the `09-research-plan.md` log.
- Never commit data, masks, predictions or weights. The repo is public.
- Splits are by image (by animal if metadata allows), never by tile. Evaluate on full stitched images.
- Every baseline uses the same pairing post-processing and our official-metric replica.
- Pixel size is 2.152 nm/px. Masks are uint16: 0 bg, 1 artifact (ignored), even X axon, X+1 myelin.
