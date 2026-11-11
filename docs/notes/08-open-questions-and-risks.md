# 08 — Open Questions and Risks

Update this list as questions get answered. Record the answer, its source and the date.

## Questions for the organizers (gabriel.jimenez@icm-institute.org)

- [x] Data obtained 2026-11-12 (licence still unstated). Can unregistered researchers use the data after the deadline, and under what licence? May results be published?
- [ ] Will the test GT be released after SIPAIM, or will a post-challenge evaluation server be offered?
- [ ] Is there official evaluation code? If not, settle: per-image vs pooled Dice; global vs per-instance HD95; IoU threshold; empty-mask HD95; border-cut fibers.
- [ ] Animal / block / condition metadata per image (for grouped splits)? Control vs disease models?
- [ ] Data provenance: animal count, conditions, source papers? (Our check on 2026-11-11 found overlap with ASTIH TEM1 unlikely; see 07.)
- [ ] Annotation convention: is the inner tongue / periaxonal space axon or myelin? How are unmyelinated axons and the scale bar labelled?
- [ ] Is transductive use of the unannotated test images allowed?
- [ ] Will there be a challenge summary paper that includes participants?

## Risks

| Risk | Mitigation |
|---|---|
| Our metric replica diverges from the official one | Config switches + sensitivity table; ask for reference code |
| Leaky validation (same animal in train and val) | Grouped splits once metadata is known; otherwise split by image and state the limitation |
| Single-lab data, so "generalization" claims are weak | External sets (ASTIH, AimSeg); leave-dataset-out evaluation |
| Data licence blocks publication | Ask early; keep the method demonstrable on CC-BY ASTIH data |
| The baseline phase consumes the project | Time-box; at most 5 baselines; shared pipeline |
| A strawman baseline is exposed by reviewers | nnU-Net self-configures; report compute and tuning budgets |
| g-MAE rewards dropping hard fibers | Always report it with fiber recall |

## Resolved

- 2026-11-10: mask format = uint16 paired IDs (site `data.md`/`submission.md`); `{0,1,2}` checklist is stale.
- 2026-11-10: dataset = 266 annotated + 134 test, 4096×3008.
- 2026-11-10: leaderboard podium is hard-coded placeholder HTML.
- 2026-11-11: overlap with ASTIH TEM1 unlikely (different lab, microscope, pixel size and image size), but unconfirmed by the organizers. Source: `docs/research_notes/.../datasets_provenance.md`.
- 2026-11-11: AxonDeepSeg's g-ratio definition (equivalent-circle diameters) matches the challenge's definition (`domain_methods.md`).
- 2026-11-11: ASTIH evaluates fibers with `stardist.matching` at IoU 0.3/0.5. TEM1 nnU-Net: Dice 0.974/0.924, but fiber F1@0.5 is only 0.791, so **pairing/instance separation is the bottleneck** (`domain_methods.md`).

## New risks (2026-11-11)

| Risk | Mitigation |
|---|---|
| Inner-tongue / periaxonal convention differs between the challenge, AxonPath and ASTIH | Inspect the challenge masks; map each external dataset explicitly |
| HD95 differs across libraries (MONAI max of directed P95 vs panoptica pooled vs DeepMind surfel-weighted) | Config switch; report which one we use (`evaluation_methodology.md`) |
| On oval fibers, g-ratio definitions (equivalent circle vs minimum Feret) differ by 0.02–0.07, more than a 1-px boundary error | Use the challenge definition; report the others as sensitivity |


## Data risks found by EDA (2026-11-12, see [11](11-deep-eda.md))

| Risk | Mitigation |
|---|---|
| Overlapping tiles (40 images in 18 groups) leak across splits | GroupKFold on `configs/splits/overlap_groups.json` |
| ~4–8% of real fibers unlabelled; test GT has the same misses | Down-weight or ignore confident unlabelled rings in training; FP taxonomy in diagnostics |
| Interstitial gaps between touching fibers look like fibers | Track FP rate on gaps; motivation for by-construction pairing |
| Annotation-noise ceiling (myelin Dice ~0.93–0.96, g floor ~0.006) | Bootstrap CIs; don't over-interpret small gains |
| Test tiles probably overlap train tiles | Do not exploit; report if confirmed |
