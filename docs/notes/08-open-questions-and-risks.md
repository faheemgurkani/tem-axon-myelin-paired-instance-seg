# 08 — Open Questions and Risks

Update this list as questions get answered. Record the answer, its source and the date.

## Questions for the organizers (gabriel.jimenez@icm-institute.org)

- [ ] Can unregistered researchers use the data after the deadline, and under what licence? May results be published?
- [ ] Will the test GT be released after SIPAIM, or will a post-challenge evaluation server be offered?
- [ ] Is there official evaluation code? If not, settle: per-image vs pooled Dice; global vs per-instance HD95; IoU threshold; empty-mask HD95; border-cut fibers.
- [ ] Animal / block / condition metadata per image (for grouped splits)? Control vs disease models?
- [ ] Data provenance: any overlap with ASTIH TEM1 (DANDI 001436) or other public sets?
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
