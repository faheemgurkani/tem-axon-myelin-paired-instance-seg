# 06 — Research Directions (our methodology, post-baselines)

> **Updated recommendation (2026-11-11, report §"The publishable gap"):** an **instance-first, part-second nested decoder**
> (filled-fiber foreground + fiber separation on filled fibers + within-fiber inner/outer part head, written as 2k/2k+1), plus an optional **dual-radius polar head**
> (novel: no prior dual-radius StarDist exists; HydraStarDist, arXiv 2504.12078, is the nearest prior work). Losses: differentiable g-ratio loss, TopoInteraction containment,
> and boundary loss on both contours. Augmentation and self-distillation go in every arm (TopoMortar). Hypotheses H1–H4. There is also a standalone MIDL-style evaluation paper
> (ambiguity sensitivity + recall-vs-g-MAE gaming curve).

The method must come from **error analysis** of the baselines ([02](02-problem-framing.md) §critique). These are
candidate directions, ranked by how directly they address the organizers' two stated gaps.

## D1. Nested-instance representation (core contribution)

Treat each fiber as **one object with two coupled contours**, so pairing and containment hold *by construction*:
- a Cellpose-style flow to the fiber center + signed distances to the inner and outer contours, **or**
- polar / star-convex **inner and outer radii** per ray (dual-radius StarDist), **or**
- a two-level distance field (fiber-instance distance + inner/outer level sets).

g-ratio then becomes a **differentiable** function of the predicted radii or areas, which allows a **g-ratio consistency
loss** alongside a boundary/HD95 surrogate. The prior work closest to this, the optic-nerve dual decoder and AimSeg's hierarchy, is not end-to-end instance-level.

## D2. Containment-topology regularization

- TopoInteraction-style constraints (Gupta et al., ECCV 2022):
  - axon ⊂ myelin
  - an axon pixel never touches background
  - myelin–myelin exclusion between instances
- Betti matching or Skeleton Recall on rings (β1 = 1), applied on fiber crops for cost.
- A new **pairing-violation metric** as the evaluation counterpart.

## D3. Staining/preparation-robust pretraining

MAE/DINO-style SSL on the challenge images + CEM500K + unlabeled ASTIH, then fine-tune. Claim robustness via
**leave-dataset-out** evaluation (challenge → TEM1/TEM2/AimSeg).

## D4. Label efficiency

Mean-teacher / pseudo-labels on the 134 unannotated test images (transductive; check the rules if submitting) and on external
unlabeled EM. Uncertainty-driven active learning.

## Venues

- MICCAI 2027 (main) or ISBI 2027 for the method.
- MIDL for a benchmark/metric paper (pairing-aware evaluation; a fair baseline comparison is itself a contribution).
- Medical Image Analysis / IEEE TMI for the journal extension with multi-dataset generalization.
- Scientific Data / NeuroImage if we release annotations.

## Paper skeleton

The 2–4 page SIPAIM-format method description is the seed. A full paper needs:
1. a topology-aware contribution (D1/D2)
2. generalization evidence on external datasets
3. the fair baseline benchmark
4. released code and weights
