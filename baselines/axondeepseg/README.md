# Baseline: axondeepseg (deferred)

Not implemented. Deferred by D-23 (time and GPU budget); see `docs/notes/07-plan.md`.

AxonDeepSeg / ASTIH code is used instead in task T-17: reproducing ASTIH's TEM1 numbers with our evaluator. The clones are in `external/ASTIH` and `external/axondeepseg`.
Their centroid-seeded watershed is the pairing rule that our component-seeded one (`postprocessing/pairing.py`) improves on.
