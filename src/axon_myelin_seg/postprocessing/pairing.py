"""Semantic (axon, myelin, boundary) probabilities → paired (X, X+1) instances.

Shared by every baseline: marker-controlled watershed seeded by axon components, myelin flooded within a distance
cap, rings without a seed and axons without a ring removed. See docs/notes/04-baselines.md.
"""


def semantic_to_paired_instances(axon_prob, myelin_prob, boundary_prob=None, **kwargs):
    raise NotImplementedError
