import numpy as np

from axon_myelin_seg.data.encoding import axon_mask, encode_pairs, fiber_ids, myelin_mask, unpaired_ids


def test_encode_and_decode_pairs():
    axons = np.array([[1, 0], [0, 2]])
    myelin = np.array([[0, 1], [2, 0]])
    labels = encode_pairs(axons, myelin)
    assert labels.dtype == np.uint16
    assert labels.tolist() == [[2, 3], [5, 4]]
    assert fiber_ids(labels).tolist() == [2, 4]
    assert axon_mask(labels).sum() == 2 and myelin_mask(labels).sum() == 2


def test_unpaired_and_artifact_ignored():
    labels = np.array([[1, 2], [7, 0]], dtype=np.uint16)
    assert unpaired_ids(labels) == {"axon_without_myelin": [2], "myelin_without_axon": [7]}
    assert not axon_mask(labels)[0, 0] and not myelin_mask(labels)[0, 0]
