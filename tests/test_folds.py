import numpy as np

from ivcm.data import contiguous_folds


def test_contiguous_folds_partition():
    folds = contiguous_folds(258, 5)
    # First 258 % 5 = 3 folds get the extra slice.
    assert [len(f) for f in folds] == [52, 52, 52, 51, 51]
    # Concatenation is the ordered, complete, disjoint cover 0..257 — which also confirms
    # each block is contiguous (array_split keeps order).
    assert np.concatenate(folds).tolist() == list(range(258))


def test_contiguous_folds_deterministic():
    a = [f.tolist() for f in contiguous_folds(258, 5)]
    b = [f.tolist() for f in contiguous_folds(258, 5)]
    assert a == b
