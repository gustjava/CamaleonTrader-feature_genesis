import numpy as np

from utils.cpcv import PurgedKFold, combinatorial_purged_cv


def test_purged_kfold_basic_properties():
    X = np.arange(30)
    cv = PurgedKFold(n_splits=5, purge=1, embargo=2)
    splits = list(cv.split(X))
    assert len(splits) == 5

    n = len(X)
    for train_idx, test_idx in splits:
        # disjoint
        assert set(train_idx).isdisjoint(set(test_idx))
        # purge: there should be a gap around test region
        if len(test_idx) > 0:
            t0, t1 = test_idx[0], test_idx[-1]
            forbidden = set(range(max(0, t0 - 1), min(n, t1 + 1 + 1)))  # purge=1
            assert not any(i in forbidden for i in train_idx)


def test_combinatorial_purged_cv_counts():
    # Create 6 contiguous groups of equal size
    groups = [np.arange(i * 10, (i + 1) * 10) for i in range(6)]
    splits = list(combinatorial_purged_cv(groups, k_leave_out=2, purge=1, embargo=1))

    # C(6,2) = 15 combinations
    assert len(splits) == 15

    # For each split, ensure disjointness and that embargo/purge excluded neighbors
    all_idx = np.concatenate(groups)
    n = len(all_idx)
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        # Map test positions in global index order
        test_pos = np.isin(all_idx, test_idx)
        if test_pos.any():
            test_positions = np.where(test_pos)[0]
            start = max(0, test_positions[0] - 1)
            stop = min(n, test_positions[-1] + 1 + 1)
            forbidden = set(all_idx[start:stop])
            assert not any(i in forbidden for i in train_idx)

