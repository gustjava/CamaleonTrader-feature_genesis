"""
CPCV Utilities: PurgedKFold and Combinatorial Purged Cross-Validation helpers.

Implements utilities to build time-aware folds that avoid leakage via purging and optional
embargo between train/test splits. Intended for Stage 4 validation. Not yet integrated into
the main pipeline execution path.
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional
import numpy as np


@dataclass
class PurgedKFold:
    """Time-aware KFold with purging and optional embargo.

    Parameters:
        n_splits: number of folds
        purge: number of samples to purge around test region from train
        embargo: number of samples to embargo after test region
    """

    n_splits: int = 5
    purge: int = 0
    embargo: int = 0

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        test_indices: List[np.ndarray] = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            test_indices.append(test_idx)
            current = stop

        for test_idx in test_indices:
            # train: exclude test, purge around, and embargo after
            train_mask = np.ones(n_samples, dtype=bool)
            # purge region
            purge_start = max(0, test_idx[0] - self.purge)
            purge_stop = min(n_samples, test_idx[-1] + 1 + self.purge)
            train_mask[purge_start:purge_stop] = False
            # embargo region
            if self.embargo > 0:
                emb_start = min(n_samples, test_idx[-1] + 1)
                emb_stop = min(n_samples, emb_start + self.embargo)
                train_mask[emb_start:emb_stop] = False
            # ensure test indices are excluded from train
            train_mask[test_idx] = False
            train_idx = indices[train_mask]
            yield train_idx, test_idx


def combinatorial_purged_cv(groups: List[np.ndarray], k_leave_out: int = 1, purge: int = 0, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate Combinatorial Purged CV splits over pre-defined time groups.

    Args:
        groups: list of index arrays representing contiguous time groups
        k_leave_out: number of groups to leave out for test in each combination
        purge: purge size applied around test region
        embargo: embargo size applied after test region

    Yields:
        (train_idx, test_idx) tuples of indices
    """
    from itertools import combinations

    n = len(groups)
    all_idx = np.concatenate(groups)
    for combo in combinations(range(n), k_leave_out):
        test_parts = [groups[i] for i in combo]
        test_idx = np.concatenate(test_parts)
        n_samples = len(all_idx)
        train_mask = np.ones(n_samples, dtype=bool)
        # mark test
        # find global positions of test indices within all_idx
        test_pos = np.isin(all_idx, test_idx)

        # purge
        if purge > 0:
            test_positions = np.where(test_pos)[0]
            if test_positions.size > 0:
                start = max(0, test_positions[0] - purge)
                stop = min(n_samples, test_positions[-1] + 1 + purge)
                train_mask[start:stop] = False
        # embargo
        if embargo > 0 and test_pos.any():
            last = np.where(test_pos)[0][-1]
            emb_start = min(n_samples, last + 1)
            emb_stop = min(n_samples, emb_start + embargo)
            train_mask[emb_start:emb_stop] = False

        # exclude test
        train_mask[test_pos] = False
        train_idx = all_idx[train_mask]
        yield train_idx, test_idx

