"""Regression test for exact threshold selection, including ties on the grid."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/scripts"))
from verification_thresholds import fold_metrics


def exhaustive(scores, same, grid, n_folds):
    accuracies, thresholds = [], []
    for test in np.array_split(np.arange(len(scores)), n_folds):
        train = np.ones(len(scores), dtype=bool)
        train[test] = False
        best, threshold = 0, 0.0
        for t in grid:
            correct = np.count_nonzero((scores[train] >= t) == same[train])
            if correct > best:
                best, threshold = correct, t
        thresholds.append(threshold)
        accuracies.append(np.mean((scores[test] >= threshold) == same[test]))
    return accuracies, thresholds


class TestThresholds(unittest.TestCase):
    def test_reference(self):
        rng = np.random.default_rng(275)
        for grid in (np.linspace(0, 1, 401), np.linspace(-0.5, 1, 601)):
            for dtype in (np.float32, np.float64):
                ties = grid[::5].astype(dtype)
                scores = np.concatenate((ties, np.nextafter(ties, dtype(-np.inf)),
                                         np.nextafter(ties, dtype(np.inf)),
                                         rng.uniform(-1, 1, 79).astype(dtype),
                                         np.array([np.nan, np.inf, -np.inf], dtype=dtype)))
                labels = rng.random(len(scores)) < 0.5
                for values, same in ((scores, labels), (scores, ~labels),
                                     (np.full_like(scores, -1), np.ones_like(labels))):
                    np.testing.assert_array_equal(fold_metrics(values, same, grid, 10),
                                                  exhaustive(values, same, grid, 10))


if __name__ == "__main__":
    unittest.main()
