"""Select thresholds on training folds using exact counts on a fixed grid."""

import operator

import numpy as np


def fold_metrics(scores, same, grid, n_folds):
    n_folds = operator.index(n_folds)
    scores = np.asarray(scores)
    same = np.asarray(same, dtype=bool)
    grid = np.asarray(grid)
    if scores.ndim != 1 or same.shape != scores.shape or not 2 <= n_folds <= len(scores):
        raise ValueError("Need equally sized score and label vectors and nonempty folds")
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or np.any(grid[1:] < grid[:-1]):
        raise ValueError("Threshold grid must be finite, nonempty and nondecreasing")
    # Each score passes exactly the first 'passed' thresholds, including ties.
    # Use the same promotion as comparing a score array to one grid scalar.
    dtype = np.result_type(scores, grid[0])
    passed = np.searchsorted(grid.astype(dtype), scores.astype(dtype), side="right")
    passed[np.isnan(scores)] = 0
    counts = []
    width, extra = divmod(len(scores), n_folds)
    boundaries = [i * width + min(i, extra) for i in range(n_folds + 1)]
    folds = [slice(start, end) for start, end in zip(boundaries, boundaries[1:])]
    for indices in folds:
        labels = same[indices]
        bins = passed[indices]
        positives = np.bincount(bins[labels], minlength=len(grid) + 1)
        negatives = np.bincount(bins[~labels], minlength=len(grid) + 1)
        # At threshold j: positives in bins > j and negatives in bins <= j.
        counts.append(labels.sum() + np.cumsum(negatives - positives)[:-1])

    total = np.sum(counts, axis=0)
    accuracies, thresholds = [], []
    for indices, held_out in zip(folds, counts):
        training = total - held_out
        best = np.argmax(training)  # first maximum, as in the original loop
        threshold = grid[best] if training[best] > 0 else 0.0
        thresholds.append(threshold)
        if training[best] > 0:
            accuracies.append(held_out[best] / (indices.stop - indices.start))
        else:
            accuracies.append(((scores[indices] >= threshold) == same[indices]).mean())
    return accuracies, thresholds
