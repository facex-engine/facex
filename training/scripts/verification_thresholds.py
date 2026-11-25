"""Select thresholds on training folds using exact counts on a fixed grid."""

import numpy as np


def fold_metrics(scores, same, grid, n_folds):
    scores = np.asarray(scores)
    same = np.asarray(same, dtype=bool)
    # Each score passes exactly the first 'passed' thresholds, including ties.
    # Use the same promotion as comparing a score array to one grid scalar.
    dtype = np.result_type(scores, grid[0])
    passed = np.searchsorted(grid.astype(dtype), scores.astype(dtype), side="right")
    passed[np.isnan(scores)] = 0
    counts = []
    folds = np.array_split(np.arange(len(scores)), n_folds)
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
            accuracies.append(held_out[best] / len(indices))
        else:
            accuracies.append(((scores[indices] >= threshold) == same[indices]).mean())
    return accuracies, thresholds
