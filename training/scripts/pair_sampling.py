"""Seed-compatible balanced verification pairs from a mapped image index."""
from pathlib import Path

import numpy as np

INDEX_RECORD = np.dtype([("offset", "<u8"), ("size", "<u4"), ("label", "<u4")])


def sample_index_pairs(path, seed, n_per):
    size = Path(path).stat().st_size
    if not size or size % INDEX_RECORD.itemsize:
        raise ValueError("Expected complete image index records")
    labels = np.memmap(path, dtype=INDEX_RECORD, mode="r")["label"]
    # Stable sorting preserves source order inside each class.
    order = np.argsort(labels, kind="stable")
    ordered = labels[order]
    starts = np.r_[0, 1 + np.flatnonzero(ordered[:-1] != ordered[1:])]
    ends = np.r_[starts[1:], len(order)]
    groups = np.flatnonzero(ends - starts >= 2)
    groups = groups[np.argsort(order[starts[groups]])]
    if len(groups) < 2 or n_per < 1:
        raise ValueError("Need two classes with two images each and a positive pair count")
    rng = np.random.default_rng(seed)
    pairs = []
    for g in rng.choice(groups, n_per):
        a, b = rng.choice(order[starts[g]:ends[g]], size=2, replace=False)
        pairs.append((int(a), int(b), True))
    for _ in range(n_per):
        g, h = rng.choice(groups, size=2, replace=False)
        a = rng.choice(order[starts[g]:ends[g]])
        b = rng.choice(order[starts[h]:ends[h]])
        pairs.append((int(a), int(b), False))
    rng.shuffle(pairs)
    return pairs
