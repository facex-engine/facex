"""Compare a Cauchy-Schwarz prefix filter with the regular exact gallery on synthetic data."""
import os
from statistics import median
import time

import numpy as np
from facex import Gallery


def main():
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    rng = np.random.default_rng(154)
    rows = rng.normal(size=(32768, 512))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    gallery = Gallery(rows)
    rows = gallery._rows
    queries = rng.normal(size=(16, 512)).astype("f").astype("d")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    for prefix in (128, 256, 384, 480):
        head = rows[:, :prefix].copy()
        norms = np.linalg.norm(rows[:, prefix:], axis=1)
        times = [[], []]
        counts = []
        for i, query in enumerate(queries):
            oracle = gallery.search(query[None], 5)[0][0]
            x = query.astype("f").astype("d")
            x /= np.linalg.norm(x)
            for path in (i % 2, 1-i % 2):
                start = time.thread_time_ns()
                if path:
                    partial = head @ x[:prefix]
                    radius = norms * np.linalg.norm(x[prefix:])
                    lower = np.partition(partial-radius, -5)[-5]
                    indices = np.flatnonzero(partial+radius >= lower-1e-11)
                    scores = rows[indices] @ x
                    selected = indices[np.argpartition(scores, -5)[-5:]]
                    counts.append(len(indices))
                else:
                    gallery.search(query[None], 5)
                times[path].append((time.thread_time_ns()-start)/1e6)
            assert set(selected) == set(oracle)
        print(prefix, median(counts), *(round(median(v),2) for v in times))


if __name__ == "__main__":
    main()
