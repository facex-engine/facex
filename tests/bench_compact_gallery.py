"""Single-query latency, including selection and final cosine scoring."""
import os
from statistics import median
import time

import numpy as np

from facex import CompactGallery, Gallery


def main():
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    rng = np.random.default_rng(811)
    for n in (4096, 32768, 65536):
        data = rng.standard_normal((n, 512), dtype=np.float32)
        galleries = (Gallery(data), CompactGallery(data))
        del data
        timings = [[], []]
        for i, query in enumerate(rng.standard_normal((41, 512), dtype=np.float32)):
            results = [None, None]
            for j in (i % 2, 1 - i % 2):
                start = time.thread_time_ns()
                results[j] = galleries[j].search(query[None, :], 5)
                elapsed = time.thread_time_ns() - start
                if i:
                    timings[j].append(elapsed / 1e6)
            for a, b in zip(*results):
                np.testing.assert_array_equal(a, b)
        print(n, "median CPU ms", *(round(median(t), 3) for t in timings), flush=True)


if __name__ == "__main__":
    main()
