"""Research benchmark for bounded integer screening of a warm backing gallery."""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter_ns

import numpy as np

from facex.gallery import CompactGallery, _rank, _select, _unit_rows


class Index:
    def __init__(self, raw, bits, scan, centered=False):
        self.bits, self.scan = bits, scan
        n = len(raw)
        self.codes = np.empty((n, 512 * bits // 8), np.uint8)
        self.scale, self.radius, self.norm = np.empty((3, n))
        self.center = None
        if centered:
            self.center = sum(_unit_rows(raw[i:i+512]).sum(axis=0)
                              for i in range(0, n, 512)) / n
            self.length = np.empty(n)
        limit = (1 << (bits - 1)) - 1
        for start in range(0, n, 512):
            sl = slice(start, start + 512)
            rows = raw[sl].astype(np.float64)
            self.norm[sl] = np.linalg.norm(rows, axis=1)
            rows /= self.norm[sl, None]
            if centered:
                rows -= self.center
                self.length[sl] = np.linalg.norm(rows, axis=1)
            scale = np.max(np.abs(rows), axis=1) / limit
            scale[scale == 0] = 1
            self.scale[sl] = scale
            code = np.rint(rows / scale[:, None]).astype(np.int8)
            self.radius[sl] = np.linalg.norm(rows - code * scale[:, None], axis=1)
            if bits == 4:
                shifted = (code + 7).astype(np.uint8)
                self.codes[sl] = shifted[:, ::2] | (shifted[:, 1::2] << 4)
            else:
                self.codes[sl] = code.view(np.uint8)

    def search(self, raw, query):
        q = _unit_rows(query[None])[0]
        target = q if self.center is None else q - self.center
        scale = np.max(np.abs(target)) / 32767 or 1.0
        encoded = np.rint(target / scale).astype(np.int16)
        delta = np.linalg.norm(target - scale * encoded)
        accum = np.empty(len(raw), np.int32)
        self.scan(self.codes.ctypes.data, encoded.ctypes.data, len(raw),
                  self.bits, accum.ctypes.data)
        approx = accum * self.scale * scale
        error = self.radius + (1 + self.radius) * delta + 1e-12
        if self.center is not None:
            distance = np.linalg.norm(target)
            approx += 1 - 0.5 * (distance**2 + self.length**2)
            error = self.radius * distance + (self.length + self.radius) * delta + 1e-12
        # Exact seed scores give a tighter lower bound than interval endpoints.
        seeds = np.sort(_select(approx, 5))
        ids, scores = _rank(raw, q, seeds, 5, self.norm)
        extra = approx + error >= scores[-1]
        extra[seeds] = False
        others = np.flatnonzero(extra)
        if len(others):
            candidates = np.sort(np.concatenate((seeds, others)))
            ids, scores = _rank(raw, q, candidates, 5, self.norm)
        return ids, scores, len(others) + 5, seeds


def run(directory, scan, clustered):
    rng = np.random.default_rng(811)
    n = 4096 if clustered else 65536
    raw = np.lib.format.open_memmap(directory / "rows.npy", mode="w+",
                                    dtype=np.float32, shape=(n, 512))
    raw[:] = rng.standard_normal(raw.shape, dtype=np.float32)
    queries = rng.standard_normal((41, 512), dtype=np.float32)
    if clustered:
        center = queries[0].copy()
        raw[:] = center + raw * 1e-4
        raw[1::2] = raw[::2]
        queries[:] = center + queries * 1e-4
    raw.flush()
    baseline = CompactGallery(raw)
    indexes = [Index(raw, b, scan) for b in (8, 4)]
    if clustered:
        indexes.extend(Index(raw, b, scan, centered=True) for b in (8, 4))
    timings = [[] for _ in range(1 + len(indexes))]
    counts, recalls = [[] for _ in indexes], [[] for _ in indexes]
    for i, q in enumerate(queries):
        results = {}
        for j in np.roll(np.arange(len(timings)), i % len(timings)):
            start = perf_counter_ns()
            result = baseline.search(q[None], 5) if j == 0 else indexes[j-1].search(raw, q)
            elapsed = (perf_counter_ns() - start) / 1e6
            results[j] = result
            if i:
                timings[j].append(elapsed)
        for j in range(1, len(timings)):
            ids, scores, count, seeds = results[j]
            np.testing.assert_array_equal(ids, results[0][0][0])
            np.testing.assert_array_equal(scores, results[0][1][0])
            if i:
                counts[j-1].append(count)
                recalls[j-1].append(len(set(ids) & set(seeds)) / 5)
    print('rows', n, 'clustered', clustered, flush=True)
    for j, t in enumerate(timings):
        print(j, 'wall ms p50 p95', np.percentile(t, [50, 95]),
              'candidates median/max', (np.median(counts[j-1]), max(counts[j-1])) if j else n,
              'rough recall', np.mean(recalls[j-1]) if j else 1, flush=True)


if __name__ == '__main__':
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        lib = path / 'scan.so'
        subprocess.run(['cc', '-O3', '-march=native', '-fopenmp-simd', '-fPIC',
                        '-shared', Path(__file__).with_suffix('.c'), '-o', lib], check=True)
        scan = ctypes.CDLL(str(lib)).scan_codes
        scan.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                         ctypes.c_int, ctypes.c_void_p]
        scan.restype = None
        for clustered in (False, True):
            run(path, scan, clustered)
