"""Compare bounded group search with exact cosine on fixture and scaled galleries."""
import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns

import numpy as np
from PIL import Image

from bench_gallery_codes import Index
from facex.gallery import CompactGallery, _unit_rows

HERE = Path(__file__).resolve().parent
GROUPS = 15


def fixture_patches():
    patches = []
    items = json.loads((HERE / "golden/summary.json").read_text())["items"]
    for item in items:
        with np.load(HERE / "golden" / (item["identity"] + ".npz")) as data:
            image = Image.fromarray(data["image"])
        face = max(item["faces"], key=lambda item: item["score"])
        xy = np.array([face["x1"], face["y1"], face["x2"], face["y2"]])
        # Deterministic crop shifts from a single photo per group.
        for dx, dy in [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2),
                       (2, 2), (-2, -2), (2, -2), (-1, 1)]:
            box = tuple(np.rint(xy + [dx, dy, dx, dy]).astype(int))
            patch = image.crop(box).resize((112, 112), Image.Resampling.BILINEAR)
            patches.append((np.asarray(patch, dtype='f') / 127.5 - 1).transpose(2, 0, 1))
    return np.asarray(patches).tobytes()


def embed_patches(weights, patches, executable):
    result = subprocess.run([executable, weights, '--batch'], input=patches,
                            capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype='f').reshape(GROUPS, 9, 512)


def fixture_views(weights):
    return embed_patches(weights, fixture_patches(), HERE.parent / 'facex-cli')


def kernel(path):
    scan = ctypes.CDLL(str(path / 'scan.so')).scan_codes
    scan.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                     ctypes.c_int, ctypes.c_void_p]
    scan.restype = None
    return scan


def prepare(path, views, scaled):
    rng = np.random.default_rng(2069)
    per = 4369 if scaled else 8
    raw = np.memmap(path / 'rows', dtype='f', mode='w+', shape=(GROUPS, per, 512))
    for g, block in enumerate(raw):
        if scaled:
            anchors = _unit_rows(views[g, :8])
            block[:] = anchors[np.arange(per) % 8] + rng.normal(0, .002, block.shape)
        else:
            block[:] = views[g, :8]
        idx = Index(block, 8, None, centered=True)
        np.savez(path / f'g{g}.npz', **{k: v for k, v in vars(idx).items()
                                         if isinstance(v, np.ndarray)})
    raw.flush()
    q = _unit_rows(views[:, 8])
    similarity = q @ q.T
    np.fill_diagonal(similarity, -np.inf)
    boundary = q + q[similarity.argmax(axis=1)]
    queries = np.vstack((q, boundary)).astype('f')
    np.save(path / 'queries', queries)
    gallery = CompactGallery(raw.reshape(-1, 512))
    ids, scores = gallery.search(queries, 5)
    np.savez(path / 'truth', ids=ids, scores=scores)


def lookup(raw, groups, centers, radii, query, trim=False):
    unit = _unit_rows(query[None])[0]
    distances = np.linalg.norm(centers - unit, axis=1)
    upper = 1 - .5 * np.maximum(0, distances - radii)**2 + 1e-12
    best = np.empty(0, int)
    scores = np.empty(0)
    visits = 0
    for g in np.argsort(-upper, kind='stable'):
        if len(best) and upper[g] < scores[-1]:
            break
        ids, values, _, _ = groups[g].search(raw[g], query)
        if trim:
            raw._mmap.madvise(mmap.MADV_DONTNEED)
        best = np.concatenate((best, g * raw.shape[1] + ids))
        scores = np.concatenate((scores, values))
        keep = np.lexsort((best, -scores))[:5]
        best, scores = best[keep], scores[keep]
        visits += 1
    return best, scores, visits


def rss():
    fields = Path('/proc/self/status').read_text().splitlines()
    return int(next(x for x in fields if x.startswith('VmRSS:')).split()[1]) / 1024


def residency(raw):
    pages = (raw.nbytes + os.sysconf('SC_PAGE_SIZE') - 1) // os.sysconf('SC_PAGE_SIZE')
    vec = (ctypes.c_ubyte * pages)()
    lib = ctypes.CDLL(None, use_errno=True)
    if lib.mincore(ctypes.c_void_p(raw.ctypes.data), ctypes.c_size_t(raw.nbytes), vec):
        raise OSError(ctypes.get_errno(), 'mincore')
    return round(sum(x & 1 for x in vec) / pages, 4)


def measure(path, method):
    queries = np.load(path / 'queries.npy')
    with np.load(path / 'truth.npz') as truth:
        ids, scores = truth['ids'], truth['scores']
    rows = path / 'rows'
    with rows.open('rb') as fd:
        os.fsync(fd.fileno())
        os.posix_fadvise(fd.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    raw = np.memmap(rows, dtype='f', mode='r').reshape(GROUPS, -1, 512)
    initial = rss()
    load_start = perf_counter_ns()
    if method.startswith('group'):
        scan = kernel(path)
        groups = []
        for g in range(GROUPS):
            idx = Index.__new__(Index)
            idx.bits, idx.scan = 8, scan
            with np.load(path / f'g{g}.npz') as arrays:
                vars(idx).update({k: arrays[k] for k in arrays.files})
            groups.append(idx)
        centers = np.stack([g.center for g in groups])
        radii = np.array([g.length.max() for g in groups])
        def search(q):
            return lookup(raw, groups, centers, radii, q, trim=method == 'group_trim')
    else:
        gallery = CompactGallery(raw.reshape(-1, 512))
        del raw
        raw = np.memmap(rows, dtype='f', mode='r').reshape(GROUPS, -1, 512)
        def search(q):
            found, values = gallery.search(q[None], 5)
            return found[0], values[0], GROUPS
    loaded = rss()
    load_ms = (perf_counter_ns() - load_start) / 1e6
    resident = residency(raw)
    times, visited = [[], []], [[], []]
    for iteration, i in enumerate([0] + list(range(len(queries))) * 3):
        start = perf_counter_ns()
        found, values, count = search(queries[i])
        elapsed = (perf_counter_ns() - start) / 1e6
        if iteration == 0:
            first_ms, first_rss = elapsed, rss()
        else:
            times[i // GROUPS].append(elapsed)
            visited[i // GROUPS].append(count)
        np.testing.assert_array_equal(found, ids[i])
        np.testing.assert_array_equal(values, scores[i])
    print(method, 'load_ms', round(load_ms, 3), 'first_ms', round(first_ms, 3), 'resident_before', resident,
          'RSS MiB at start/loaded/first/end', [initial, loaded, first_rss, rss()],
          'p50/p95 ms', np.round(np.percentile(times, [50, 95], axis=1).T, 3),
          'peak', next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')), 'KiB', 'median groups', [np.median(visited[0]), np.median(visited[1])], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--worker', choices=['group', 'group_trim', 'baseline'])
    args = p.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.worker:
        measure(args.input, args.worker)
    else:
        views = fixture_views(args.input)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            subprocess.run(['cc', '-O3', '-march=native', '-fopenmp-simd', '-shared',
                            '-fPIC', HERE / 'bench_gallery_codes.c', '-o', path / 'scan.so'],
                           check=True)
            for scaled in (False, True):
                prepare(path, views, scaled)
                print('scaled', scaled, flush=True)
                for method in ('group', 'group_trim', 'baseline'):
                    subprocess.run([sys.executable, __file__, path, '--worker', method], check=True)
