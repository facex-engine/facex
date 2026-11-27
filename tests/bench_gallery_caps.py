"""Benchmarks for spherical caps, threshold decisions and exact query batches."""
import argparse
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns

import numpy as np

from bench_gallery_groups import GROUPS, fixture_views, prepare, rss
from facex.gallery import CompactGallery, _FP32_SLACK, _finalists, _rank, _unit_rows


def cap_index(directory):
    raw = np.memmap(directory / 'rows', 'f', 'r').reshape(GROUPS, -1, 512)
    count = raw.shape[1]
    packed = np.memmap(directory / 'packed', 'f', 'w+', shape=(GROUPS * count, 512))
    axes, cosines, offsets, owners, order = [], [], [0], [], []
    for owner, block in enumerate(raw):
        unit = _unit_rows(block)
        # Farthest first on a deterministic enrollment sample, without query labels.
        rng = np.random.default_rng(2026 + owner)
        sample_ids = rng.choice(len(unit), min(len(unit), 128), replace=False)
        sample = unit[sample_ids]
        chosen = [0]
        nearest = sample @ sample[0]
        for _ in range(min(7, len(sample) - 1)):
            nxt = nearest.argmin()
            if nearest[nxt] >= 1 - 1e-12:
                break
            chosen.append(nxt)
            nearest = np.maximum(nearest, sample @ sample[nxt])
        labels = np.argmax(unit @ sample[chosen].T, axis=1)
        for label in range(len(chosen)):
            take = np.flatnonzero(labels == label)
            if len(take) == 0:
                continue
            members = unit[take]
            axis = members[np.argmax(members @ members.mean(axis=0))]
            axes.append(axis)
            cosines.append(np.min(members @ axis) - 1e-12)
            owners.append(owner)
            start = offsets[-1]
            offsets.append(start + len(take))
            packed[start:offsets[-1]] = block[take]
            order.extend(owner * count + take)
    packed.flush()
    np.savez(directory / 'caps', axes=axes, cosines=cosines, owners=owners)
    np.savez(directory / 'layout', offsets=offsets, order=order,
             norms=np.concatenate([np.linalg.norm(x.astype('d'), axis=1)
                                   for x in np.array_split(packed, GROUPS)]))


def bounds(axes, cosines, q):
    dot = np.clip(axes @ q, -1, 1)
    c = np.clip(cosines, -1, 1)
    upper = np.where(dot >= c, 1, dot*c + np.sqrt((1-dot*dot)*(1-c*c)))
    return dot - 1e-12, upper + 1e-12


def top_five(raw, layout, upper, q):
    best, scores = np.empty(0, int), np.empty(0)
    for leaf in np.argsort(-upper, kind='stable'):
        if len(best) == 5 and upper[leaf] < scores[-1]:
            break
        start, stop = layout['offsets'][leaf:leaf+2]
        rows = raw[start:stop]
        norms = layout['norms'][start:stop]
        # Fixture rows have ordinary norms. This benchmark accepts this workload only.
        rough = (rows @ q.astype('f')).astype('d') / norms
        k = min(5, len(rows))
        take = _finalists(rough, k, _FP32_SLACK)
        ids, values = _rank(rows, q, take, k, norms)
        best = np.concatenate((best, layout['order'][start+ids]))
        scores = np.concatenate((scores, values))
        keep = np.lexsort((best, -scores))[:5]
        best, scores = best[keep], scores[keep]
        raw._mmap.madvise(mmap.MADV_DONTNEED)
    return best, scores


def decision(low, high, threshold, margin):
    winner = low.argmax()
    rival = np.max(np.delete(high, winner))
    if low[winner] >= threshold and low[winner] > rival and low[winner] - rival >= margin:
        return int(winner)
    if high.max() < threshold:
        return -1
    return None


def evaluate(path, method):
    queries = np.load(path / 'queries.npy')
    rng = np.random.default_rng(77)
    queries = np.vstack((queries, rng.standard_normal((15, 512)).astype('f')))
    with np.load(path / 'oracle.npz') as archive:
        truth = dict(archive)
    if method == 'batch':
        raw = np.memmap(path / 'rows', 'f', 'r').reshape(-1, 512)
        gallery = CompactGallery(raw)
        queries = queries[:40]
        timings = [[], []]
        for repeat in range(5):
            for start in range(0, len(queries), 8):
                batch = queries[start:start+8]
                for variant in ((0, 1) if repeat % 2 else (1, 0)):
                    t = perf_counter_ns()
                    if variant:
                        unit = _unit_rows(batch)
                        dots = unit.astype('f') @ gallery._rows.T
                        out = []
                        for q, row in zip(unit, dots):
                            rough = row.astype('d') / gallery._norms
                            take = _finalists(rough, 5, _FP32_SLACK)
                            out.append(_rank(gallery._rows, q, take, 5, gallery._norms))
                        ids, scores = map(np.asarray, zip(*out))
                    else:
                        ids, scores = gallery.search(batch, 5)
                    elapsed = (perf_counter_ns() - t) / 1e6
                    np.testing.assert_array_equal(ids, truth['ids'][start:start+8])
                    np.testing.assert_array_equal(scores, truth['scores'][start:start+8])
                    if repeat:
                        timings[variant].append(elapsed)
        print('batch of eight p50/p95 ms', np.percentile(timings, [50, 95], axis=1).T)
        return
    if method != 'baseline':
        with np.load(path / 'caps.npz') as archive:
            caps = dict(archive)
    if method == 'certificate':
        print('certificate bytes', sum(v.nbytes for v in caps.values()))
        for threshold in (.65, .85):
            decisions = np.zeros((3, 3), int)
            for i, query in enumerate(queries):
                lower, upper = bounds(caps['axes'], caps['cosines'], _unit_rows(query[None])[0])
                owners = caps['owners']
                low, high = np.full(GROUPS, -np.inf), np.full(GROUPS, -np.inf)
                np.maximum.at(low, owners, lower)
                np.maximum.at(high, owners, upper)
                result = decision(low, high, threshold, .05)
                ref = truth['identity'][i]
                assert np.all(low <= ref) and np.all(ref <= high)
                expected = decision(ref, ref, threshold, .05)
                if result is not None:
                    assert result == (expected if expected is not None else -1)
                decisions[i//15, 0 if result is None else 1 if result == -1 else 2] += 1
            print('threshold', threshold, 'unresolved/reject/accept', decisions.tolist())
        return
    if method == 'baseline':
        rows = np.memmap(path / 'rows', 'f', 'r').reshape(-1, 512)
        gallery = CompactGallery(rows)
        del rows
    else:
        with np.load(path / 'layout.npz') as archive:
            layout = dict(archive)
        raw = np.memmap(path / 'packed', 'f', 'r').reshape(-1, 512)
    times = [[], [], []]
    for repeat in range(4):
        for i, query in enumerate(queries):
            t = perf_counter_ns()
            if method == 'baseline':
                ids, scores = gallery.search(query[None], 5)
                ids, scores = ids[0], scores[0]
            else:
                q = _unit_rows(query[None])[0]
                _, upper = bounds(caps['axes'], caps['cosines'], q)
                ids, scores = top_five(raw, layout, upper, q)
            elapsed = (perf_counter_ns() - t) / 1e6
            np.testing.assert_array_equal(ids, truth['ids'][i])
            np.testing.assert_array_equal(scores, truth['scores'][i])
            if repeat:
                times[i//15].append(elapsed)
    status = Path('/proc/self/status').read_text().splitlines()
    peak_kib = int(next(x for x in status if x.startswith('VmHWM:')).split()[1])
    print(method, 'near/boundary/random p50/p95 ms', np.percentile(times, [50,95], axis=1).T,
          'end/peak RSS MiB', rss(), peak_kib / 1024)


def oracle(path):
    queries = np.load(path / 'queries.npy')
    queries = np.vstack((queries, np.random.default_rng(77).standard_normal((15,512)).astype('f')))
    raw = np.memmap(path / 'rows', 'f', 'r').reshape(GROUPS,-1,512)
    gallery = CompactGallery(raw.reshape(-1,512))
    ids, scores = gallery.search(queries, 5)
    per_identity = np.stack([CompactGallery(block).search(queries, 1)[1][:,0]
                             for block in raw], axis=1)
    np.savez(path / 'oracle', ids=ids, scores=scores, identity=per_identity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--worker', choices=['lookup','baseline','certificate','batch'])
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.worker:
        evaluate(args.input, args.worker)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            prepare(path, fixture_views(args.input), True)
            cap_index(path)
            oracle(path)
            for method in ('lookup','baseline','certificate','batch'):
                subprocess.run([sys.executable, __file__, path, '--worker', method], check=True)
