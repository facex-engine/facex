"""Replay a local MOT17Labels.zip through the baseline and current C tracker.

Uses the seven training sequences with public FRCNN detections. These local
counts use the matching protocol below; official MOTChallenge scores require
its complete evaluation procedure.
"""
import argparse
import ctypes
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
BASE = "b21cc3d2f292818b23879f988b32dce34b335895"
DET = np.dtype([("box", "f", 4), ("score", "f"), ("cls", "i")])
TRACK = np.dtype([("id", "i"), ("cls", "i"), ("score", "f"), ("box", "f", 4),
                  ("age", "i"), ("hits", "i"), ("since", "i")])


def group_frames(rows):
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    return {int(frame[0, 0]): frame for frame in
            np.split(rows, np.flatnonzero(np.diff(rows[:, 0])) + 1) if len(frame)}


def corners(rows):
    boxes = rows[:, 2:6].copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def matches(a, b):
    inter = (np.minimum(a[:, None, 2:], b[None, :, 2:]) -
             np.maximum(a[:, None, :2], b[None, :, :2])).clip(0).prod(2)
    union = ((a[:, 2:] - a[:, :2]).prod(1)[:, None] +
             (b[:, 2:] - b[:, :2]).prod(1) - inter)
    iou = inter / np.maximum(union, 1e-8)
    rows, cols = linear_sum_assignment(np.where(iou > .5, 1 + iou, 0), maximize=True)
    return [(i, j) for i, j in zip(rows, cols) if iou[i, j] > .5]


def evaluate(source, archive):
    binary = source.with_suffix(".so")
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                    "-I", str(ROOT / "nn2/include"), str(source), "-lm", "-o", str(binary)],
                   check=True)
    lib = ctypes.CDLL(str(binary))
    lib.nn2_tracker_create.restype = ctypes.c_void_p
    matched = missing = switches = 0
    times = []
    for gt_name in sorted(n for n in archive.namelist() if "FRCNN/gt/gt.txt" in n):
        gt = np.loadtxt(archive.open(gt_name), delimiter=",")
        det = np.loadtxt(archive.open(gt_name.replace("gt/gt", "det/det")), delimiter=",")
        gt_frames = group_frames(gt[(gt[:, 6] == 1) & (gt[:, 7] == 1) & (gt[:, 8] >= .5)])
        det_frames = group_frames(det[det[:, 6] >= .5])
        engine = None
        try:
            for frame in range(1, max(gt_frames) + 1):
                if frame % 120 == 1:
                    if engine:
                        lib.nn2_tracker_free(engine)
                    engine = ctypes.c_void_p(lib.nn2_tracker_create(5, 2))
                    identities = {}
                truth = gt_frames.get(frame, np.empty((0, 9)))
                observed = det_frames.get(frame, np.empty((0, 10)))
                detections = np.zeros(len(observed), DET)
                detections["box"] = corners(observed)
                detections["score"] = observed[:, 6]
                out = np.empty(128, TRACK)
                t = time.thread_time_ns()
                n = lib.nn2_tracker_update(engine, detections.ctypes.data_as(ctypes.c_void_p),
                                           len(detections), out.ctypes.data_as(ctypes.c_void_p), len(out))
                times.append((time.thread_time_ns() - t) / 1000)
                current = out[:n][out["since"][:n] == 0]
                pairs = matches(current["box"], corners(truth))
                matched += len(pairs)
                missing += len(truth) - len(pairs)
                for i, j in pairs:
                    who, track_id = int(truth[j, 1]), int(current["id"][i])
                    switches += who in identities and identities[who] != track_id
                    identities[who] = track_id
        finally:
            if engine:
                lib.nn2_tracker_free(engine)
    return matched, missing, switches, round(statistics.median(times), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.labels) as archive:
        directory = Path(tmp)
        for name in ("baseline", "current"):
            source = directory / (name + ".c")
            path = "nn2/src/track.c"
            source.write_bytes(subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
                               if name == "baseline" else (ROOT / path).read_bytes())
            print(name, "matched, missed, ID changes, CPU microseconds",
                  evaluate(source, archive))


if __name__ == "__main__":
    main()
