"""Equal pixel budget WIDER FACE experiment. Run with the YuNet binary and WIDER archives."""
import argparse
import ctypes
import io
import os
from pathlib import Path
import subprocess
import tempfile
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image

from verify_yunet import ROOT, annotations, native_run, match

MODES = ("square640", "rectangle", "two448", "four320")
GRID = np.round(np.arange(.30, .911, .01), 2)


def windows(mode, w, h):
    if mode in ("square640", "rectangle"):
        return [(0, 0, w, h)]
    def halves(n):
        # Adjacent crops overlap by one eighth of the full side.
        end = min(n, (9 * n + 15) // 16)
        return [(0, end), (n - end, n)]
    xs = halves(w) if mode == "four320" or w >= h else [(0, w)]
    ys = halves(h) if mode == "four320" or w < h else [(0, h)]
    return [(l, t, r, b) for t, b in ys for l, r in xs]


def shape(mode, w, h):
    if mode == "rectangle":
        # Choose the largest uniform scale with padded area <= 640 squared.
        # Avoid pathological side lengths on panoramas.
        scale = min(1280 / max(w, h), max(
            min(32 * k / w, 32 * (400 // k) / h) for k in range(1, 41)))
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        return rw, rh, (rw + 31) // 32 * 32, (rh + 31) // 32 * 32
    side = {"square640": 640, "two448": 448, "four320": 320}[mode]
    scale = side / max(w, h)
    return max(1, round(w * scale)), max(1, round(h * scale)), side, side


def deduplicate(rows):
    order = np.argsort(-rows[:, 4], kind="stable")
    keep = []
    area = (rows[:, 2:4] - rows[:, :2]).prod(1)
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        inter = (np.minimum(rows[i, 2:4], rows[rest, 2:4]) -
                 np.maximum(rows[i, :2], rows[rest, :2])).clip(0).prod(1)
        order = rest[inter <= .4 * (area[i] + area[rest] - inter)]
    return rows[keep]


def predict(lib, engine, image, mode):
    rows = []
    for l, t, r, b in windows(mode, *image.size):
        w, h = r - l, b - t
        rw, rh, cw, ch = shape(mode, w, h)
        ox, oy = (cw - rw) // 2, (ch - rh) // 2
        canvas = np.zeros((ch, cw, 3), dtype="u1")
        region = image if (w, h) == image.size else image.crop((l, t, r, b))
        canvas[oy:oy+rh, ox:ox+rw] = region.resize((rw, rh))
        det = native_run(lib, engine, canvas, capacity=10000)[:, :5].copy()
        det[:, (0, 2)] = (det[:, (0, 2)] - ox) * (w / rw) + l
        det[:, (1, 3)] = (det[:, (1, 3)] - oy) * (h / rh) + t
        rows.append(det)
    return rows[0] if len(rows) == 1 else deduplicate(np.concatenate(rows))


def counts(out, gt, small, threshold):
    det = out[out[:, 4] >= threshold]
    found = match(det, gt)
    return [len(found), len(det) - len(found), small[found].sum()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "images", "labels"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as images, ZipFile(args.labels) as archive:
        binary = Path(tmp) / "detector.so"
        subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                        str(ROOT / "src/detect.c"), "-lm", "-o", str(binary)], check=True)
        lib = ctypes.CDLL(str(binary))
        lib.detect_init.restype = ctypes.c_void_p
        engine = ctypes.c_void_p(lib.detect_init(os.fsencode(args.weights)))
        if not engine:
            raise ValueError("Cannot load YuNet weights")
        lib.detect_set_score_threshold(engine, ctypes.c_float(.30))
        try:
            run(lib, engine, images, annotations(archive))
        finally:
            lib.detect_free(engine)


def run(lib, engine, images, labels):
    names = sorted(labels)
    test = np.random.default_rng(114).choice(names, 640, replace=False)
    dev = np.random.default_rng(725).choice(sorted(set(names) - set(test)), 128, replace=False)
    rng = np.random.default_rng(815)
    for mode in MODES:
        predict(lib, engine, Image.new("RGB", (800, 600)), mode)
    thresholds = {}
    for split, subset in (("dev", dev), ("test", test)):
        grid = {m: GRID if split == "dev" else [thresholds[m]] for m in MODES}
        stats = {m: [] for m in MODES}
        times = {m: [] for m in MODES}
        totals = np.zeros(2, int)
        for name in subset:
            image = Image.open(io.BytesIO(images.read("WIDER_val/images/" + name))).convert("RGB")
            gt = labels[name].copy()
            small = np.sqrt(gt[:, 2] * gt[:, 3]) * (320 / max(image.size)) < 16
            gt[:, 2:] += gt[:, :2]
            totals += [len(gt), small.sum()]
            for mode in rng.permutation(MODES):
                start = time.thread_time_ns()
                det = predict(lib, engine, image, mode)
                times[mode].append((time.thread_time_ns() - start) * 1e-6)
                stats[mode].append([counts(det, gt, small, t) for t in grid[mode]])
        print(split, "GT and small:", totals.tolist(), flush=True)
        data = {m: np.array(stats[m]) for m in MODES}
        if split == "dev":
            ref = int(np.flatnonzero(GRID == .5)[0])
            fp_budget = data["square640"][:, ref, 1].sum()
        else:
            bootstrap = np.random.default_rng(637).integers(len(test), size=(2000, len(test)))
        for m in MODES:
            overall = data[m].sum(0)
            if split == "dev":
                ok = np.flatnonzero(overall[:, 1] <= fp_budget)
                if not len(ok):
                    raise ValueError("No threshold within false positive budget")
                i = ref if m == "square640" else ok[np.argmax(overall[ok, 0])]
                thresholds[m] = GRID[i]
            else:
                i = 0
            print(m, "threshold", thresholds[m], "TP FP smallTP", overall[i].tolist(),
                  "CPU mean/median ms", np.round([np.mean(times[m]), np.median(times[m])], 3), flush=True)
            if split == "test" and m != "square640":
                delta = data[m][:, 0] - data["square640"][:, 0]
                print("paired 95% interval TP FP smallTP",
                      np.percentile(delta[bootstrap].sum(1), [2.5, 97.5], axis=0).round(1).tolist(), flush=True)


if __name__ == "__main__":
    main()
