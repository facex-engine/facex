"""Confirm the selected detector channel order on a further disjoint image subset."""
import argparse
import ctypes
import io
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image

from bench_yunet_bottlenecks import GRID, annotations, load_lib, match, native_run, prepare_rectangle, source_boxes


def collect(lib, engine, photos, labels, names):
    output = []
    for name in names:
        image = Image.open(io.BytesIO(photos.read("WIDER_val/images/" + name))).convert("RGB")
        w, h = image.size
        canvas, rw, rh = prepare_rectangle(np.asarray(image))
        ch, cw = canvas.shape[:2]
        gt = labels[name].copy()
        small = np.sqrt(gt[:, 2:].prod(1)) * 320 / max(w, h) < 16
        gt[:, 2:] += gt[:, :2]
        predictions = []
        for arr in (canvas, np.ascontiguousarray(canvas[:, :, ::-1])):
            rows = native_run(lib, engine, arr, capacity=32)
            predictions.append(source_boxes(rows, w, h, rw, rh, cw, ch))
        output.append((predictions, gt, small))
    return output


def evaluate(data, order, t):
    records = []
    for predictions, gt, small in data:
        det = predictions[order]
        det = det[det[:, 4] >= np.float32(t)]
        good = match(det, gt)
        records.append((len(good), len(det) - len(good), small[good].sum()))
    return np.array(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "images", "labels"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    with ZipFile(args.labels) as archive:
        labels = annotations(archive)
    pool = set(labels)
    for seed, n in ((114, 640), (725, 128)):
        pool -= set(np.random.default_rng(seed).choice(sorted(pool), n, replace=False))
    previous = np.random.default_rng(3519).choice(sorted(pool), 672, replace=False)
    names = np.random.default_rng(9101).choice(sorted(pool-set(previous)), 384, replace=False)
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as photos:
        lib = load_lib(Path(tmp), "native")
        eng = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
        if not eng:
            raise RuntimeError("Failed to open YuNet weights")
        lib.detect_set_score_threshold(eng, ctypes.c_float(.40))
        try:
            dev = collect(lib, eng, photos, labels, previous[:160])
            budget = int(.75 * evaluate(dev, 0, .58).sum(0)[1])
            candidates = [(t, evaluate(dev, 1, t).sum(0)) for t in GRID]
            threshold, values = max((p for p in candidates if p[1][1] <= budget),
                                    key=lambda p: (p[1][2], p[1][0]))
            print("locked conservative threshold", threshold, "dev", values, flush=True)
            data = collect(lib, eng, photos, labels, names)
        finally:
            lib.detect_free(eng)
    print("test GT and small", sum(len(r[1]) for r in data), sum(r[2].sum() for r in data))
    base = evaluate(data, 0, .58)
    bs = np.random.default_rng(478).integers(len(names), size=(2000, len(names)))
    for order, t in ((0, .58), (1, .58), (1, .66), (1, threshold)):
        counts = evaluate(data, order, t)
        print(order, t, counts.sum(0), np.percentile((counts-base)[bs].sum(1), [2.5, 97.5], axis=0))


if __name__ == "__main__":
    main()
