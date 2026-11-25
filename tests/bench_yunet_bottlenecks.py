"""Evaluate eight independent changes to the rectangular YuNet operating point."""
import argparse
import ctypes
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
from verify_yunet import ROOT, annotations, match, native_run

sys.path.insert(0, str(ROOT / "python"))
from facex.yunet import prepare_rectangle

MODES = ("baseline", "min8", "wide_aspect", "iou55", "relative_merge",
         "limit128", "small_score", "box_expand", "bgr")
GRID = np.round(np.arange(.40, .911, .01), 2)


def load_lib(tmp, name, raw=False):
    # Build the exact reference revision so historical results remain reproducible.
    src = subprocess.check_output(["git", "show", "cb672bee:src/detect.c"], cwd=ROOT, text=True)
    if raw:
        start = src.index("                if (bw < 16")
        end = src.index("                candidates[nf].score", start)
        src = src[:start] + "                if (bw <= 0 || bh <= 0) continue;\n" + src[end:]
        start = src.index("    if (nf > 1) nf = nms_sort")
        end = src.index("    if (nf > max_faces)", start)
        src = src[:start] + src[end:]
    file = tmp / (name + ".c")
    file.write_text(src)
    so = tmp / (name + ".so")
    subprocess.run(["gcc", "-O3", "-march=native", "-I", str(ROOT / "include"),
                    "-shared", "-fPIC", str(file), "-lm", "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.detect_init.restype = ctypes.c_void_p
    return lib


def suppress(rows, overlaps):
    active = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        if active[i]:
            active[i+1:] &= ~overlaps[i, i+1:]
    return rows[active]


def replay(raw, mode, cw, ch):
    rows = raw.copy()
    dims = rows[:, 2:4] - rows[:, :2]
    w, h = dims.T
    aspect = w / (h + 1e-6)
    lo, hi = (.18, 4.) if mode == "wide_aspect" else (.3, 3.)
    minimum = 8 if mode == "min8" else 16
    ok = ((dims.min(1) >= minimum) & (w <= cw*.95) & (h <= ch*.95) &
          (aspect >= lo) & (aspect <= hi))
    rows = rows[ok]
    if mode == "small_score":
        rows[:, 4] = np.minimum(1, rows[:, 4] * np.where(dims[ok].min(1) < 32, 1.10, 1.))
    rows = rows[np.argsort(-rows[:, 4], kind="stable")]
    a = rows[:, :4]
    area = (a[:, 2:] - a[:, :2]).prod(1)
    inter = (np.minimum(a[:, None, 2:], a[None, :, 2:]) -
             np.maximum(a[:, None, :2], a[None, :, :2])).clip(0).prod(2)
    iou = inter / np.maximum(area[:, None] + area - inter, 1e-8)
    rows = suppress(rows, iou > (.55 if mode == "iou55" else .4))
    nose = rows[:, 9:11]
    eyes = (rows[:, 5:7] + rows[:, 7:9]) / 2
    sizes = np.sqrt((rows[:, 2:4] - rows[:, :2]).prod(1))
    scale = np.minimum(sizes[:, None], sizes)
    nose_bound, eye_bound = (.25*scale, .20*scale) if mode == "relative_merge" else (25, 20)
    close = (np.linalg.norm(nose[:, None] - nose, axis=2) < nose_bound) | (
        np.linalg.norm(eyes[:, None] - eyes, axis=2) < eye_bound)
    rows = suppress(rows, close)[:128 if mode == "limit128" else 32]
    if mode == "box_expand":
        margin = (rows[:, 2:4] - rows[:, :2]) * .04
        rows[:, :2] -= margin
        rows[:, 2:4] += margin
    return rows


def source_boxes(rows, w, h, rw, rh, cw, ch):
    boxes = rows[:, :5].copy()
    for axis, original, resized, full in ((0, w, rw, cw), (1, h, rh, ch)):
        ids = [axis, axis+2]
        boxes[:, ids] = np.clip((boxes[:, ids] - (full-resized)//2) *
                                (original/resized), 0, original)
    return boxes


def extract(weights, images, labels, names, tmp):
    libs = [load_lib(tmp, name, raw) for name, raw in (("reference", False), ("candidate", True))]
    engines = [ctypes.c_void_p(lib.detect_init(bytes(weights))) for lib in libs]
    if not all(engines):
        raise RuntimeError("Cannot open weights")
    for lib, eng, threshold in zip(libs, engines, (.58, .30)):
        lib.detect_set_score_threshold(eng, ctypes.c_float(threshold))
    records = []
    try:
        for i, name in enumerate(names):
            im = Image.open(io.BytesIO(images.read("WIDER_val/images/" + name))).convert("RGB")
            w, h = im.size
            canvas, rw, rh = prepare_rectangle(np.asarray(im))
            ch, cw = canvas.shape[:2]
            rgb, bgr = [native_run(libs[1], engines[1], x, capacity=10000)
                        for x in (canvas, np.ascontiguousarray(canvas[:, :, ::-1]))]
            gt = labels[name].copy()
            small = np.sqrt(gt[:, 2:4].prod(1))*320/max(w, h) < 16
            gt[:, 2:] += gt[:, :2]
            outputs = []
            for mode in MODES:
                rows = replay(bgr if mode == "bgr" else rgb, mode, cw, ch)
                if mode == "baseline" and i % 13 == 0:
                    actual = native_run(libs[0], engines[0], canvas, capacity=32)
                    np.testing.assert_array_equal(actual, rows[rows[:, 4] >= .58])
                outputs.append(source_boxes(rows, w, h, rw, rh, cw, ch))
            # An oracle can select any candidate and discard all false positives.
            # Restricting GT to small boxes gives an optimistic localization ceiling.
            boxes = source_boxes(rgb, w, h, rw, rh, cw, ch)
            minimum_side = (rgb[:, 2:4] - rgb[:, :2]).min(1)
            confident = rgb[:, 4] >= .58
            masks = (rgb[:, 4] >= .30, confident, confident & (minimum_side >= 8),
                     confident & (minimum_side >= 16))
            oracle = [len(match(boxes[mask], gt[small])) for mask in masks]
            records.append((gt, small, outputs, oracle))
    finally:
        for lib, eng in zip(libs, engines):
            lib.detect_free(eng)
    return records


def measure(data, mode, thresholds):
    result = np.zeros((len(data), len(thresholds), 3), dtype=int)
    for i, (gt, small, outputs, _) in enumerate(data):
        det = outputs[mode]
        by_count = {}
        for j, t in enumerate(thresholds):
            n = int((det[:, 4] >= np.float32(t)).sum())
            if n not in by_count:
                found = match(det[:n], gt)
                by_count[n] = (len(found), n - len(found), small[found].sum())
            result[i, j] = by_count[n]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("weights", "images", "labels"):
        ap.add_argument(arg, type=Path)
    args = ap.parse_args()
    with ZipFile(args.labels) as archive:
        labels = annotations(archive)
    old = np.random.default_rng(114).choice(sorted(labels), 640, replace=False)
    pool = set(labels) - set(old)
    pool -= set(np.random.default_rng(725).choice(sorted(pool), 128, replace=False))
    names = np.random.default_rng(3519).choice(sorted(pool), 672, replace=False)
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as photos:
        data = extract(args.weights, photos, labels, names, Path(tmp))
    dev, test = data[:160], data[160:]
    budget = measure(dev, 0, [.58]).sum(0)[0, 1]
    locked = []
    for m, name in enumerate(MODES):
        scores = measure(dev, m, GRID).sum(0)
        feasible = np.flatnonzero(scores[:, 1] <= budget)
        i = np.flatnonzero(GRID == .58)[0] if m == 0 else max(
            feasible, key=lambda j: (scores[j, 2], scores[j, 0], -abs(GRID[j]-.58)))
        locked.append((GRID[i], scores[i, 2]))
        print("dev", name, GRID[i], scores[i].tolist(), flush=True)
    winner = int(np.argmax([entry[1] for entry in locked]))
    print("locked winner", MODES[winner], flush=True)
    print("test images, annotations, small", len(test), sum(len(x[0]) for x in test),
          sum(x[1].sum() for x in test), flush=True)
    print("small candidate ceilings", np.sum([x[3] for x in test], axis=0).tolist())
    base = measure(test, 0, [.58])[:, 0]
    bootstrap = np.random.default_rng(547).integers(len(test), size=(2000, len(test)))
    for m, name in enumerate(MODES):
        counts = measure(test, m, [locked[m][0]])[:, 0]
        ci = np.percentile((counts-base)[bootstrap].sum(1), [2.5, 97.5], axis=0)
        print("test", name, counts.sum(0).tolist(), "paired interval", ci.round(1).tolist())


if __name__ == "__main__":
    main()
