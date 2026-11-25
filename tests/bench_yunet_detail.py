"""Compare detail recovery at full frame, crop and confidence operating points."""
import argparse
import ctypes
import io
import os
from pathlib import Path
import tempfile
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image

from bench_yunet_bottlenecks import annotations, load_lib, match, native_run, prepare_rectangle, source_boxes
from bench_yunet_tiles import deduplicate

MODES = ("current", "soft", "full800", "guided448")


def detect(lib, engine, rgb, side):
    h, w = rgb.shape[:2]
    canvas, rw, rh = prepare_rectangle(rgb, bgr=True, side=side)
    ch, cw = canvas.shape[:2]
    boxes = native_run(lib, engine, canvas, capacity=32)
    return source_boxes(boxes, w, h, rw, rh, cw, ch)


def refine(lib, engine, rgb, boxes):
    h, w = rgb.shape[:2]
    dim = np.sqrt((boxes[:, 2:4] - boxes[:, :2]).prod(1)) * 320 / max(w, h)
    hints = boxes[(dim < 20) & (boxes[:, 4] < .85)]
    if not len(hints):
        return boxes, False
    # Select the most populated half-frame neighborhood of small proposals.
    centers = (hints[:, :2] + hints[:, 2:4]) * .5
    delta = np.abs(centers[:, None] - centers)
    counts = ((delta[:,:,0] <= w/4) & (delta[:,:,1] <= h/4)).sum(1)
    cx, cy = centers[np.argmax(counts)]
    cw, ch = max(1, w//2), max(1, h//2)
    x, y = int(np.clip(cx-cw/2, 0, w-cw)), int(np.clip(cy-ch/2, 0, h-ch))
    extra = detect(lib, engine, rgb[y:y+ch, x:x+cw], 448)
    extra[:, :4] += [x, y, x, y]
    return deduplicate(np.concatenate((boxes, extra)))[:32], True


def score(data, mode, threshold):
    records, identities = [], []
    for gt, small, predictions, _, _ in data:
        det = predictions[mode]
        det = det[det[:,4] >= np.float32(threshold)]
        good = match(det, gt)
        records.append([len(good), len(det)-len(good), small[good].sum()])
        identities.append(set(good[small[good]]))
    return np.array(records), identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "images", "labels"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with ZipFile(args.labels) as archive:
        labels = annotations(archive)
    pool = set(labels)
    for seed, size in ((114, 640), (725, 128), (3519, 672), (9101, 384)):
        pool -= set(np.random.default_rng(seed).choice(sorted(pool), size, replace=False))
    names = np.random.default_rng(60177).choice(sorted(pool), 448, replace=False)
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as photos:
        lib = load_lib(Path(tmp), "native")
        engine = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
        if not engine:
            raise RuntimeError("Failed to load detector")
        lib.detect_set_score_threshold(engine, ctypes.c_float(.35))
        try:
            for side in (448, 640, 800):
                detect(lib, engine, np.zeros((480, 640, 3), dtype=np.uint8), side)
            data = []
            for i, name in enumerate(names):
                image = np.asarray(Image.open(io.BytesIO(photos.read("WIDER_val/images/"+name))).convert("RGB"))
                gt = labels[name].copy()
                small = np.sqrt(gt[:,2:].prod(1))*320/max(image.shape[:2]) < 16
                gt[:,2:] += gt[:,:2]
                boxes, ms = {}, {}
                for side in ((640,800) if i%2 else (800,640)):
                    start = time.thread_time_ns()
                    boxes[side] = detect(lib, engine, image, side)
                    ms[side] = (time.thread_time_ns()-start)/1e6
                start = time.thread_time_ns()
                guided, used = refine(lib, engine, image, boxes[640])
                add = (time.thread_time_ns()-start)/1e6
                data.append((gt,small,[boxes[640],boxes[640],boxes[800],guided],
                             [ms[640],ms[640],ms[800],ms[640]+add],used))
        finally:
            lib.detect_free(engine)
    dev, test = data[:128], data[128:]
    budget = score(dev,0,.70)[0].sum(0)[1]
    thresholds = [.70,.55]
    for mode in (2,3):
        trials = [(t,score(dev,mode,t)[0].sum(0)) for t in np.round(np.arange(.40,.961,.01),2)]
        best = max((r for r in trials if r[1][1]<=budget), key=lambda r:(r[1][2],r[1][0],-abs(r[0]-.7)))
        thresholds.append(best[0])
    for i,t in enumerate(thresholds):
        print("dev",MODES[i],t,score(dev,i,t)[0].sum(0))
    original, ids = score(test,0,.70)
    print("held out GT / small / crop calls",sum(len(x[0]) for x in test),sum(x[1].sum() for x in test),sum(x[4] for x in test))
    bs=np.random.default_rng(60178).integers(len(test),size=(2000,len(test)))
    for i,t in enumerate(thresholds):
        result, after = score(test,i,t)
        delta = result-original
        times=np.array([x[3][i] for x in test])
        gained=sum(len(a-b) for a,b in zip(after,ids))
        lost=sum(len(b-a) for a,b in zip(after,ids))
        print("test",MODES[i],t,result.sum(0),"small recovered/lost", gained,lost,
              "CPU mean/median",np.round([times.mean(),np.median(times)],2),
              "delta interval", np.percentile(delta[bs].sum(1),[2.5,97.5],axis=0).round(1).tolist())


if __name__ == "__main__":
    main()
