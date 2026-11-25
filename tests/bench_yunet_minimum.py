"""Paired minimum-side ablation on images excluded from earlier FaceX quality splits."""
import argparse
import ctypes
import io
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
from bench_yunet_bottlenecks import ROOT, annotations, match, native_run, prepare_rectangle, replay, source_boxes

MINIMA = (16, 12, 8)
GRID = np.round(np.arange(.55, .971, .01), 2)


def partition(labels):
    pool = set(labels)
    for seed, n in ((114, 640), (725, 128), (3519, 672), (9101, 384), (60177, 448)):
        group = np.random.default_rng(seed).choice(sorted(pool), n, replace=False)
        pool -= set(group)
    # Reuse previously seen images for calibration only. All remaining images
    # are held out from every previous named development and evaluation split.
    dev = np.random.default_rng(1702).choice(sorted(set(labels)-pool), 256, replace=False)
    return dev, sorted(pool)


def build(tmp, minimum):
    src = subprocess.check_output(["git", "show", "ee217f5:src/detect.c"], cwd=ROOT, text=True)
    if minimum is None:
        a = src.index("                if (bw < 16")
        b = src.index("                candidates[nf].score", a)
        src = src[:a] + "                if (bw <= 0 || bh <= 0) continue;\n" + src[b:]
        a = src.index("    if (nf > 1) nf = nms_sort")
        b = src.index("    if (nf > max_faces)", a)
        src = src[:a] + src[b:]
    else:
        src = src.replace("if (bw < 16 || bh < 16)", f"if (bw < {minimum} || bh < {minimum})")
    file = tmp / f"det{minimum}.c"
    so = file.with_suffix(".so")
    file.write_text(src)
    subprocess.run(["gcc", "-O3", "-march=native", "-fPIC", "-shared", "-I",
                    str(ROOT/"include"), str(file), "-lm", "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.detect_init.restype = ctypes.c_void_p
    return lib


def collect(engines, names, labels, photos, side):
    records = []
    for i, name in enumerate(names):
        image = np.asarray(Image.open(io.BytesIO(photos.read("WIDER_val/images/"+name))).convert("RGB"))
        h,w = image.shape[:2]
        canvas,rw,rh = prepare_rectangle(image,bgr=True,side=side)
        ch,cw = canvas.shape[:2]
        raw = native_run(*engines[None],canvas,capacity=16000)
        dims = (raw[:,2:4]-raw[:,:2]).min(1)
        boxes = []
        for minimum in MINIMA:
            pred = replay(raw[dims >= minimum], "min8", cw, ch)
            if i % 83 == 0:
                np.testing.assert_array_equal(pred, native_run(*engines[minimum],canvas,capacity=32))
            boxes.append(source_boxes(pred,w,h,rw,rh,cw,ch))
        gt = labels[name].copy()
        small = np.sqrt(gt[:,2:].prod(1))*320/max(w,h) < 16
        gt[:,2:] += gt[:,:2]
        records.append((gt, small, boxes))
    return records


def evaluate(data, m, thresholds):
    result = np.zeros((len(data),len(thresholds),3), dtype=int)
    for i,(gt,small,boxes) in enumerate(data):
        cached = {}
        for j,threshold in enumerate(thresholds):
            n = int((boxes[m][:,4] >= np.float32(threshold)).sum())
            if n not in cached:
                good = match(boxes[m][:n],gt)
                cached[n] = [len(good),n-len(good),small[good].sum()]
            result[i,j] = cached[n]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("weights","images","labels"):
        ap.add_argument(key,type=Path)
    ap.add_argument("--side",type=int,choices=(640,800),required=True)
    args = ap.parse_args()
    score = .70 if args.side == 640 else .75
    with ZipFile(args.labels) as z:
        labels = annotations(z)
    dev, test = partition(labels)
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as photos:
        engines = {}
        try:
            for m in (None, *MINIMA):
                lib = build(Path(tmp),m)
                eng = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
                if not eng:
                    raise RuntimeError("Cannot load YuNet")
                lib.detect_set_score_threshold(eng,ctypes.c_float(.50))
                engines[m] = (lib,eng)
            calibration = collect(engines,dev,labels,photos,args.side)
            budget = evaluate(calibration,0,[score]).sum(0)[0,1]
            locked = [score]
            for m in (1,2):
                totals = evaluate(calibration,m,GRID).sum(0)
                ok = np.flatnonzero(totals[:,1] <= budget)
                best = max(ok,key=lambda j:(totals[j,2],totals[j,0],-abs(GRID[j]-score)))
                locked.append(GRID[best])
            print("side, calibration FP budget, locked thresholds",args.side,budget,locked,flush=True)
            holdout = collect(engines,test,labels,photos,args.side)
        finally:
            for lib,eng in engines.values():
                lib.detect_free(eng)
    print("images, valid GT, small GT",len(test),sum(len(x[0]) for x in holdout),sum(x[1].sum() for x in holdout))
    base = evaluate(holdout,0,[score])[:,0]
    bs = np.random.default_rng(1703).integers(len(test),size=(4000,len(test)))
    for m,minimum in enumerate(MINIMA):
        for t in dict.fromkeys([score, locked[m]]):
            counts = evaluate(holdout,m,[t])[:,0]
            ci = np.percentile((counts-base)[bs].sum(1), [2.5,97.5], axis=0)
            print(minimum, t, counts.sum(0).tolist(), "delta CI", ci.round(1).tolist())


if __name__ == "__main__":
    main()
