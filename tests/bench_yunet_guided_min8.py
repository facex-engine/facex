"""Recheck guided refinement with the eight pixel minimum and the optimized detector."""
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
from bench_yunet_detail import annotations, detect, refine, score
from verify_yunet import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "images", "labels"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with ZipFile(args.labels) as z:
        labels = annotations(z)
    pool = set(labels)
    for seed, n in ((114,640),(725,128),(3519,672),(9101,384)):
        pool -= set(np.random.default_rng(seed).choice(sorted(pool),n,replace=False))
    names = np.random.default_rng(60177).choice(sorted(pool),448,replace=False)
    data = []
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as photos:
        so = Path(tmp)/"native.so"
        subprocess.run(["gcc","-O3","-march=native","-fPIC","-shared","-I",str(ROOT/"include"),
                        str(ROOT/"src/detect.c"),"-lm","-o",str(so)],check=True)
        lib = ctypes.CDLL(str(so))
        lib.detect_init.restype = ctypes.c_void_p
        e = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
        if not e:
            raise RuntimeError("Cannot load detector")
        lib.detect_set_min_box_side(e,ctypes.c_int(8))
        lib.detect_set_score_threshold(e,ctypes.c_float(.35))
        try:
            for name in names:
                rgb = np.asarray(Image.open(io.BytesIO(photos.read("WIDER_val/images/"+name))).convert("RGB"))
                gt = labels[name].copy()
                small = np.sqrt(gt[:,2:].prod(1))*320/max(rgb.shape[:2]) < 16
                gt[:,2:] += gt[:,:2]
                t0 = time.thread_time_ns()
                base = detect(lib,e,rgb,640)
                t1 = time.thread_time_ns()
                guided,used = refine(lib,e,rgb,base)
                t2 = time.thread_time_ns()
                data.append((gt,small,[base,guided],[(t1-t0)/1e6,(t2-t0)/1e6],used))
        finally:
            lib.detect_free(e)
    dev,test = data[:128],data[128:]
    budget = score(dev,0,.75)[0].sum(0)[1]
    options = [(t,score(dev,1,t)[0].sum(0)) for t in np.round(np.arange(.50,.981,.01),2)]
    threshold,_ = max((o for o in options if o[1][1]<=budget),key=lambda o:(o[1][2],o[1][0]))
    baseline = score(test,0,.75)[0]
    guided = score(test,1,threshold)[0]
    bs = np.random.default_rng(315).integers(len(test),size=(2000,len(test)))
    print("threshold",threshold,"baseline vs guided TP FP small",baseline.sum(0),guided.sum(0))
    print("mean CPU ms",np.mean([x[3] for x in test],axis=0),"cropped",sum(x[4] for x in test))
    print("paired delta CI",np.percentile((guided-baseline)[bs].sum(1),[2.5,97.5],axis=0))


if __name__ == "__main__":
    main()
