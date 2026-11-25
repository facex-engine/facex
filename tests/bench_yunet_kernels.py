"""Compare all returned YuNet coordinates and end-to-end detector CPU on AVX2/FMA."""
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

from bench_yunet_bottlenecks import ROOT, load_lib, native_run, prepare_rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("images", type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with tempfile.TemporaryDirectory() as work, ZipFile(args.images) as photos:
        tmp = Path(work)
        new = tmp / "current.so"
        subprocess.run(["gcc", "-O3", "-march=native", "-I", str(ROOT/"include"), "-fPIC",
                        "-shared", str(ROOT/"src/detect.c"), "-lm", "-o", str(new)], check=True)
        libs = [load_lib(tmp, "reference"), ctypes.CDLL(str(new))]
        engines = []
        for lib in libs:
            lib.detect_init.restype = ctypes.c_void_p
            eng = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
            if not eng:
                raise RuntimeError("Failed to load detector")
            engines.append(eng)
            lib.detect_set_score_threshold(eng, ctypes.c_float(.40))
        times = np.zeros((64, 2, 2))
        names = np.random.default_rng(781).choice(sorted(n for n in photos.namelist() if n.endswith(".jpg")),65,replace=False)
        try:
            for i, name in enumerate(names):
                rgb = np.asarray(Image.open(io.BytesIO(photos.read(name))).convert("RGB"))
                for j, side in enumerate((640, 800)):
                    pred = {}
                    for k in (i%2, 1-i%2):
                        start = time.thread_time_ns()
                        canvas, _, _ = prepare_rectangle(rgb, bgr=True, side=side)
                        pred[k] = native_run(libs[k], engines[k], canvas, capacity=10000)
                        if i:
                            times[i-1,j,k] = (time.thread_time_ns()-start)/1e6
                    np.testing.assert_array_equal(pred[0], pred[1])
            print("CPU ms, rows 640 and 800, columns reference and current")
            print("median", np.median(times,axis=0), "mean", times.mean(0))
        finally:
            for lib, eng in zip(libs, engines):
                lib.detect_free(eng)


if __name__ == "__main__":
    main()
