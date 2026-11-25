"""Paired end-to-end timings and exact geometry checks for the pooling and head changes."""
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

from bench_yunet_bottlenecks import ROOT, native_run, prepare_rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("images", type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    base = subprocess.check_output(["git","show","6192e8e:src/detect.c"],cwd=ROOT,text=True)
    current = (ROOT/"src/detect.c").read_text()
    marker = "    /* === HEADS + DECODE === */"
    pool_only = current.split(marker)[0] + marker + base.split(marker)[1]
    engines = []
    with tempfile.TemporaryDirectory() as tmp, ZipFile(args.images) as photos:
        try:
            for i, src in enumerate((base,pool_only,current)):
                path = Path(tmp)/f"v{i}.c"
                path.write_text(src)
                libpath = path.with_suffix(".so")
                subprocess.run(["gcc","-O3","-march=x86-64-v3","-shared","-fPIC","-I",
                                str(ROOT/"include"),str(path),"-lm","-o",str(libpath)],check=True)
                lib = ctypes.CDLL(str(libpath))
                lib.detect_init.restype=ctypes.c_void_p
                eng = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
                if not eng:
                    raise RuntimeError("Failed to load weights")
                engines.append((lib,eng))
            names=np.random.default_rng(597).choice(sorted(n for n in photos.namelist() if n.endswith(".jpg")),193,replace=False)
            ts=np.zeros((192,3))
            for i,name in enumerate(names):
                rgb=np.asarray(Image.open(io.BytesIO(photos.read(name))).convert("RGB"))
                canvas,*_=prepare_rectangle(rgb,bgr=True,side=640 if i%2 else 800)
                pred={}
                for k in np.roll(np.arange(3),i%3):
                    lib,eng=engines[k]
                    lib.detect_set_score_threshold(eng,ctypes.c_float(.35 if i%3==0 else .75))
                    start=time.thread_time_ns()
                    pred[k]=native_run(lib,eng,canvas,capacity=10000)
                    if i:
                        ts[i-1,k]=(time.thread_time_ns()-start)/1e6
                for k in (1,2):
                    np.testing.assert_array_equal(pred[k],pred[0])
            for j,side in enumerate((640,800)):
                t=ts[j::2]
                print(side,"mean, median CPU ms",np.round(t.mean(0),2),np.round(np.median(t,0),2))
        finally:
            for lib,eng in engines:
                lib.detect_free(eng)


if __name__ == "__main__":
    main()
