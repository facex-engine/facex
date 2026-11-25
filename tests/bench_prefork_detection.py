"""Measure process-isolated WSGI inference using Gunicorn's lazy and prefork layouts."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import select
import time

import numpy as np
from PIL import Image

from server import app, initialize_detector


def memory(pid):
    with open(f"/proc/{pid}/smaps_rollup") as src:
        return next(int(line.split()[1]) for line in src if line.startswith("Pss:"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workers", type=int, choices=(1,2,4))
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0,set(cpus))
    if args.preload:
        initialize_detector()
    root = Path(__file__).resolve().parents[1]
    images = []
    for path in sorted((root/"tests/golden").glob("*.npz"))[:8]:
        with np.load(path) as f:
            buf=io.BytesIO()
            Image.fromarray(f["image"]).save(buf,format="JPEG")
            images.append(buf.getvalue())
    workers=[]
    for i in range(args.workers):
        rd,wr=os.pipe()
        rd2,wr2=os.pipe()
        pid=os.fork()
        if pid == 0:
            os.close(wr);os.close(rd2)
            try:
                c=app.test_client()
                def run(j):
                    res=c.post("/detect",data={"file":(io.BytesIO(images[j]),"input.jpg")})
                    if res.status_code != 200:
                        raise RuntimeError(res.data)
                    return hashlib.sha256(json.dumps(res.json["faces"],sort_keys=True).encode()).hexdigest()
                run(i%len(images))
                os.write(wr2,b"R")
                while True:
                    msg=os.read(rd,1)
                    if msg==b"Q" or not msg:
                        break
                    os.write(wr2,run(msg[0]).encode())
                os._exit(0)
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(1)
        os.close(rd);os.close(wr2)
        workers.append((pid,wr,rd2))
    try:
        for _,_,r in workers:
            assert os.read(r,1)==b"R"
        pss=sum(memory(pid) for pid,_,_ in workers)+memory(os.getpid())
        answers={};elapsed=[]
        start=time.monotonic()
        for k in range(0,64,args.workers):
            tick=time.monotonic()
            waiting={}
            for j,(_,w,r) in enumerate(workers):
                idx=(k+j)%len(images)
                os.write(w,bytes([idx]));waiting[r]=idx
            while waiting:
                readable,_,_=select.select(list(waiting),[],[],30)
                if not readable: raise RuntimeError("Timed out")
                for r in readable:
                    val=os.read(r,64)
                    assert len(val)==64
                    i=waiting.pop(r)
                    assert answers.setdefault(i,val)==val
                    elapsed.append((time.monotonic()-tick)*1000)
        total=time.monotonic()-start
        print(args.workers,args.preload,"req/s",round(64/total,2),"p95 ms",round(np.percentile(elapsed,95),2),
              "PSS MiB",round(pss/1024,1),"checksum",hashlib.sha256(b"".join(answers[i] for i in sorted(answers))).hexdigest())
    finally:
        for pid,w,r in workers:
            try: os.write(w,b"Q")
            except BrokenPipeError: pass
            os.close(w);os.close(r);os.waitpid(pid,0)


if __name__ == "__main__":
    main()
