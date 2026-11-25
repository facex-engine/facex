"""Evaluate refresh on annotated OpenCV tracking clips and controlled ID reuse."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from facex import FaceX
from facex.video import VideoEmbeddings, _iou


def unit(x):
    return x / max(1e-12, np.linalg.norm(x))


def crossing_check(engine, examples):
    # Replay two annotated source crops on converging paths with a known ID swap.
    patches = []
    for rgb, face in examples[:2]:
        x1, y1, x2, y2 = np.rint(face["bbox"]).astype(int)
        patch = Image.fromarray(rgb[y1:y2, x1:x2]).resize((96,96))
        pts = (np.asarray(face["keypoints"]) - [x1,y1]) * [96/(x2-x1),96/(y2-y1)]
        patches.append((patch,pts))
    cache = VideoEmbeddings(engine)
    wrong = 0
    refs = []
    for frame, (left,right) in enumerate(((8,184),(14,178),(36,156),(62,130),
                                          (82,110),(110,82),(130,62),(156,36),(178,14))):
        canvas = Image.new("RGB",(288,120))
        tracks = []
        for person, x in enumerate((left,right)):
            patch, pts = patches[person]
            canvas.paste(patch,(x,10))
            tracks.append(dict(track_id=person if frame<5 else 1-person,
                               bbox=(x,10,x+96,106),score=.99,
                               keypoints=(pts+[x,10]).tolist()))
        rgb = np.asarray(canvas)
        out = cache.update(rgb,frame,tracks)
        if not refs:
            refs = [unit(engine.embed_landmarks(rgb,f["keypoints"])) for f in tracks]
        for f,t in zip(out,tracks):
            oracle = engine.embed_landmarks(rgb,t["keypoints"])
            if frame == 5:
                assert f["refreshed"]
                np.testing.assert_allclose(oracle,f["embedding"],atol=1e-5)
            wrong += int(np.argmax(refs @ unit(oracle)) != np.argmax(refs @ unit(f["embedding"])))
    print("controlled crossing, gallery disagreements over 18 comparisons:",wrong)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--frames",type=int,default=160)
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()
    engine = FaceX(input_policy="rectangle_bgr")
    examples, galleries, samples = [], [], []
    for name in ("david", "dudek", "faceocc2"):
        folder = args.directory/name
        vid = folder/(name+".webm")
        gt = np.loadtxt(folder/"gt.txt",delimiter=",",ndmin=2)
        info = json.loads(subprocess.check_output(["ffprobe","-v","quiet","-show_streams",
                                                   "-of","json",str(vid)]))["streams"][0]
        w,h = info["width"],info["height"]
        cache = VideoEmbeddings(engine)
        first, record = None, []
        # The David annotations begin at source frame 300 (one-based).
        offset = 299 if name == "david" else 0
        with subprocess.Popen(["ffmpeg","-loglevel","error","-threads","1","-i",str(vid),
                               "-frames:v",str(offset+args.start+args.frames),"-f","rawvideo","-pix_fmt","rgb24","-threads","1","-"],
                              stdout=subprocess.PIPE) as proc:
            for source_frame in range(offset+min(args.start+args.frames,len(gt))):
                t = source_frame-offset
                raw = proc.stdout.read(w*h*3)
                if not raw:
                    break
                if t < args.start:
                    continue
                rgb = np.frombuffer(raw,dtype="u1").reshape(h,w,3)
                faces = engine.detect(rgb,embeddings=False)
                x,y,bw,bh = gt[t,:4]
                box = (x,y,x+bw,y+bh)
                face = max(faces,key=lambda f:_iou(box,f["bbox"]),default=None)
                tracks = [dict(face,track_id=1)] if face and _iou(box,face["bbox"])>=.20 else []
                out = cache.update(rgb,t,tracks)
                if out:
                    ref = engine.embed_landmarks(rgb,face["keypoints"])
                    if first is None:
                        first = unit(ref)
                        examples.append((rgb.copy(),tracks[0]))
                    record.append((t,out[0]["refreshed"],unit(ref),unit(out[0]["embedding"])))
            if proc.wait() != 0:
                raise RuntimeError("Video decoding failed")
        if first is None:
            raise RuntimeError("No target face matched in "+name)
        galleries.append(first)
        samples.append(record)
    gallery = np.stack(galleries)
    for idx,rec in enumerate(samples):
        # Exclude the enrollment frame from the nearest-template comparison.
        ids = [(np.argmax(gallery@r),np.argmax(gallery@c)) for _,_,r,c in rec[1:]]
        print(idx,"matched frames, refreshes, first frame",len(rec),sum(r[1] for r in rec),rec[0][0],
              "reference/cached wrong",sum(a!=idx for a,b in ids),sum(b!=idx for a,b in ids),
              "label disagreements",sum(a!=b for a,b in ids),flush=True)
    cache = VideoEmbeddings(engine)
    img,one = examples[0]
    cache.update(img,0,[one])
    cache.update(img,1,[one])
    assert all(f["refreshed"] for f in cache.update(img,2,[one,dict(one,track_id=2)]))
    assert cache.update(img,3,[one])[0]["refreshed"]
    cache.update(img,4,[])
    img,two = examples[1]
    after = cache.update(img,5,[two])[0]
    assert after["refreshed"]
    np.testing.assert_allclose(after["embedding"],engine.embed_landmarks(img,two["keypoints"]),atol=1e-5)
    print("ambiguity, separation and ID reuse after loss refreshed correctly")
    crossing_check(engine,examples)


if __name__ == "__main__":
    main()
