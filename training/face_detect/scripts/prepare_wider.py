"""
Download and prepare WIDER FACE for training.

WIDER FACE annotations include bounding boxes only (no landmarks).
For 5-point face landmarks we use the RetinaFace-style supplementary
annotations from https://github.com/biubug6/Pytorch_Retinaface (Apache 2.0).

Files we need:
  WIDER_train.zip          1.4 GB  → 12880 training images
  WIDER_val.zip            350 MB  → 3226 val images
  wider_face_train_bbx_gt.txt  (official bbox annotations)
  retinaface_train_label.txt   (with 5 landmarks per face)

We'll skip the 1.6GB test set (no labels public anyway).
"""

import os, sys, urllib.request, ssl, zipfile, hashlib
from pathlib import Path

OUT = Path("D:/apps/facex/training/data/widerface")
OUT.mkdir(parents=True, exist_ok=True)


# Hugging Face mirrors are reliable and don't need Google Drive auth.
URLS = {
    "WIDER_train.zip":
        "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
    "WIDER_val.zip":
        "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",
    # Official annotations
    "wider_face_split.zip":
        "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip",
    # RetinaFace-style 5-landmark annotations (BSD-3 license, used by many)
    "retinaface_gt_v1.1.zip":
        "https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/data/retinaface_gt_v1.1.zip",
}


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  ✓ {dest.name} ({dest.stat().st_size / 1024**2:.1f} MB) already present")
        return
    print(f"  ↓ downloading {dest.name} from {url}")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            buf = r.read(1 << 20)
            if not buf: break
            f.write(buf); got += len(buf)
            if total:
                print(f"    {got/1024**2:.0f}/{total/1024**2:.0f} MB", end="\r", flush=True)
        print()


def extract(zip_path, dest_dir):
    if not zip_path.exists(): return
    print(f"  → unzipping {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def main():
    print(f"target dir: {OUT}")
    for name, url in URLS.items():
        try:
            download(url, OUT / name)
        except Exception as e:
            print(f"  ! {name}: {e}")

    # Unzip the smaller ones
    extract(OUT / "wider_face_split.zip", OUT)
    extract(OUT / "retinaface_gt_v1.1.zip", OUT)
    # Large zips: extract only if not already extracted
    if not (OUT / "WIDER_train" / "images").exists():
        extract(OUT / "WIDER_train.zip", OUT)
    if not (OUT / "WIDER_val" / "images").exists():
        extract(OUT / "WIDER_val.zip", OUT)

    # Summary
    for sub in ("WIDER_train", "WIDER_val", "wider_face_split"):
        p = OUT / sub
        if p.exists():
            n = sum(1 for _ in p.rglob("*.jpg"))
            print(f"  ✓ {sub}: {n} jpg files")

    rf = OUT / "retinaface_gt_v1.1"
    if rf.exists():
        for f in rf.rglob("label.txt"):
            lines = sum(1 for _ in open(f))
            print(f"  ✓ {f.relative_to(OUT)}: {lines} lines")


if __name__ == "__main__":
    main()
