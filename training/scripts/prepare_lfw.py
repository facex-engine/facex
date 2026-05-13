"""
Prepare LFW for InsightFace-style verification eval.

Steps:
  1. Download lfw-deepfunneled.tgz (~111 MB) if missing
  2. Read D:/apps/facex/lfw_pairs.txt
  3. For each image referenced by a pair: center-crop to 168x168 then
     resize to 112x112 (deep-funneled images are already roughly aligned)
  4. JPEG-encode and pickle the (bins, issame) tuple to lfw.bin

The .bin format matches InsightFace's verifier (used by lfw_eval.py).
"""

import io
import pickle
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

# UMass `vis-www.cs.umass.edu` was the original host but its DNS went away.
# Try several known mirrors in order; first one that works wins.
LFW_URLS = [
    "https://web.archive.org/web/2023/https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
    "https://huggingface.co/datasets/atalaydenknalbant/lfw-deepfunneled/resolve/main/lfw-deepfunneled.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",  # in case DNS recovers
]
DATA_DIR = Path("D:/apps/facex/training/data")
LFW_TGZ = DATA_DIR / "lfw-deepfunneled.tgz"
LFW_DIR = DATA_DIR / "lfw-deepfunneled"
LFW_BIN = DATA_DIR / "lfw.bin"
PAIRS_TXT = Path("D:/apps/facex/lfw_pairs.txt")


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LFW_TGZ.exists() and LFW_TGZ.stat().st_size > 50_000_000:
        print(f"Already downloaded: {LFW_TGZ} ({LFW_TGZ.stat().st_size/1e6:.1f} MB)")
        return
    last_err = None
    def hook(blocks, bs, total):
        if blocks % 200 == 0:
            done = blocks * bs / 1e6
            print(f"  {done:>6.1f} / {total/1e6:.1f} MB")
    for url in LFW_URLS:
        try:
            print(f"Trying: {url}")
            urlretrieve(url, LFW_TGZ, reporthook=hook)
            print("Done.")
            return
        except Exception as e:
            print(f"  failed: {e}")
            last_err = e
            try: LFW_TGZ.unlink()
            except FileNotFoundError: pass
    raise SystemExit(f"All LFW mirrors failed. Last error: {last_err}")


def extract():
    if LFW_DIR.exists() and any(LFW_DIR.iterdir()):
        print(f"Already extracted: {LFW_DIR}")
        return
    print(f"Extracting {LFW_TGZ}...")
    with tarfile.open(LFW_TGZ, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    print("Done.")


def find_image(name: str, idx: int) -> Path:
    return LFW_DIR / name / f"{name}_{idx:04d}.jpg"


def crop_resize(img: Image.Image, out: int = 112) -> Image.Image:
    """deep-funneled is 250x250 with face roughly centered. Crop to 168 then resize."""
    W, H = img.size
    box = 168
    left = (W - box) // 2
    top = (H - box) // 2
    img = img.crop((left, top, left + box, top + box))
    return img.resize((out, out), Image.BILINEAR)


def jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def build():
    if LFW_BIN.exists():
        print(f"Already built: {LFW_BIN}")
        return
    print(f"Reading pairs: {PAIRS_TXT}")
    lines = PAIRS_TXT.read_text().strip().split("\n")
    n_folds, n_per_fold = map(int, lines[0].split())
    pairs = lines[1:]
    expected = n_folds * n_per_fold * 2
    assert len(pairs) == expected, f"got {len(pairs)} pairs, expected {expected}"

    bins, issame = [], []
    missing = 0
    for fold in range(n_folds):
        offset = fold * n_per_fold * 2
        for k in range(n_per_fold):  # same pairs
            parts = pairs[offset + k].split()
            name, i, j = parts[0], int(parts[1]), int(parts[2])
            p1, p2 = find_image(name, i), find_image(name, j)
            if not p1.exists() or not p2.exists():
                missing += 1; continue
            bins.append(jpeg_bytes(crop_resize(Image.open(p1).convert("RGB"))))
            bins.append(jpeg_bytes(crop_resize(Image.open(p2).convert("RGB"))))
            issame.append(True)
        for k in range(n_per_fold):  # diff pairs
            parts = pairs[offset + n_per_fold + k].split()
            n1, i, n2, j = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1, p2 = find_image(n1, i), find_image(n2, j)
            if not p1.exists() or not p2.exists():
                missing += 1; continue
            bins.append(jpeg_bytes(crop_resize(Image.open(p1).convert("RGB"))))
            bins.append(jpeg_bytes(crop_resize(Image.open(p2).convert("RGB"))))
            issame.append(False)
    print(f"Built {len(issame)} pairs ({missing} missing)")

    with open(LFW_BIN, "wb") as f:
        pickle.dump((bins, issame), f, protocol=4)
    print(f"Wrote {LFW_BIN}  ({LFW_BIN.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    download()
    extract()
    build()
