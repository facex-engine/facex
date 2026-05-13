"""Scrape tongue-out face photos from the web.

Strategy: DDGS image search for several variant queries, dedup by URL,
download with browser-like User-Agent and timeout, then crop to faces
with YuNet so we get clean 112x112 inputs.
"""

import os, time, hashlib, urllib.request, ssl
from pathlib import Path

import cv2
import numpy as np
from ddgs import DDGS

OUT = Path("D:/apps/facex/training/data/tongue/raw_pos")
OUT.mkdir(parents=True, exist_ok=True)
CROP_OUT = Path("D:/apps/facex/training/data/tongue/faces_pos")
CROP_OUT.mkdir(parents=True, exist_ok=True)

YUNET = "D:/apps/facex/_unencrypted_backup/yunet_2023mar.onnx"

QUERIES = [
    "selfie tongue out",
    "person sticking tongue out",
    "tongue out face photo",
    "tongue out portrait",
    "kid tongue out",
    "girl tongue out selfie",
    "man tongue out funny",
    "tongue out face close up",
    "high five tongue out",
    "playful tongue out",
    "child sticking tongue out",
    "adult tongue out face",
    "happy tongue out",
    "rock concert tongue out",
    "tongue sticking out close",
    "miley cyrus tongue",
    "albert einstein tongue",
    "musician tongue out stage",
    "athlete tongue out celebration",
    "actress tongue out red carpet",
    "kawaii anime girl tongue out",
    "boy tongue out funny face",
    "tongue out kiss face",
    "tongue out wink",
    "summer tongue out outdoor",
]

MAX_PER_QUERY = 200
TARGET_TOTAL = 1500

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
ctx = ssl.create_default_context()


def fetch(url, dst, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            data = r.read(5 * 1024 * 1024)   # cap 5 MB
        if len(data) < 2048: return False
        dst.write_bytes(data)
        return True
    except Exception:
        return False


def crop_face(src_path, dst_path, det):
    img = cv2.imread(str(src_path))
    if img is None: return False
    h, w = img.shape[:2]
    if min(h, w) < 96: return False
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0: return False
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = best[:4]
    if fw < 60 or fh < 60: return False
    cx, cy = x + fw / 2, y + fh / 2
    side = max(fw, fh) * 1.3
    sx = int(max(0, cx - side / 2)); sy = int(max(0, cy - side / 2))
    ex = int(min(w, cx + side / 2)); ey = int(min(h, cy + side / 2))
    crop = img[sy:ey, sx:ex]
    if crop.shape[0] < 64 or crop.shape[1] < 64: return False
    crop = cv2.resize(crop, (112, 112))
    cv2.imwrite(str(dst_path), crop)
    return True


def main():
    seen = set()
    downloaded = 0

    with DDGS() as ddgs:
        for q in QUERIES:
            print(f"\n# {q}")
            try:
                results = ddgs.images(q, max_results=MAX_PER_QUERY)
            except Exception as e:
                print(f"  ddgs failed: {e}"); continue
            for r in results:
                if downloaded >= TARGET_TOTAL: break
                url = r.get("image")
                if not url or url in seen: continue
                seen.add(url)
                ext = os.path.splitext(url.split("?")[0])[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"
                name = hashlib.md5(url.encode()).hexdigest()[:16] + ext
                dst = OUT / name
                if dst.exists(): continue
                if fetch(url, dst):
                    downloaded += 1
                    if downloaded % 25 == 0:
                        print(f"  saved {downloaded} so far")
            if downloaded >= TARGET_TOTAL: break

    print(f"\nraw saved: {downloaded}")

    # Now face-crop
    det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), score_threshold=0.6)
    cropped = 0
    for src in sorted(OUT.glob("*")):
        if src.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"): continue
        dst = CROP_OUT / (src.stem + ".jpg")
        if dst.exists(): continue
        if crop_face(src, dst, det): cropped += 1
    print(f"face-cropped: {cropped} → {CROP_OUT}")


if __name__ == "__main__":
    main()
