"""Prepare the rectangular input used by the native YuNet detector."""
from functools import lru_cache
import math

import numpy as np
from PIL import Image


@lru_cache(maxsize=256)
def rectangle_shape(w, h, *, side=640):
    if min(w, h) <= 0:
        raise ValueError("Image dimensions must be positive")
    if side not in (448, 640, 800):
        raise ValueError("Unsupported detector pixel budget")
    cells = (side // 32) ** 2
    # Width capacity increases with cols and height capacity decreases.
    limit = 2 * side // 32
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi) // 2
        if mid * h < (cells // mid) * w:
            lo = mid + 1
        else:
            hi = mid
    def capacity(cols):
        return min(32 * cols / w, 32 * (cells // cols) / h)
    scale = min(2 * side / max(w, h), max(capacity(lo), capacity(max(1, lo - 1))))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    return rw, rh, (rw + 31) // 32 * 32, (rh + 31) // 32 * 32


def prepare_rectangle(image, *, bgr=False, side=640):
    h, w = image.shape[:2]
    rw, rh, cw, ch = rectangle_shape(w, h, side=side)
    x, y = (cw - rw) // 2, (ch - rh) // 2
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    if (rw, rh) == (w, h) and image.dtype == np.uint8 and image.shape[2:] == (3,):
        pixels = image
    else:
        bicubic = getattr(Image, "Resampling", Image).BICUBIC
        pixels = np.asarray(Image.fromarray(image).resize((rw, rh), bicubic))
    region = canvas[y:y + rh, x:x + rw]
    if bgr and pixels.shape[2:] == (3,):
        for channel in range(3):
            region[:, :, channel] = pixels[:, :, 2 - channel]
    else:
        region[:] = pixels[:, :, ::-1] if bgr else pixels
    return canvas, rw, rh


def guided_crop(faces, w, h):
    centers = []
    for f in faces:
        l, t, r, b = f["bbox"]
        if f["score"] < .85 and math.sqrt(max(0., (r-l)*(b-t)))*320/max(w,h) < 20 and r > l and b > t:
            centers.append(((l+r)/2, (t+b)/2))
    if not centers:
        return None
    if len(centers) > 1:
        points = np.asarray(centers)
        xs, ys = points.T
        best = 0
        # Bounded scratch even for a large caller supplied candidate list.
        for start in range(0, len(points), 64):
            rows = points[start:start + 64]
            counts = ((np.abs(rows[:, :1] - xs) <= w/4) &
                      (np.abs(rows[:, 1:] - ys) <= h/4)).sum(axis=1)
            i = int(np.argmax(counts))
            if counts[i] > best:
                best = counts[i]
                center = centers[start + i]
    else:
        center = centers[0]
    cx, cy = center
    cw, ch = max(1,w//2), max(1,h//2)
    return int(min(max(cx-cw/2,0),w-cw)), int(min(max(cy-ch/2,0),h-ch)), cw, ch


def merge_faces(base, extra, x, y):
    translated = []
    for f in extra:
        l,t,r,b = f["bbox"]
        translated.append(dict(f, bbox=(x+l,y+t,x+r,y+b),
                               keypoints=[(x+kx,y+ky) for kx,ky in f["keypoints"]]))
    kept, regions = [], []
    for f in sorted(base + translated, key=lambda face: -face["score"]):
        l,t,r,b = f["bbox"]
        area = (r-l)*(b-t)
        for pl,pt,pr,pb,previous_area in regions:
            if r <= pl or pr <= l or b <= pt or pb <= t:
                continue
            inter = (min(r,pr)-max(l,pl))*(min(b,pb)-max(t,pt))
            if inter > .4*(area + previous_area - inter):
                break
        else:
            kept.append(f)
            regions.append((l,t,r,b,area))
            if len(kept) == 32:
                break
    return kept
