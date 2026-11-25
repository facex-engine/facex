"""Prepare the rectangular input used by the native YuNet detector."""
import numpy as np
from PIL import Image


def rectangle_shape(w, h, *, side=640):
    if min(w, h) <= 0:
        raise ValueError("Image dimensions must be positive")
    if side not in (448, 640, 800):
        raise ValueError("Unsupported detector pixel budget")
    cells = (side // 32) ** 2
    scale = min(2 * side / max(w, h), max(
        min(32 * cols / w, 32 * (cells // cols) / h) for cols in range(1, 2 * side // 32 + 1)))
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
        if np.sqrt((r-l)*(b-t))*320/max(w,h) < 20 and f["score"] < .85:
            centers.append(((l+r)/2, (t+b)/2))
    if not centers:
        return None
    centers = np.asarray(centers)
    delta = np.abs(centers[:, None] - centers)
    counts = ((delta[:,:,0] <= w/4) & (delta[:,:,1] <= h/4)).sum(1)
    cx, cy = centers[np.argmax(counts)]
    cw, ch = max(1,w//2), max(1,h//2)
    return int(np.clip(cx-cw/2,0,w-cw)), int(np.clip(cy-ch/2,0,h-ch)), cw, ch


def merge_faces(base, extra, x, y):
    for f in extra:
        l,t,r,b = f["bbox"]
        f["bbox"] = (x+l,y+t,x+r,y+b)
        f["keypoints"] = [(x+kx,y+ky) for kx,ky in f["keypoints"]]
    kept = []
    for f in sorted(base + extra, key=lambda face: -face["score"]):
        l,t,r,b = f["bbox"]
        for prior in kept:
            pl,pt,pr,pb = prior["bbox"]
            inter = max(0,min(r,pr)-max(l,pl))*max(0,min(b,pb)-max(t,pt))
            if inter > .4*((r-l)*(b-t)+(pr-pl)*(pb-pt)-inter):
                break
        else:
            kept.append(f)
            if len(kept) == 32:
                break
    return kept
