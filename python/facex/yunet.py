"""Prepare the rectangular input used by the native YuNet detector."""
import numpy as np
from PIL import Image


def rectangle_shape(w, h):
    if min(w, h) <= 0:
        raise ValueError("Image dimensions must be positive")
    # Each cell is 32 x 32. A 640 x 640 canvas fits 400 cells.
    scale = min(1280 / max(w, h), max(
        min(32 * cols / w, 32 * (400 // cols) / h) for cols in range(1, 41)))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    return rw, rh, (rw + 31) // 32 * 32, (rh + 31) // 32 * 32


def prepare_rectangle(image):
    h, w = image.shape[:2]
    rw, rh, cw, ch = rectangle_shape(w, h)
    x, y = (cw - rw) // 2, (ch - rh) // 2
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    canvas[y:y + rh, x:x + rw] = Image.fromarray(image).resize((rw, rh), bicubic)
    return canvas, rw, rh
