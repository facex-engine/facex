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
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    pixels = np.asarray(Image.fromarray(image).resize((rw, rh), bicubic))
    canvas[y:y + rh, x:x + rw] = pixels[:, :, ::-1] if bgr else pixels
    return canvas, rw, rh
