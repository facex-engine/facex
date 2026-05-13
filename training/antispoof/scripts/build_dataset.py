"""
Build anti-spoof training data via composite SCENE rendering.

v3: face is placed inside a 112×112 scene that also contains background
context (room, hand, phone bezel for screen attacks, paper edge for print
attacks). This matches what the browser actually captures when we use a
WIDER crop (side * 2.5 instead of 1.3) — the model now learns the geometry
of replay attacks (rectangular bezel + specular highlight + hand on edge)
in addition to the texture artifacts of the printed/displayed face itself.

Output:
  D:/apps/facex/training/data/antispoof/
    train_raw.bin    (N images × 112×112×3 uint8)
    train_labels.bin (N int8, 0=spoof, 1=real)
    meta.json
"""

import io
import json
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

MS1M = Path("D:/apps/facex/training/data/ms1m")
OUT = Path("D:/apps/facex/training/data/antispoof")
SIZE = 112
N_REAL = 30_000
N_PRINT = 15_000
N_SCREEN = 15_000

MS1M_INDEX = struct.Struct("<QII")


# ===================== building blocks =====================


def synth_bg(rng, palette: str = "room") -> np.ndarray:
    """Fake natural background — blurred low-frequency color blobs."""
    if palette == "room":
        base = np.array([rng.uniform(120, 210), rng.uniform(115, 195), rng.uniform(105, 185)])
    elif palette == "dark":
        base = np.array([rng.uniform(25, 85), rng.uniform(25, 85), rng.uniform(25, 85)])
    elif palette == "office":
        base = np.array([rng.uniform(180, 230), rng.uniform(180, 220), rng.uniform(180, 220)])
    else:
        base = np.array([rng.uniform(80, 200)] * 3)
    bg = np.full((SIZE, SIZE, 3), base[None, None], dtype=np.float32)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.uniform(-20, SIZE + 20), rng.uniform(-20, SIZE + 20)
        r = rng.uniform(25, 70)
        col = np.clip(base + rng.uniform(-40, 40, 3), 0, 255)
        m = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * r ** 2))
        bg += (col[None, None] - bg) * m[..., None] * float(rng.uniform(0.3, 0.7))
    bg = np.clip(bg, 0, 255)
    bg = np.asarray(Image.fromarray(bg.astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(float(rng.uniform(2.0, 7.0)))),
                    dtype=np.float32)
    return bg


def add_hand_blobs(canvas: np.ndarray, rng, n: int = None) -> None:
    """Paint skin-toned blobs at random edges to simulate a hand holding object."""
    if n is None: n = int(rng.integers(1, 4))
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    skin_palettes = [
        (230, 195, 165), (215, 175, 140), (190, 145, 110),
        (165, 120, 90),  (135, 95, 70),
    ]
    for _ in range(n):
        edge = rng.choice(["L", "R", "B", "T"])
        if edge == "L":
            cx, cy = rng.uniform(-15, 30), rng.uniform(20, SIZE - 20)
        elif edge == "R":
            cx, cy = rng.uniform(SIZE - 30, SIZE + 15), rng.uniform(20, SIZE - 20)
        elif edge == "B":
            cx, cy = rng.uniform(15, SIZE - 15), rng.uniform(SIZE - 20, SIZE + 15)
        else:
            cx, cy = rng.uniform(15, SIZE - 15), rng.uniform(-15, 30)
        rx = rng.uniform(20, 45); ry = rng.uniform(20, 45)
        sp = skin_palettes[int(rng.integers(0, len(skin_palettes)))]
        col = np.array(sp, dtype=np.float32) + rng.uniform(-10, 10, 3)
        m = np.exp(-((xs - cx) ** 2 / (2 * rx ** 2) + (ys - cy) ** 2 / (2 * ry ** 2)))
        strength = float(rng.uniform(0.6, 0.95))
        canvas[:] = canvas * (1 - m[..., None] * strength) + col[None, None] * (m[..., None] * strength)


def paste_resize(face: np.ndarray, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                 alpha: np.ndarray = None) -> None:
    """Resize face → w×h, paste at (x,y) on canvas (in place)."""
    f = np.asarray(Image.fromarray(face).resize((w, h), Image.BILINEAR), dtype=np.float32)
    x1, y1 = x, y; x2, y2 = x + w, y + h
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = w - max(0, x2 - SIZE); sy2 = h - max(0, y2 - SIZE)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(SIZE, x2), min(SIZE, y2)
    if dx2 <= dx1 or dy2 <= dy1: return
    if alpha is None:
        canvas[dy1:dy2, dx1:dx2] = f[sy1:sy2, sx1:sx2]
    else:
        a = alpha[sy1:sy2, sx1:sx2, None]
        canvas[dy1:dy2, dx1:dx2] = (canvas[dy1:dy2, dx1:dx2] * (1 - a) +
                                     f[sy1:sy2, sx1:sx2] * a)


def webcam_capture(arr: np.ndarray, rng) -> np.ndarray:
    """Apply realistic webcam-capture artifacts to whole scene."""
    pil = Image.fromarray(arr.astype(np.uint8))
    if rng.random() < 0.85:
        q = int(rng.integers(35, 92))
        buf = io.BytesIO(); pil.save(buf, "JPEG", quality=q); buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    if rng.random() < 0.5:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.6))))
    out = np.asarray(pil, dtype=np.float32)
    if rng.random() < 0.9:
        out += rng.standard_normal((SIZE, SIZE, 3)) * float(rng.uniform(1.5, 5.5))
    if rng.random() < 0.8:
        out += np.array([rng.uniform(-6, 6)] * 3, dtype=np.float32)[None, None]
    if rng.random() < 0.7:
        out = (out - 127.5) * float(rng.uniform(0.85, 1.15)) + 127.5 + float(rng.uniform(-12, 12))
    return np.clip(out, 0, 255).astype(np.uint8)


# ===================== face-only artifact augs (texture) =====================


def print_artifact(img: np.ndarray, rng) -> np.ndarray:
    """Texture-level print artifacts applied to the face itself."""
    pil = Image.fromarray(img)
    s = float(rng.uniform(0.35, 0.65))
    side = max(1, int(img.shape[0] * s))
    pil = pil.resize((side, side), Image.BILINEAR).resize(img.shape[:2][::-1], Image.BILINEAR)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
    arr = np.asarray(pil, dtype=np.float32)
    h, w = arr.shape[:2]
    freq = float(rng.uniform(0.25, 0.5))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    arr += ((np.sin(ys * freq + rng.uniform(0, 6.28)) +
             np.sin(xs * freq + rng.uniform(0, 6.28))) *
            float(rng.uniform(2, 6)))[..., None]
    mean = arr.mean(axis=2, keepdims=True)
    arr = mean + (arr - mean) * float(rng.uniform(0.6, 0.85))
    arr += np.array([rng.uniform(2, 8), rng.uniform(0, 4), rng.uniform(-4, 0)],
                    dtype=np.float32)[None, None]
    arr += rng.standard_normal((h, w, 1)) * float(rng.uniform(2, 5))
    return np.clip(arr, 0, 255).astype(np.uint8)


def screen_artifact(img: np.ndarray, rng) -> np.ndarray:
    """Texture-level screen artifacts applied to the face itself."""
    pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(
        radius=float(rng.uniform(0.2, 0.7))))
    arr = np.asarray(pil, dtype=np.float32)
    h, w = arr.shape[:2]
    angle = float(rng.uniform(0, 3.14))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    uu = xs * np.cos(angle) - ys * np.sin(angle)
    vv = xs * np.sin(angle) + ys * np.cos(angle)
    arr += (np.sin(uu * float(rng.uniform(0.6, 1.3))) *
            np.cos(vv * float(rng.uniform(0.6, 1.3))) *
            float(rng.uniform(3, 9)))[..., None]
    mean = arr.mean(axis=2, keepdims=True)
    arr = mean + (arr - mean) * float(rng.uniform(1.10, 1.40))
    arr += np.array([rng.uniform(-4, 0), rng.uniform(-2, 2), rng.uniform(2, 8)],
                    dtype=np.float32)[None, None]
    arr += rng.standard_normal((h, w, 3)) * float(rng.uniform(1, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)


# ===================== scene renderers =====================


def soft_face_mask(w: int, h: int, soft: float = 0.92) -> np.ndarray:
    """Elliptical alpha mask for blending faces into scenes."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    rx, ry = cx * soft, cy * soft
    d = ((xs - cx) ** 2 / rx ** 2 + (ys - cy) ** 2 / ry ** 2)
    m = np.clip(1 - d, 0, 1)
    return m ** 0.5


def render_real(face: np.ndarray, rng) -> np.ndarray:
    """Real scene: face matches browser's wider crop layout.

    Browser uses side = face_side * 1.9 then resizes to 112×112, so the
    actual face occupies ~112/1.9 ≈ 59 px. We sample 48-78 px to cover
    natural variation (user closer or further from camera).
    """
    bg = synth_bg(rng, palette=str(rng.choice(["room", "office", "dark"])))
    fs = int(rng.integers(48, 79))
    fx = SIZE // 2 - fs // 2 + int(rng.integers(-8, 9))
    fy = SIZE // 2 - fs // 2 + int(rng.integers(-8, 9))
    mask = soft_face_mask(fs, fs, soft=0.96)
    paste_resize(face, bg, fx, fy, fs, fs, alpha=mask)
    # 35% have hand/shoulder/clothing at edge (natural webcam scene)
    if rng.random() < 0.35:
        add_hand_blobs(bg, rng, n=int(rng.integers(1, 3)))
    return webcam_capture(bg, rng)


def render_screen_attack(face: np.ndarray, rng) -> np.ndarray:
    """Replay attack: face inside phone bezel held in hand."""
    bg = synth_bg(rng, palette=str(rng.choice(["room", "office", "dark", "dark"])))
    # Hand silhouettes around the phone
    add_hand_blobs(bg, rng, n=int(rng.integers(1, 4)))
    # Phone bezel — rounded black rectangle
    bezel_w = int(rng.integers(55, 88))
    bezel_h = int(rng.integers(72, 105))
    bx = SIZE // 2 - bezel_w // 2 + int(rng.integers(-8, 9))
    by = SIZE // 2 - bezel_h // 2 + int(rng.integers(-8, 9))
    # Tilt: paint bezel via PIL rounded rect for soft corners
    bezel_canvas = Image.new("L", (SIZE, SIZE), 0)
    dr = ImageDraw.Draw(bezel_canvas)
    radius = int(rng.integers(4, 12))
    dr.rounded_rectangle([bx, by, bx + bezel_w, by + bezel_h], radius=radius, fill=255)
    bm = np.asarray(bezel_canvas, dtype=np.float32) / 255.0
    bezel_color = float(rng.uniform(5, 35))
    bg = bg * (1 - bm[..., None]) + bezel_color * bm[..., None]
    # Inner screen area
    margin = int(rng.integers(3, 8))
    sx = bx + margin; sy = by + margin
    sw = max(8, bezel_w - 2 * margin); sh = max(8, bezel_h - 2 * margin)
    face_displayed = screen_artifact(face, rng)
    fimg = np.asarray(Image.fromarray(face_displayed).resize((sw, sh), Image.BILINEAR),
                      dtype=np.float32)
    x1, y1 = max(0, sx), max(0, sy)
    x2, y2 = min(SIZE, sx + sw), min(SIZE, sy + sh)
    if x2 > x1 and y2 > y1:
        fx1, fy1 = x1 - sx, y1 - sy
        bg[y1:y2, x1:x2] = fimg[fy1:fy1 + (y2 - y1), fx1:fx1 + (x2 - x1)]
    # Specular streak across screen (LCD glare)
    if rng.random() < 0.7:
        ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        angle = float(rng.uniform(0, 3.14))
        offset = float(rng.uniform(-40, 40))
        line = xs * np.cos(angle) + ys * np.sin(angle) + offset
        streak = np.exp(-line ** 2 / (2 * float(rng.uniform(20, 60)) ** 2))
        bg = np.clip(bg + streak[..., None] * float(rng.uniform(15, 55)), 0, 255)
    return webcam_capture(bg, rng)


def render_print_attack(face: np.ndarray, rng) -> np.ndarray:
    """Print attack: face on paper held in hand."""
    bg = synth_bg(rng, palette=str(rng.choice(["room", "office"])))
    add_hand_blobs(bg, rng, n=int(rng.integers(1, 3)))
    # Paper sheet — off-white rectangle, varied size (some held close, some far)
    pw = int(rng.integers(55, 105))
    ph = int(rng.integers(65, 110))
    px = SIZE // 2 - pw // 2 + int(rng.integers(-6, 7))
    py = SIZE // 2 - ph // 2 + int(rng.integers(-6, 7))
    paper_canvas = Image.new("L", (SIZE, SIZE), 0)
    dr = ImageDraw.Draw(paper_canvas)
    dr.rectangle([px, py, px + pw, py + ph], fill=255)
    pm = np.asarray(paper_canvas, dtype=np.float32) / 255.0
    paper_color = np.array([rng.uniform(225, 250), rng.uniform(225, 248),
                             rng.uniform(215, 245)], dtype=np.float32)
    bg = bg * (1 - pm[..., None]) + paper_color[None, None] * pm[..., None]
    # Print artifacts on face
    face_printed = print_artifact(face, rng)
    fimg = np.asarray(Image.fromarray(face_printed).resize((pw, ph), Image.BILINEAR),
                      dtype=np.float32)
    x1, y1 = max(0, px), max(0, py)
    x2, y2 = min(SIZE, px + pw), min(SIZE, py + ph)
    if x2 > x1 and y2 > y1:
        fx1, fy1 = x1 - px, y1 - py
        bg[y1:y2, x1:x2] = fimg[fy1:fy1 + (y2 - y1), fx1:fx1 + (x2 - x1)]
    # Paper texture overlay
    if rng.random() < 0.6:
        noise = rng.standard_normal((SIZE, SIZE, 1)) * float(rng.uniform(1.5, 4))
        bg = bg + noise * pm[..., None]
        bg = np.clip(bg, 0, 255)
    return webcam_capture(bg, rng)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_f = open(OUT / "train_raw.bin", "wb", buffering=8 * 1024 * 1024)
    lbl_f = open(OUT / "train_labels.bin", "wb", buffering=4 * 1024 * 1024)

    idx_mm = np.memmap(MS1M / "index.bin", dtype=np.uint8, mode="r")
    blob = np.memmap(MS1M / "blob.jpg.bin", dtype=np.uint8, mode="r")
    n_ms1m = idx_mm.size // 16
    rng = np.random.default_rng(42)

    total = N_REAL + N_PRINT + N_SCREEN
    print(f"building {total:,} scenes ({N_REAL} real + {N_PRINT} print + {N_SCREEN} screen)")
    sampled = rng.choice(n_ms1m, size=total, replace=False)

    t0 = time.time(); n_ok = 0
    for i, src_idx in enumerate(sampled):
        off, sz, _ = MS1M_INDEX.unpack(idx_mm[src_idx * 16:(src_idx + 1) * 16].tobytes())
        jpeg = blob[off:off + sz].tobytes()
        try:
            img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            if img.shape != (SIZE, SIZE, 3):
                img = np.asarray(Image.fromarray(img).resize((SIZE, SIZE), Image.BILINEAR))
        except Exception:
            continue

        if i < N_REAL:
            out = render_real(img, rng); label = 1
        elif i < N_REAL + N_PRINT:
            out = render_print_attack(img, rng); label = 0
        else:
            out = render_screen_attack(img, rng); label = 0

        raw_f.write(out.tobytes())
        lbl_f.write(bytes([label]))
        n_ok += 1
        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            print(f"  {i+1:,}/{total:,}  {n_ok/dt:.0f} img/s  ETA {(total-i-1)*dt/(i+1)/60:.1f}min")

    raw_f.close(); lbl_f.close()
    (OUT / "meta.json").write_text(json.dumps({
        "n_samples": n_ok, "n_real": N_REAL, "n_print": N_PRINT, "n_screen": N_SCREEN,
        "image_size": SIZE, "image_bytes": SIZE * SIZE * 3, "version": 3,
    }, indent=2))
    print(f"\ndone. {n_ok:,} samples")


if __name__ == "__main__":
    main()
