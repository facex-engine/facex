"""
Prepare WFLW for landmark training.

For each annotation line:
  - parse 98 (x,y) landmarks + bbox + image path
  - crop the bbox region (with small margin)
  - resize to 112x112
  - rescale landmarks into [0, 1] crop-local coordinates
  - JPEG-encode the crop
  - write to a blob+index store identical in spirit to the MS1M layout

Output:
  D:/apps/facex/training/data/wflw/
    train/{blob.jpg.bin, index.bin, meta.json}
    test/{blob.jpg.bin, index.bin, meta.json}

Index record (one per face):
    u64 offset  (jpeg start in blob.jpg.bin)
    u32 size    (jpeg byte length)
    f32[196]    (98 landmarks as x,y in [0,1] relative to the 112x112 crop)
"""

import io
import json
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("D:/apps/facex/training/data")
ANN_DIR = ROOT / "WFLW_annotations/list_98pt_rect_attr_train_test"
IMG_DIR = ROOT / "WFLW_images"
OUT = ROOT / "wflw"

CROP_SIZE = 112
MARGIN = 0.15        # 15% of bbox size on each side
NUM_PTS = 98
INDEX_REC = struct.Struct(f"<QI{NUM_PTS * 2}f")   # 12 + 4*196 = 796 bytes
INDEX_SIZE = INDEX_REC.size


def parse_line(line: str):
    parts = line.strip().split()
    pts = np.array([float(p) for p in parts[:NUM_PTS * 2]],
                   dtype=np.float32).reshape(NUM_PTS, 2)
    bbox = [int(parts[NUM_PTS * 2 + i]) for i in range(4)]   # xmin ymin xmax ymax
    img_path = parts[-1]
    return pts, bbox, img_path


def square_bbox(bbox, w, h, margin: float = MARGIN):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(bw, bh) * (1 + 2 * margin)
    sx1 = cx - side / 2
    sy1 = cy - side / 2
    sx2 = cx + side / 2
    sy2 = cy + side / 2
    return sx1, sy1, sx2, sy2


def process_split(name: str, ann_file: Path):
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    blob_path = out_dir / "blob.jpg.bin"
    idx_path = out_dir / "index.bin"
    meta_path = out_dir / "meta.json"

    blob_f = open(blob_path, "wb", buffering=8 * 1024 * 1024)
    idx_f = open(idx_path, "wb", buffering=2 * 1024 * 1024)

    n = 0
    skipped = 0
    offset = 0
    t0 = time.time()

    with open(ann_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            pts, bbox, img_rel = parse_line(line)
            img_p = IMG_DIR / img_rel
            if not img_p.exists():
                skipped += 1; continue
            try:
                img = Image.open(img_p).convert("RGB")
            except Exception:
                skipped += 1; continue
            W, H = img.size

            sx1, sy1, sx2, sy2 = square_bbox(bbox, W, H)

            # Clamp + pad if crop extends beyond image
            # Use PIL crop with negative values + paste onto black canvas
            side = int(round(sx2 - sx1))
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            paste_x = max(0, -int(round(sx1)))
            paste_y = max(0, -int(round(sy1)))
            src_x1 = max(0, int(round(sx1)))
            src_y1 = max(0, int(round(sy1)))
            src_x2 = min(W, int(round(sx2)))
            src_y2 = min(H, int(round(sy2)))
            src_crop = img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(src_crop, (paste_x, paste_y))

            # Rescale landmarks to [0, 1] relative to canvas, then to 112x112 [0,1]
            pts_local = (pts - np.array([sx1, sy1], dtype=np.float32)) / side
            # clamp slightly outside [0,1] to keep numerics sane
            pts_local = np.clip(pts_local, -0.1, 1.1)

            face = canvas.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
            buf = io.BytesIO()
            face.save(buf, "JPEG", quality=95)
            jpeg = buf.getvalue()

            blob_f.write(jpeg)
            idx_f.write(INDEX_REC.pack(offset, len(jpeg), *pts_local.reshape(-1)))
            offset += len(jpeg)
            n += 1

            if n % 1000 == 0:
                dt = time.time() - t0
                mb = offset / 1e6
                print(f"  {name}: {n:>5} faces | {mb:>5.1f} MB | {n/dt:.0f} it/s | skipped={skipped}")

    blob_f.close(); idx_f.close()
    meta = {
        "n_faces": n, "skipped": skipped,
        "n_landmarks": NUM_PTS,
        "crop_size": CROP_SIZE, "margin": MARGIN,
        "blob_bytes": offset,
        "blob_path": str(blob_path),
        "index_path": str(idx_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{name}] done: {n} faces, {skipped} skipped, blob {offset/1e6:.1f} MB")


def main():
    process_split("train", ANN_DIR / "list_98pt_rect_attr_train.txt")
    process_split("test",  ANN_DIR / "list_98pt_rect_attr_test.txt")


if __name__ == "__main__":
    main()
