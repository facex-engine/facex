"""
Pre-decode all 59K MS1M images referenced by mp_labels into a flat
raw RGB blob (112×112×3 uint8). Eliminates per-batch JPEG decode.

Output:
  D:/apps/facex/training/data/mp_labels/raw.bin    (n * 37632 bytes)
  D:/apps/facex/training/data/mp_labels/labels.bin (n * 478*3*4 bytes — just labels, no ms1m_idx)
"""

import io
import json
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = Path("D:/apps/facex/training/data/mp_labels")
MS1M = Path("D:/apps/facex/training/data/ms1m")
SIZE = 112
N_POINTS = 478
LABEL_REC = struct.Struct(f"<I{N_POINTS * 3}f")
LABEL_SZ = LABEL_REC.size
MS1M_INDEX = struct.Struct("<QII")


def main():
    meta = json.loads((LABELS / "meta.json").read_text())
    n = meta["n_records"]
    print(f"pre-decoding {n} images...")
    labels_mm = np.memmap(LABELS / "index.bin", dtype=np.uint8, mode="r")
    ms1m_idx_mm = np.memmap(MS1M / "index.bin", dtype=np.uint8, mode="r")
    blob_mm = np.memmap(MS1M / "blob.jpg.bin", dtype=np.uint8, mode="r")

    raw_f = open(LABELS / "raw.bin", "wb", buffering=8 * 1024 * 1024)
    lbl_f = open(LABELS / "labels.bin", "wb", buffering=4 * 1024 * 1024)

    img_bytes = SIZE * SIZE * 3
    t0 = time.time()
    for i in range(n):
        rec = labels_mm[i * LABEL_SZ:(i + 1) * LABEL_SZ].tobytes()
        unpacked = LABEL_REC.unpack(rec)
        ms1m_idx = unpacked[0]
        # 478*3 floats — strip the first u32, write as flat float32
        labels = np.array(unpacked[1:], dtype=np.float32).tobytes()
        lbl_f.write(labels)

        off, sz, _ = MS1M_INDEX.unpack(
            ms1m_idx_mm[ms1m_idx * 16:(ms1m_idx + 1) * 16].tobytes())
        jpeg = blob_mm[off:off + sz].tobytes()
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).tobytes()
        assert len(arr) == img_bytes
        raw_f.write(arr)

        if (i + 1) % 5000 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  {(i+1)/dt:.0f} img/s  ETA {(n - i - 1) * dt / (i + 1) / 60:.1f} min")
    raw_f.close(); lbl_f.close()
    raw_size = (LABELS / "raw.bin").stat().st_size
    lbl_size = (LABELS / "labels.bin").stat().st_size
    print(f"\nraw.bin: {raw_size/1e6:.1f} MB ({raw_size//img_bytes} images)")
    print(f"labels.bin: {lbl_size/1e6:.1f} MB ({lbl_size//(N_POINTS*3*4)} records)")


if __name__ == "__main__":
    main()
