"""Build a binary tongue-out dataset:
   class 1 = tongue out (from web-scraped face crops)
   class 0 = no tongue   (from MS1M neutral faces)

Output:
   D:/apps/facex/training/data/smile/train_raw.bin  (N*112*112*3 uint8)
   D:/apps/facex/training/data/smile/train_labels.bin (N uint8)
   D:/apps/facex/training/data/smile/meta.json
"""

import io, json, struct, time
from pathlib import Path
import numpy as np
import cv2

POS_DIR = Path("D:/apps/facex/training/data/smile/faces_pos")
OUT = Path("D:/apps/facex/training/data/smile")
OUT.mkdir(parents=True, exist_ok=True)
MS1M = Path("D:/apps/facex/training/data/ms1m")
SIZE = 112
N_NEG = 3000

MS1M_INDEX = struct.Struct("<QII")


def main():
    # === positives ===
    pos_paths = sorted([p for p in POS_DIR.glob("*.jpg")])
    print(f"positives available: {len(pos_paths)}")
    if len(pos_paths) < 50:
        print("not enough positives, abort"); return

    raw_f = open(OUT / "train_raw.bin", "wb", buffering=8 << 20)
    lbl_f = open(OUT / "train_labels.bin", "wb", buffering=4 << 20)
    n_pos = 0; n_neg = 0

    for p in pos_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        if img.shape != (SIZE, SIZE, 3):
            img = cv2.resize(img, (SIZE, SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_f.write(rgb.tobytes())
        lbl_f.write(bytes([1]))
        n_pos += 1

    # === negatives: random MS1M samples ===
    idx_mm = np.memmap(MS1M / "index.bin", dtype=np.uint8, mode="r")
    blob = np.memmap(MS1M / "blob.jpg.bin", dtype=np.uint8, mode="r")
    n_ms1m = idx_mm.size // 16
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_ms1m, size=N_NEG * 2, replace=False)

    from PIL import Image
    for src_idx in sample_indices:
        if n_neg >= N_NEG: break
        off, sz, _ = MS1M_INDEX.unpack(idx_mm[src_idx * 16:(src_idx + 1) * 16].tobytes())
        jpeg = blob[off:off + sz].tobytes()
        try:
            img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        except Exception:
            continue
        if img.shape != (SIZE, SIZE, 3):
            img = cv2.resize(img, (SIZE, SIZE))
        raw_f.write(img.tobytes())
        lbl_f.write(bytes([0]))
        n_neg += 1

    raw_f.close(); lbl_f.close()
    total = n_pos + n_neg
    (OUT / "meta.json").write_text(json.dumps({
        "n_samples": total, "n_pos": n_pos, "n_neg": n_neg,
        "image_size": SIZE, "image_bytes": SIZE * SIZE * 3,
    }, indent=2))
    print(f"\ndone. {n_pos} positives + {n_neg} negatives = {total} samples")


if __name__ == "__main__":
    main()
