"""
Extract MediaPipe FaceMesh 478-landmark labels for a subset of MS1M.

Output:
  D:/apps/facex/training/data/mp_labels/index.bin  (struct: <I {478*3}f)
    one record per image with 478 (x, y, z) coords in [0,1] (z roughly [-0.2, 0])
  D:/apps/facex/training/data/mp_labels/meta.json
"""

import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


MS1M = Path("D:/apps/facex/training/data/ms1m")
MP_MODEL = Path("D:/apps/facex/training/data/face_landmarker.task")
OUT_DIR = Path("D:/apps/facex/training/data/mp_labels")

N_TARGET = 60_000          # how many MS1M images to label
N_POINTS = 478
REC = struct.Struct(f"<I{N_POINTS * 3}f")   # u32 ms1m_idx + 478*3 floats
REC_SZ = REC.size


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = OUT_DIR / "index.bin"
    meta_path = OUT_DIR / "meta.json"

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MP_MODEL)),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    det = vision.FaceLandmarker.create_from_options(opts)

    ms1m_idx = np.memmap(MS1M / "index.bin", dtype=np.uint8, mode="r")
    blob     = np.memmap(MS1M / "blob.jpg.bin", dtype=np.uint8, mode="r")
    n_ms1m = ms1m_idx.size // 16
    MS_INDEX = struct.Struct("<QII")

    # Sample N_TARGET evenly across the dataset (gives diverse identities).
    step = max(1, n_ms1m // N_TARGET)
    sample_idxs = list(range(0, n_ms1m, step))[:N_TARGET]
    print(f"MS1M total: {n_ms1m:,}, sampling every {step} -> {len(sample_idxs):,} images")

    out_f = open(idx_path, "wb", buffering=4 * 1024 * 1024)
    t0 = time.time()
    n_ok = 0; n_miss = 0
    for k, src_idx in enumerate(sample_idxs):
        off, sz, _ = MS_INDEX.unpack(ms1m_idx[src_idx * 16:(src_idx + 1) * 16].tobytes())
        jpeg = blob[off:off + sz].tobytes()
        try:
            img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        except Exception:
            n_miss += 1; continue

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        res = det.detect(mp_img)
        if not res.face_landmarks:
            n_miss += 1
            continue
        lm = res.face_landmarks[0]
        if len(lm) != N_POINTS:
            n_miss += 1; continue

        flat = []
        for p in lm:
            flat.extend([p.x, p.y, p.z])
        out_f.write(REC.pack(src_idx, *flat))
        n_ok += 1

        if (k + 1) % 1000 == 0:
            dt = time.time() - t0
            print(f"  {k+1:,}/{len(sample_idxs):,}  ok={n_ok:,}  miss={n_miss}  "
                  f"{n_ok/dt:.1f} img/s  ETA {(len(sample_idxs) - k - 1) * dt / (k+1) / 60:.1f} min")

    out_f.close()
    meta = {
        "n_records": n_ok, "n_missed": n_miss,
        "n_points": N_POINTS,
        "record_bytes": REC_SZ,
        "ms1m_total": n_ms1m,
        "sampled_indices": len(sample_idxs),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\ndone. {n_ok:,} records ({idx_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
