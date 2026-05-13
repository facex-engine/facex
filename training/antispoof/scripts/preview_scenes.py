"""Quick preview of the new composite scenes — saves a 3x4 grid."""

import io, struct
from pathlib import Path
import numpy as np
from PIL import Image

import build_dataset as bd

MS1M = Path("D:/apps/facex/training/data/ms1m")
rng = np.random.default_rng(7)

idx_mm = np.memmap(MS1M / "index.bin", dtype=np.uint8, mode="r")
blob = np.memmap(MS1M / "blob.jpg.bin", dtype=np.uint8, mode="r")
n_ms1m = idx_mm.size // 16

faces = []
for src_idx in rng.choice(n_ms1m, size=4, replace=False):
    off, sz, _ = bd.MS1M_INDEX.unpack(idx_mm[src_idx * 16:(src_idx + 1) * 16].tobytes())
    jpeg = blob[off:off + sz].tobytes()
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    if img.shape != (112, 112, 3):
        img = np.asarray(Image.fromarray(img).resize((112, 112), Image.BILINEAR))
    faces.append(img)

rows = []
for f in faces:
    row = [bd.render_real(f, rng), bd.render_print_attack(f, rng), bd.render_screen_attack(f, rng)]
    rows.append(np.concatenate(row, axis=1))
grid = np.concatenate(rows, axis=0)

# Add column labels
out = Image.fromarray(grid)
out.save("D:/apps/facex/training/antispoof/scripts/preview.png")
print(f"saved preview {grid.shape}")
print("columns: REAL | PRINT | SCREEN")
