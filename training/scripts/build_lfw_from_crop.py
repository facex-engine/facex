"""
Build lfw.bin from Conrad Sanderson's lfwcrop (64x64 PPM faces).

Reads D:/apps/facex/lfw_pairs.txt for the standard 6000-pair eval split,
loads each referenced face from lfwcrop/lfwcrop_color/faces, upscales
to 112x112, JPEG-encodes, and pickles as (bins, issame).
"""

import io
import pickle
from pathlib import Path

from PIL import Image

DATA = Path("D:/apps/facex/training/data")
LFW_DIR = DATA / "lfwcrop/lfwcrop_color/faces"
PAIRS = Path("D:/apps/facex/lfw_pairs.txt")
OUT = DATA / "lfw.bin"


def find_image(name: str, idx: int) -> Path:
    return LFW_DIR / f"{name}_{idx:04d}.ppm"


def encode(img: Image.Image) -> bytes:
    if img.size != (112, 112):
        img = img.resize((112, 112), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def main():
    lines = PAIRS.read_text().strip().split("\n")
    n_folds, n_per_fold = map(int, lines[0].split())
    pairs = lines[1:]
    expected = n_folds * n_per_fold * 2
    assert len(pairs) == expected, f"{len(pairs)} != {expected}"

    bins, issame = [], []
    missing = 0
    for fold in range(n_folds):
        off = fold * n_per_fold * 2
        for k in range(n_per_fold):  # same
            parts = pairs[off + k].split()
            name, i, j = parts[0], int(parts[1]), int(parts[2])
            p1, p2 = find_image(name, i), find_image(name, j)
            if not p1.exists() or not p2.exists():
                missing += 1; continue
            bins.append(encode(Image.open(p1).convert("RGB")))
            bins.append(encode(Image.open(p2).convert("RGB")))
            issame.append(True)
        for k in range(n_per_fold):  # diff
            parts = pairs[off + n_per_fold + k].split()
            n1, i, n2, j = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1, p2 = find_image(n1, i), find_image(n2, j)
            if not p1.exists() or not p2.exists():
                missing += 1; continue
            bins.append(encode(Image.open(p1).convert("RGB")))
            bins.append(encode(Image.open(p2).convert("RGB")))
            issame.append(False)
    print(f"built {len(issame)} pairs, {missing} missing")
    with open(OUT, "wb") as f:
        pickle.dump((bins, issame), f, protocol=4)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
