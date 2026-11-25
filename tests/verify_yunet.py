"""Check native YuNet against the repository fixture and optional WIDER annotations.

Accepts weights exported by tools/export_yunet_weights.py. The optional WIDER
run reports a fixed operating point, with RGB input and the C engine's default
threshold. It counts unmatched detections and ignores invalid GT boxes.
"""
import argparse
import ctypes
import io
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def native_run(lib, engine, image, capacity=525):
    height, width, _ = image.shape
    out = np.empty((capacity, 15), dtype=np.float32)
    count = lib.detect_run(engine, image.ctypes.data_as(ctypes.c_void_p),
                           width, height, out.ctypes.data_as(ctypes.c_void_p), capacity)
    if count < 0 or count > capacity:
        raise ValueError("Native detector returned an invalid count")
    return out[:count]


def check_golden(lib, engine):
    root = ROOT / "tests/golden"
    expected = None
    for stride in (8, 16, 32):
        def head(name):
            return np.load(root / f"onnx_160_{name}_{stride}.npy")[0]
        scores = np.sqrt(head("cls").ravel() * head("obj").ravel())
        idx = np.argmax(scores)
        if expected is not None and scores[idx] <= expected[4]:
            continue
        gy, gx = divmod(int(idx), 160 // stride)
        ox, oy, lw, lh = head("bbox")[idx]
        cx, cy = (gx + ox) * stride, (gy + oy) * stride
        hw, hh = np.exp([lw, lh]) * (stride / 2)
        points = (head("kps")[idx].reshape(5, 2) + [gx, gy]) * stride
        expected = np.r_[cx - hw, cy - hh, cx + hw, cy + hh, scores[idx], points.ravel()]
    image = np.fromfile(ROOT / "tests/test_face_160.raw", dtype="u1").reshape(160, 160, 3)
    out = native_run(lib, engine, image)
    np.testing.assert_allclose(out[0], expected, atol=.002, rtol=0)
    np.testing.assert_array_equal(native_run(lib, engine, image, 1), out[:1])
    lib.detect_set_score_threshold(engine, ctypes.c_float(.005))
    for path in sorted(root.glob("*.npz")):
        with np.load(path) as fixture:
            face = np.asarray(Image.fromarray(fixture["image"]).resize((160, 160)))
        candidates = native_run(lib, engine, face)
        for capacity in (1, 3):
            np.testing.assert_array_equal(native_run(lib, engine, face, capacity),
                                          candidates[:capacity])
    lib.detect_set_score_threshold(engine, ctypes.c_float(.5))
    # Resize and re-use the same engine, including a non-aligned shape.
    for size in ((161, 165), (320, 320), (640, 640)):
        native_run(lib, engine, np.asarray(Image.fromarray(image).resize(size)))
    np.testing.assert_array_equal(native_run(lib, engine, image), out)
    print("Golden maximum absolute difference", float(np.max(abs(out[0] - expected))))


def annotations(archive):
    lines = iter(archive.read("wider_face_split/wider_face_val_bbx_gt.txt").decode().splitlines())
    result = {}
    for name in lines:
        if not name.endswith(".jpg"):
            continue
        boxes = []
        for _ in range(int(next(lines))):
            row = list(map(int, next(lines).split()))
            if len(row) >= 10 and row[7] == 0 and min(row[2:4]) > 0:
                boxes.append(row[:4])
        result[name] = np.array(boxes, dtype=float).reshape(-1, 4)
    return result


def match(detections, gt):
    a = detections[:, :4]
    inter = (np.minimum(a[:, None, 2:], gt[None, :, 2:]) -
             np.maximum(a[:, None, :2], gt[None, :, :2])).clip(0).prod(2)
    union = ((a[:, 2:] - a[:, :2]).prod(1)[:, None] +
             (gt[:, 2:] - gt[:, :2]).prod(1) - inter)
    iou = inter / np.maximum(union, 1e-8)
    rows, cols = linear_sum_assignment(np.where(iou >= .5, 100 + iou, 0), maximize=True)
    return cols[iou[rows, cols] >= .5]


def evaluate_wider(lib, engine, image_archive, annotation_archive, size):
    labels = annotations(annotation_archive)
    names = np.random.default_rng(114).choice(sorted(labels), 640, replace=False)
    counts = np.zeros(3, int)
    for name in names:
        image = Image.open(io.BytesIO(image_archive.read("WIDER_val/images/" + name))).convert("RGB")
        width, height = image.size
        scale = size / max(width, height)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        x, y = (size - w) // 2, (size - h) // 2
        canvas = np.zeros((size, size, 3), "u1")
        canvas[y:y+h, x:x+w] = image.resize((w, h))
        gt = labels[name] * [w / width, h / height, w / width, h / height]
        gt[:, 2:] += gt[:, :2]
        gt += [x, y, x, y]
        det = native_run(lib, engine, canvas)
        found = len(match(det, gt))
        counts += [found, len(det) - found, len(gt) - found]
    print(size, "matched, unmatched detections, missed boxes", counts.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--wider", type=Path, nargs=2, metavar=("IMAGES_ZIP", "LABELS_ZIP"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[160, 320])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "detector.so"
        subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-I", str(ROOT / "include"),
                        str(ROOT / "src/detect.c"), "-lm", "-o", str(binary)], check=True)
        lib = ctypes.CDLL(str(binary))
        lib.detect_init.restype = ctypes.c_void_p
        engine = ctypes.c_void_p(lib.detect_init(bytes(args.weights)))
        if not engine:
            raise RuntimeError("Unable to load YNET weights")
        try:
            check_golden(lib, engine)
            if args.wider:
                with ZipFile(args.wider[0]) as images, ZipFile(args.wider[1]) as labels:
                    for size in args.sizes:
                        evaluate_wider(lib, engine, images, labels, size)
        finally:
            lib.detect_free(engine)


if __name__ == "__main__":
    main()
