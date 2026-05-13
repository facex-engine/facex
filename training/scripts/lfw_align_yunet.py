"""
Re-align LFW-crop faces using YuNet detector + 5-point similarity transform.

LFW-crop is 64x64, model expects 112x112 aligned. Our straight upscale loses
the per-image alignment info. YuNet (in weights/yunet_2023mar.onnx) detects
faces with 5 landmarks (left eye, right eye, nose, left mouth, right mouth);
we apply InsightFace's canonical 5-point destination and warp.

Builds D:/apps/facex/training/data/lfw_aligned.bin (replaces lfw.bin).
"""

import io
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATA = Path("D:/apps/facex/training/data")
LFW_DIR = DATA / "lfwcrop/lfwcrop_color/faces"
PAIRS = Path("D:/apps/facex/lfw_pairs.txt")
YUNET = Path("D:/apps/facex/weights/yunet_2023mar.onnx")
OUT = DATA / "lfw_aligned.bin"

# InsightFace canonical 5-point template for 112x112 aligned faces.
ARC_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def load_yunet():
    """Wrapper around cv2.FaceDetectorYN — handles multi-scale decoding + NMS."""
    return cv2.FaceDetectorYN.create(
        str(YUNET), "",
        (320, 320),
        score_threshold=0.3,
        nms_threshold=0.3,
        top_k=10,
    )


def detect_landmarks(det, img_rgb):
    """Returns 5x2 landmarks in pixel coords of img_rgb, or None on failure."""
    H, W = img_rgb.shape[:2]
    det.setInputSize((W, H))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    # Pick highest-confidence face.
    # FaceDetectorYN output: [x, y, w, h, kp0x, kp0y, ..., kp4y, score]
    best = faces[np.argmax(faces[:, -1])]
    if best[-1] < 0.3:
        return None
    return best[4:14].reshape(5, 2).astype(np.float32)


def align_face(img_rgb, landmarks):
    """Apply similarity transform to map landmarks -> ARC_TEMPLATE,
    output 112x112 aligned RGB."""
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32),
                                        ARC_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    out = cv2.warpAffine(img_rgb, M, (112, 112),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return out


def fallback_center(img_rgb):
    """No detection -> assume face already centered, just resize."""
    return cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_CUBIC)


def encode(img_rgb):
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def find_image(name, idx):
    return LFW_DIR / f"{name}_{idx:04d}.ppm"


def load_face(sess, path, cache):
    if path in cache:
        return cache[path]
    arr = np.asarray(Image.open(path).convert("RGB"))
    # Upscale 64 -> 256 first; YuNet works better on larger input
    big = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_CUBIC)
    kps = detect_landmarks(sess, big)
    if kps is not None:
        aligned = align_face(big, kps)
        if aligned is None:
            aligned = fallback_center(big)
            detected = False
        else:
            detected = True
    else:
        aligned = fallback_center(big)
        detected = False
    enc = encode(aligned)
    cache[path] = (enc, detected)
    return cache[path]


def main():
    sess = load_yunet()
    lines = PAIRS.read_text().strip().split("\n")
    n_folds, n_per_fold = map(int, lines[0].split())
    pairs = lines[1:]

    bins, issame = [], []
    cache = {}
    detected_count = 0
    total = 0
    missing = 0
    for fold in range(n_folds):
        off = fold * n_per_fold * 2
        for k in range(n_per_fold):
            parts = pairs[off + k].split()
            name, i, j = parts[0], int(parts[1]), int(parts[2])
            p1, p2 = find_image(name, i), find_image(name, j)
            if not p1.exists() or not p2.exists():
                missing += 1; continue
            e1, d1 = load_face(sess, p1, cache)
            e2, d2 = load_face(sess, p2, cache)
            detected_count += int(d1) + int(d2)
            total += 2
            bins.append(e1); bins.append(e2)
            issame.append(True)
        for k in range(n_per_fold):
            parts = pairs[off + n_per_fold + k].split()
            n1, i, n2, j = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1, p2 = find_image(n1, i), find_image(n2, j)
            if not p1.exists() or not p2.exists():
                missing += 1; continue
            e1, d1 = load_face(sess, p1, cache)
            e2, d2 = load_face(sess, p2, cache)
            detected_count += int(d1) + int(d2)
            total += 2
            bins.append(e1); bins.append(e2)
            issame.append(False)
        if (fold + 1) % 2 == 0:
            print(f"  fold {fold+1}/{n_folds}  detected {detected_count}/{total} ({100*detected_count/max(1,total):.1f}%)")
    print(f"pairs: {len(issame)}  missing: {missing}  detection rate: {detected_count}/{total} ({100*detected_count/max(1,total):.1f}%)")
    with open(OUT, "wb") as f:
        pickle.dump((bins, issame), f, protocol=4)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
