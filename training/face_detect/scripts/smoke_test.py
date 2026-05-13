"""Run trained face detector on sample images, save visualizations."""

import sys, math
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ONNX = "D:/apps/facex/wasm/facex_detect.onnx"
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
print(f"loaded {ONNX}")

STRIDES = (8, 16, 32)


def preprocess(img_pil):
    """Letterbox 320×320, normalize [-1, 1], return (CHW tensor, scale, pad)."""
    W, H = img_pil.size
    s = 320 / max(W, H)
    nw, nh = int(W * s), int(H * s)
    img = img_pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (320, 320), (0, 0, 0))
    canvas.paste(img, (0, 0))
    arr = np.asarray(canvas, dtype=np.float32)
    arr = (arr - 127.5) / 128.0
    arr = arr.transpose(2, 0, 1)[None]
    return arr, s


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def decode(outputs, score_th=0.3, nms_iou=0.4):
    """Decode multi-scale outputs to list of (x1,y1,x2,y2,score, kps[5,2])."""
    all_boxes = []; all_scores = []; all_kps = []
    for li, s in enumerate(STRIDES):
        c, b, k = outputs[3*li], outputs[3*li+1], outputs[3*li+2]
        H, W = c.shape[-2:]
        cls = sigmoid(c[0, 0])         # (H, W)
        box = np.maximum(b[0], 0)      # (4, H, W)
        kps = k[0]                     # (10, H, W)
        ys, xs = np.where(cls >= score_th)
        for y, x in zip(ys, xs):
            cx = (x + 0.5) * s; cy = (y + 0.5) * s
            l, t, r, bb = box[:, y, x]
            x1 = cx - l * s; y1 = cy - t * s
            x2 = cx + r * s; y2 = cy + bb * s
            kp_offs = kps[:, y, x].reshape(5, 2)
            kp_xy = kp_offs * s + np.array([cx, cy])
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(float(cls[y, x]))
            all_kps.append(kp_xy)
    if not all_boxes: return []
    # NMS
    boxes = np.array(all_boxes); scores = np.array(all_scores); kpa = np.array(all_kps)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        a = boxes[i]; rest = boxes[order[1:]]
        ix1 = np.maximum(a[0], rest[:, 0]); iy1 = np.maximum(a[1], rest[:, 1])
        ix2 = np.minimum(a[2], rest[:, 2]); iy2 = np.minimum(a[3], rest[:, 3])
        iw = np.maximum(ix2 - ix1, 0); ih = np.maximum(iy2 - iy1, 0)
        inter = iw * ih
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_r = np.maximum(rest[:, 2] - rest[:, 0], 0) * np.maximum(rest[:, 3] - rest[:, 1], 0)
        iou = inter / (area_a + area_r - inter + 1e-7)
        order = order[1:][iou < nms_iou]
    return [(boxes[i], scores[i], kpa[i]) for i in keep]


def annotate(img_pil, dets, scale):
    img = img_pil.copy().convert("RGB")
    dr = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), score, kps in dets:
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
        dr.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        dr.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill=(0, 255, 0))
        for kx, ky in kps:
            kx /= scale; ky /= scale
            dr.ellipse([kx-3, ky-3, kx+3, ky+3], fill=(255, 100, 100))
    return img


def run(src_path):
    img = Image.open(src_path).convert("RGB")
    arr, scale = preprocess(img)
    outs = sess.run(None, {"input": arr})
    # outs in declared order: cls_p3, box_p3, kps_p3, cls_p4, ...
    dets = decode(outs, score_th=0.3)
    print(f"  {Path(src_path).name}: {len(dets)} faces detected (≥0.3)")
    for (x1,y1,x2,y2), sc, _ in dets[:5]:
        print(f"    box={x1/scale:.0f},{y1/scale:.0f}-{x2/scale:.0f},{y2/scale:.0f}  score={sc:.3f}")
    out = annotate(img, dets, scale)
    dst = Path("D:/apps/facex/training/face_detect/runs") / f"viz_{Path(src_path).name}"
    out.save(dst)
    print(f"    → {dst}")


# Test on a few real photos
test_paths = [
    "D:/apps/facex/training/Silent-Face-Anti-Spoofing/images/sample/image_T1.jpg",
    "D:/apps/facex/training/Silent-Face-Anti-Spoofing/images/sample/image_F1.jpg",
    "D:/apps/facex/training/Silent-Face-Anti-Spoofing/images/sample/image_F2.jpg",
    # Multi-face WIDER FACE val image
    "D:/apps/facex/training/data/widerface/WIDER_val/images/0--Parade/0_Parade_Parade_0_873.jpg",
]
for p in test_paths:
    if Path(p).exists():
        run(p)
    else:
        print(f"  SKIP missing: {p}")
