"""
Sprint 5 — Run YuNet 2023mar ONNX on 16 LFW test images, save golden tensors.

These golden tensors are the reference the new C engine must reproduce
(within INT8-quantization tolerance) at Sprint 19.

Usage:
  python tools/yunet_golden.py weights/yunet_2023mar.onnx weights/golden/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from PIL import Image
except ImportError:
    print("pip install pillow", file=sys.stderr)
    sys.exit(1)


# Fixed test set: 16 first-image-of-identity from LFW
TEST_IDENTITIES = [
    "Abel_Pacheco", "Akhmed_Zakayev", "Amber_Tamblyn", "Anders_Fogh_Rasmussen",
    "Angela_Bassett", "Aaron_Eckhart", "Aaron_Patterson", "Adam_Sandler",
    "Adolfo_Aguilar_Zinser", "Agnes_Bruckner", "Ai_Sugiyama", "Aitor_Karanka",
    "Alan_Ball", "Alan_Trammell", "Alanis_Morissette", "Alastair_Campbell",
]


def letterbox(img_np, target=160):
    """Resize so that longer side = target, then center-pad to target x target.
    Returns: (img_padded[target,target,3], scale, dx, dy)"""
    h, w = img_np.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if nw != w or nh != h:
        from PIL import Image as _I
        pil = _I.fromarray(img_np)
        pil = pil.resize((nw, nh), _I.BILINEAR)
        img_np = np.asarray(pil)
    dx = (target - nw) // 2
    dy = (target - nh) // 2
    out = np.zeros((target, target, 3), dtype=np.uint8)
    out[dy:dy + nh, dx:dx + nw] = img_np
    return out, scale, dx, dy


def preprocess(img_uint8_hwc):
    """YuNet 2023mar takes raw uint8 -> CHW float32 in [0..255] (no /255).
    See opencv_zoo YuNet README. We pass the same."""
    chw = img_uint8_hwc.transpose(2, 0, 1).astype(np.float32)  # 3xHxW
    chw = chw[None, ...]  # 1x3xHxW
    return chw


def decode_anchor_free(outputs, strides=(8, 16, 32), input_size=160,
                       score_thresh=0.5, kps_count=5):
    """Decode YuNet outputs into a list of (x1,y1,x2,y2,score,kps[10])."""
    o = outputs
    # Names: cls_8, cls_16, cls_32, obj_8, ..., bbox_8, ..., kps_8, ...
    cls = {8: o["cls_8"][0], 16: o["cls_16"][0], 32: o["cls_32"][0]}
    obj = {8: o["obj_8"][0], 16: o["obj_16"][0], 32: o["obj_32"][0]}
    bbox = {8: o["bbox_8"][0], 16: o["bbox_16"][0], 32: o["bbox_32"][0]}
    kps = {8: o["kps_8"][0], 16: o["kps_16"][0], 32: o["kps_32"][0]}

    faces = []
    for s in strides:
        feat_h = input_size // s
        feat_w = input_size // s
        n_cells = feat_h * feat_w
        if len(cls[s]) != n_cells:
            raise ValueError(f"stride {s} output does not match input size {input_size}")
        # ONNX emits probabilities. The native head applies its sigmoid in C.
        scores = np.sqrt(np.clip(cls[s] * obj[s], 0, 1)).ravel()
        for idx in np.flatnonzero(scores >= score_thresh):
            sc = float(scores[idx])
            gy, gx = divmod(int(idx), feat_w)
            offset_x, offset_y, log_w, log_h = bbox[s][idx]
            cx, cy = (gx + offset_x) * s, (gy + offset_y) * s
            w, h = np.exp(log_w) * s, np.exp(log_h) * s
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            kp = kps[s][idx].copy()
            kp[0::2] = (gx + kp[0::2]) * s
            kp[1::2] = (gy + kp[1::2]) * s
            faces.append((x1, y1, x2, y2, sc, kp.tolist()))
    return faces


def nms(faces, iou_thresh=0.4):
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: -f[4])
    keep = []
    for f in faces:
        x1, y1, x2, y2, sc, _ = f
        area_a = (x2 - x1) * (y2 - y1)
        drop = False
        for g in keep:
            gx1, gy1, gx2, gy2, _, _ = g
            ix1 = max(x1, gx1); iy1 = max(y1, gy1)
            ix2 = min(x2, gx2); iy2 = min(y2, gy2)
            iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_b = (gx2 - gx1) * (gy2 - gy1)
            union = area_a + area_b - inter
            if union > 0 and inter / union > iou_thresh:
                drop = True; break
        if not drop:
            keep.append(f)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("out_dir")
    ap.add_argument("--lfw", default="lfw")
    ap.add_argument("--input-size", type=int, default=160)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_size <= 0 or args.input_size % 32:
        ap.error("--input-size must be a positive multiple of 32")
    # Public YuNet declares fixed spatial dimensions despite using convolutions
    # throughout. Set the reference graph to the same resolution as the C run.
    import onnx
    model = onnx.load(args.onnx_path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    for dim in dims[2:4]:
        dim.dim_value = args.input_size
    for output in model.graph.output:
        stride = int(output.name.rsplit("_", 1)[1])
        output.type.tensor_type.shape.dim[1].dim_value = (args.input_size // stride) ** 2
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Loaded {args.onnx_path}")
    print(f"Outputs: {out_names}")

    summary = {"input_size": args.input_size, "items": []}
    for ident in TEST_IDENTITIES[:args.n]:
        path = Path(args.lfw) / ident / f"{ident}_0001.jpg"
        if not path.exists():
            print(f"  SKIP missing {path}")
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        padded, scale, dx, dy = letterbox(img, args.input_size)
        x = preprocess(padded)
        outs = sess.run(out_names, {"input": x})
        out_dict = {n: arr for n, arr in zip(out_names, outs)}
        decoded = decode_anchor_free(out_dict, input_size=args.input_size)
        decoded = nms(decoded)
        # save raw tensors + decoded
        np.savez(out_dir / f"{ident}.npz",
                 image=padded, scale=scale, dx=dx, dy=dy,
                 **out_dict)
        summary["items"].append({
            "identity": ident,
            "src_size": img.shape[:2],
            "letterbox": {"scale": scale, "dx": dx, "dy": dy},
            "n_faces": len(decoded),
            "faces": [
                {"x1": f[0], "y1": f[1], "x2": f[2], "y2": f[3],
                 "score": f[4], "kps": f[5]}
                for f in decoded
            ],
        })
        print(f"  {ident}: {len(decoded)} face(s) "
              f"{[round(f[4],3) for f in decoded]}")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, cls=NpEncoder))
    n_with = sum(1 for it in summary["items"] if it["n_faces"] >= 1)
    print()
    print(f"Done. {n_with}/{len(summary['items'])} images with at least 1 face.")
    print(f"Tensors + summary in {out_dir}/")


if __name__ == "__main__":
    main()
