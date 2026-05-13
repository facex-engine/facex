#!/usr/bin/env python3
"""onnx_to_tflite.py — Convert FaceX models from ONNX to INT8 TFLite.

Pipeline:  ONNX  →  TF SavedModel  (via onnx2tf)  →  TFLite INT8  (via tf.lite)

Output is what tools/compile_vela.sh expects as input. No magic: this is
the standard NXP / Arm conversion path documented in the eIQ guide and the
Ethos-U Vela docs.

Why an external converter? PyTorch → ONNX is straightforward; ONNX → TFLite
is the part with the brittle quantization story. We use `onnx2tf`
(BSD-3-clause, well-maintained) which preserves shapes and per-channel
scales correctly for Arm / NXP delegates.

Install:
    pip install onnx2tf onnxruntime tensorflow numpy

Usage:
    python3 tools/onnx_to_tflite.py edgeface_xs.onnx weights/edgeface_xs_int8.tflite
    python3 tools/onnx_to_tflite.py yunet.onnx weights/yunet_int8.tflite \\
        --calib-dir calib_faces/

`--calib-dir` should hold ~100 face crops (any size, JPEG/PNG). Vela requires
INT8 throughout the graph; we use a representative dataset for activation
quantization. If omitted, the script falls back to random-noise calibration —
which compiles, but the resulting quantization is poor and accuracy will
suffer. Always supply real calibration data for production.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("error: numpy not installed (pip install numpy)", file=sys.stderr)
    sys.exit(1)


def load_calib_images(calib_dir: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load up to 100 RGB images from `calib_dir`, resize to `target_hw`,
    return float32 array shaped [N, H, W, 3] in [-1, 1]."""
    try:
        from PIL import Image
    except ImportError:
        print("error: Pillow needed for calibration (pip install Pillow)", file=sys.stderr)
        sys.exit(1)
    files = sorted(p for p in calib_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:100]
    if not files:
        print(f"warn: no images found in {calib_dir} — using noise calibration",
              file=sys.stderr)
        return np.random.uniform(-1, 1, size=(8, *target_hw, 3)).astype(np.float32)
    arrs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize(target_hw[::-1], Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        arrs.append(a)
    print(f"  calibration: {len(arrs)} images from {calib_dir}", file=sys.stderr)
    return np.stack(arrs, axis=0)


def run_onnx2tf(onnx_path: Path, work_dir: Path) -> None:
    """Invoke onnx2tf as a subprocess. Args are a list (no shell)."""
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(work_dir),
        "-osd",     # output saved_model dir
        "-onwdt",   # don't write debug tflite
        "-nuo",     # no upgrade ops
        "-coion",   # constant folding
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to the `onnx2tf` console-script if the module form failed.
        cmd_fallback = [
            "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(work_dir),
            "-osd", "-onwdt", "-nuo", "-coion",
        ]
        res = subprocess.run(cmd_fallback, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise RuntimeError(f"onnx2tf failed (exit {res.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_in", help="input .onnx file")
    ap.add_argument("tflite_out", help="output .tflite file (INT8 quantized)")
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="directory of representative images for activation quantization")
    ap.add_argument("--input-hw", default="112,112",
                    help="model input height,width (default 112,112 for embedder)")
    ap.add_argument("--keep-tf", action="store_true",
                    help="keep the intermediate SavedModel (for debugging)")
    args = ap.parse_args()

    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        print("error: onnx2tf not installed (pip install onnx2tf)", file=sys.stderr)
        return 1
    try:
        import tensorflow as tf
    except ImportError:
        print("error: tensorflow not installed (pip install tensorflow)", file=sys.stderr)
        return 1

    onnx_path = Path(args.onnx_in).resolve()
    out_path  = Path(args.tflite_out).resolve()
    if not onnx_path.exists():
        print(f"error: {onnx_path} not found", file=sys.stderr); return 1

    target_hw = tuple(int(x) for x in args.input_hw.split(","))
    if len(target_hw) != 2:
        print("error: --input-hw must be H,W", file=sys.stderr); return 1

    print(f"[1/3] ONNX → SavedModel: {onnx_path}", file=sys.stderr)
    work = Path(tempfile.mkdtemp(prefix="facex_onnx2tf_"))
    try:
        run_onnx2tf(onnx_path, work)
        sm_dir = work
        if not (sm_dir / "saved_model.pb").exists():
            for child in work.iterdir():
                if child.is_dir() and (child / "saved_model.pb").exists():
                    sm_dir = child; break
        if not (sm_dir / "saved_model.pb").exists():
            print(f"error: onnx2tf did not produce a SavedModel under {work}",
                  file=sys.stderr)
            return 2

        print(f"[2/3] gathering calibration images ({target_hw[0]}x{target_hw[1]} RGB)",
              file=sys.stderr)
        calib = load_calib_images(args.calib_dir, target_hw) if args.calib_dir \
                else np.random.uniform(-1, 1, size=(8, *target_hw, 3)).astype(np.float32)

        def representative_dataset():
            for i in range(calib.shape[0]):
                yield [calib[i:i+1]]

        print(f"[3/3] SavedModel → INT8 TFLite: {out_path}", file=sys.stderr)
        conv = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type  = tf.int8
        conv.inference_output_type = tf.int8
        tflite_bytes = conv.convert()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(tflite_bytes)
        print(f"  ok: {out_path} ({len(tflite_bytes):,} bytes)", file=sys.stderr)
    finally:
        if not args.keep_tf:
            shutil.rmtree(work, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
