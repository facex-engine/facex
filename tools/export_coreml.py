#!/usr/bin/env python3
"""export_coreml.py — Convert FaceX models from ONNX to a Core ML
`.mlpackage` for ANE dispatch on Apple Silicon.

Pipeline:  ONNX  →  Core ML mlprogram  (via coremltools)  →  optional INT8 palettization

Output is what `src/backend_coreml.m` (built into libfacex.a when
COREML=1) loads at runtime.

Install:
    pip install coremltools onnx numpy Pillow

Usage:
    python3 tools/export_coreml.py edgeface_xs.onnx weights/edgeface_xs.mlpackage
    python3 tools/export_coreml.py yunet.onnx weights/yunet.mlpackage \\
        --input-hw 160,160 --no-palettize

`--palettize` (default on) reduces weights to ~6 bits per parameter
via k-means on each conv filter — drops the package from ≈ 7 MB to
≈ 1.8 MB and unlocks the ANE INT8 path on macOS 14+. Pass
`--no-palettize` to keep FP16/FP32 weights (slightly higher
accuracy, larger package, slower ANE dispatch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("error: numpy not installed (pip install numpy)", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_in", help="input .onnx file")
    ap.add_argument("mlpackage_out", help="output .mlpackage directory")
    ap.add_argument("--input-hw", default="112,112",
                    help="model input H,W (default 112,112 for embedder)")
    ap.add_argument("--minimum-deployment-target", default="macOS13",
                    help="minimum macOS target (macOS13 = Ventura → unlocks the "
                         "Core ML mlprogram format which is required for ANE INT8). "
                         "Use macOS14 for compute-plan introspection.")
    ap.add_argument("--no-palettize", action="store_true",
                    help="skip INT8 weight palettization (keeps FP16, larger package)")
    ap.add_argument("--palettize-bits", type=int, default=6,
                    choices=[2, 4, 6, 8],
                    help="bits-per-weight for palettization (default 6)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx_in).resolve()
    out_path  = Path(args.mlpackage_out).resolve()
    if not onnx_path.exists():
        print(f"error: {onnx_path} not found", file=sys.stderr); return 1
    if out_path.suffix != ".mlpackage":
        print("warn: output path does not end in .mlpackage — Core ML expects that suffix",
              file=sys.stderr)

    try:
        import coremltools as ct
    except ImportError:
        print("error: coremltools not installed (pip install coremltools)",
              file=sys.stderr)
        return 1

    H, W = (int(x) for x in args.input_hw.split(","))
    print(f"[1/2] ONNX → Core ML mlprogram: {onnx_path}", file=sys.stderr)

    # Convert. We force ML Program format (vs older NeuralNetwork) because
    # the Core ML compiler needs it for ANE INT8 dispatch on macOS 14+.
    deploy_target = getattr(ct.target, args.minimum_deployment_target,
                             ct.target.macOS13)
    image_input = ct.TensorType(
        name="input",
        shape=(1, 3, H, W),
        dtype=np.float32,
    )
    mlmodel = ct.convert(
        str(onnx_path),
        inputs=[image_input],
        convert_to="mlprogram",
        minimum_deployment_target=deploy_target,
        compute_precision=ct.precision.FLOAT16,
    )

    if not args.no_palettize:
        try:
            from coremltools.optimize.coreml import (
                OpPalettizerConfig,
                OptimizationConfig,
                palettize_weights,
            )
        except ImportError:
            print("warn: coremltools.optimize not available (need coremltools 7+) — "
                  "skipping palettization", file=sys.stderr)
        else:
            print(f"[2/2] palettizing to {args.palettize_bits} bits", file=sys.stderr)
            cfg = OptimizationConfig(
                global_config=OpPalettizerConfig(
                    nbits=args.palettize_bits,
                    mode="kmeans",
                ),
            )
            mlmodel = palettize_weights(mlmodel, config=cfg)
    else:
        print("[2/2] palettization disabled (--no-palettize)", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))

    # Report.
    sz = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
    print(f"  ok: {out_path} ({sz / 1024:.0f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
