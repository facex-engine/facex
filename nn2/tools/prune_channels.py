#!/usr/bin/env python3
"""
Channel-prune YOLOv8n: reduce all channel widths by a factor.
Creates a new, narrower model that runs faster.

Width multipliers:
  1.00 = original (16,32,64,128,256) — 3.1M params, 8.5ms
  0.75 = pruned   (12,24,48,96,192)  — 1.7M params, ~5ms target
  0.50 = tiny     (8,16,32,64,128)   — 0.8M params, ~3ms target

For NVR with 3 classes (person,car,bicycle): 0.75 should have minimal quality impact.
"""

import struct
import numpy as np
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Need ultralytics: pip install ultralytics")
    sys.exit(1)

def export_pruned(width_mult=0.75, num_classes=80, imgsz=320):
    """Export a width-scaled YOLOv8n."""
    import yaml, tempfile, os

    # YOLOv8n architecture with scaled channels
    # Original channels: [16, 32, 64, 128, 256] (after first conv)
    # Scale factor applied to all
    base = [16, 32, 64, 128, 256]
    scaled = [max(8, int(c * width_mult / 8) * 8) for c in base]  # round to 8
    print(f"Channel widths: {base} → {scaled}")

    # Train a scaled model from scratch? No — that requires training data.
    # Instead: load pretrained YOLOv8n and prune channels by importance (L1 norm).

    model = YOLO("yolov8n.pt")

    # Export standard model first (we'll measure the baseline)
    print(f"\nExporting standard YOLOv8n@{imgsz}...")
    model.export(format="onnx", imgsz=imgsz, simplify=True)

    # For actual pruning, we need torch-pruning library
    try:
        import torch_pruning as tp
        import torch

        print(f"\nPruning with width_mult={width_mult}...")
        pruning_model = YOLO("yolov8n.pt").model

        # Setup pruner
        example_input = torch.randn(1, 3, imgsz, imgsz)
        imp = tp.importance.MagnitudeImportance(p=2)  # L2 norm
        ignored_layers = []

        # Don't prune detection head outputs
        for m in pruning_model.modules():
            if hasattr(m, 'cv2') and hasattr(m, 'cv3'):  # Detect head
                ignored_layers.append(m)

        pruner = tp.pruner.MagnitudePruner(
            pruning_model,
            example_input,
            importance=imp,
            pruning_ratio=1.0 - width_mult,
            ignored_layers=ignored_layers,
        )

        pruner.step()

        # Count params
        n_params = sum(p.numel() for p in pruning_model.parameters())
        print(f"Pruned model: {n_params/1e6:.2f}M params")

        # Export pruned ONNX
        torch.onnx.export(pruning_model, example_input,
                          f"yolov8n_pruned_{int(width_mult*100)}.onnx",
                          opset_version=13)
        print(f"Exported yolov8n_pruned_{int(width_mult*100)}.onnx")

    except ImportError:
        print("\ntorch-pruning not available. Using width-scaled YOLOv8 instead.")
        print("Install: pip install torch-pruning")
        print("\nAlternative: use ultralytics YOLOv8-nano with custom yaml...")

        # Create custom yaml with scaled channels
        custom_yaml = {
            'nc': num_classes,
            'scales': {
                'n': [0.33 * width_mult, 0.25 * width_mult, 1024]
            },
            'backbone': [
                [[-1, 1, 'Conv', [64, 3, 2]]],
                [[-1, 1, 'Conv', [128, 3, 2]]],
                [[-1, 3, 'C2f', [128, True]]],
                [[-1, 1, 'Conv', [256, 3, 2]]],
                [[-1, 6, 'C2f', [256, True]]],
                [[-1, 1, 'Conv', [512, 3, 2]]],
                [[-1, 6, 'C2f', [512, True]]],
                [[-1, 1, 'Conv', [1024, 3, 2]]],
                [[-1, 3, 'C2f', [1024, True]]],
                [[-1, 1, 'SPPF', [1024, 5]]],
            ]
        }
        print(f"\nTo create a {width_mult:.0%} width model:")
        print(f"  yolo train model=yolov8n.yaml data=coco.yaml epochs=100 \\")
        print(f"       scale={width_mult}")

    # Also benchmark different input sizes
    print(f"\n=== Input size comparison ===")
    for size in [320, 256, 224, 192]:
        gflops = 2.04 * (size/320)**2  # approximate
        est_ms = 8.5 * (size/320)**2   # linear scaling estimate
        print(f"  {size}×{size}: ~{gflops:.1f} GFLOPs, ~{est_ms:.1f}ms estimated")

if __name__ == "__main__":
    wm = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
    export_pruned(width_mult=wm)
