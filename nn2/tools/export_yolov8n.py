#!/usr/bin/env python3
"""
export_yolov8n.py — Export YOLOv8n weights to nn2 binary format.

Usage:
    pip install ultralytics
    python tools/export_yolov8n.py [--model yolov8n.pt] [--size 640] [--output weights/yolov8n.bin]

Weight file format:
    [4]  magic "NN2W"
    [4]  uint32 num_classes
    [4]  uint32 input_size
    [4]  uint32 num_tensors
    For each tensor:
        [4]  uint32 size_in_floats
        [N*4] float32 data

Tensors are written in the exact order expected by nn2_load_weights():
    backbone convs → backbone c2f blocks → sppf → neck c2f → neck convs → head
Each conv = [weight_tensor, bias_tensor] (bias includes folded BN).
"""

import struct
import argparse
import numpy as np
import torch


def fold_bn(conv, bn):
    """Fold BatchNorm into Conv weights: w_new, b_new."""
    w = conv.weight.data.clone()
    if conv.bias is not None:
        b = conv.bias.data.clone()
    else:
        b = torch.zeros(w.shape[0])

    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = gamma / std

    # Fold: w_new = w * scale, b_new = (b - mean) * scale + beta
    w_new = w * scale.reshape(-1, 1, 1, 1) if w.dim() == 4 else w * scale.reshape(-1, 1)
    b_new = (b - mean) * scale + beta

    return w_new.numpy(), b_new.numpy()


def write_tensor(f, data):
    """Write a single tensor: [size_in_floats][float32 data]."""
    flat = data.flatten().astype(np.float32)
    f.write(struct.pack('<I', len(flat)))
    f.write(flat.tobytes())


def write_conv(f, conv_module, tensors_count):
    """Write a Conv (with fused BN) or raw Conv2d."""
    if hasattr(conv_module, 'conv') and hasattr(conv_module, 'bn'):
        # ultralytics Conv = conv + bn + act
        w, b = fold_bn(conv_module.conv, conv_module.bn)
    elif hasattr(conv_module, 'weight'):
        # raw nn.Conv2d (no BN, used in detection head output)
        w = conv_module.weight.data.numpy()
        b = conv_module.bias.data.numpy() if conv_module.bias is not None else np.zeros(w.shape[0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown conv type: {type(conv_module)}")

    # Reshape to [cout, cin*k*k] for GEMM
    cout = w.shape[0]
    w_flat = w.reshape(cout, -1)

    write_tensor(f, w_flat)
    write_tensor(f, b)
    return tensors_count + 2


def write_c2f(f, c2f, tc):
    """Write a C2f block: cv1 + n*bottleneck + cv2."""
    tc = write_conv(f, c2f.cv1, tc)
    for bn in c2f.m:
        tc = write_conv(f, bn.cv1, tc)
        tc = write_conv(f, bn.cv2, tc)
    tc = write_conv(f, c2f.cv2, tc)
    return tc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--size', type=int, default=640, help='Input size')
    parser.add_argument('--output', default='weights/yolov8n.bin', help='Output path')
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    m = model.model.model  # nn.Sequential of layers

    nc = model.model.nc
    print(f"Model: {args.model}, classes={nc}, input_size={args.size}")

    with open(args.output, 'wb') as f:
        # Header
        f.write(b'NN2W')
        f.write(struct.pack('<III', nc, args.size, 0))  # num_tensors placeholder
        tc = 0

        # Backbone convs (layers 0, 1, 3, 5, 7 in YOLOv8n)
        backbone_conv_indices = [0, 1, 3, 5, 7]
        for idx in backbone_conv_indices:
            tc = write_conv(f, m[idx], tc)
            print(f"  backbone conv {idx}: {m[idx].conv.weight.shape}")

        # Backbone C2f (layers 2, 4, 6, 8)
        c2f_indices = [2, 4, 6, 8]
        for idx in c2f_indices:
            tc = write_c2f(f, m[idx], tc)
            print(f"  backbone c2f {idx}: n={len(m[idx].m)}")

        # SPPF (layer 9)
        sppf = m[9]
        tc = write_conv(f, sppf.cv1, tc)
        tc = write_conv(f, sppf.cv2, tc)
        print(f"  sppf: cv1={sppf.cv1.conv.weight.shape}, cv2={sppf.cv2.conv.weight.shape}")

        # Neck C2f blocks (layers 12, 15, 18, 21)
        neck_c2f_indices = [12, 15, 18, 21]
        for idx in neck_c2f_indices:
            tc = write_c2f(f, m[idx], tc)
            print(f"  neck c2f {idx}: n={len(m[idx].m)}")

        # Neck convs (layers 16, 19) — downsample
        neck_conv_indices = [16, 19]
        for idx in neck_conv_indices:
            tc = write_conv(f, m[idx], tc)
            print(f"  neck conv {idx}: {m[idx].conv.weight.shape}")

        # Head (layer 22 = Detect)
        detect = m[22]
        for s in range(3):
            # bbox branch: cv2[s] = Sequential(Conv, Conv, Conv2d)
            for layer in detect.cv2[s]:
                tc = write_conv(f, layer, tc)
            # cls branch: cv3[s] = Sequential(Conv, Conv, Conv2d)
            for layer in detect.cv3[s]:
                tc = write_conv(f, layer, tc)
            print(f"  head scale {s}: bbox={[l for l in detect.cv2[s]]}, cls={[l for l in detect.cv3[s]]}")

        # Patch num_tensors in header
        f.seek(12)
        f.write(struct.pack('<I', tc))

    file_size = open(args.output, 'rb').seek(0, 2)
    print(f"\nExported: {args.output} ({tc} tensors, {file_size} bytes)")


if __name__ == '__main__':
    main()
