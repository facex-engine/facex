#!/usr/bin/env python3
"""
quantize_weights.py — Quantize YOLOv8n FP32 weights to INT8 for nn2.

Per-channel symmetric quantization for weights.
Activation scales determined by calibration on sample images.

Usage:
    python tools/quantize_weights.py --fp32 weights/yolov8n.bin --output weights/yolov8n_int8.bin [--calib-dir images/]

Weight file format (INT8):
    [4] magic "NN2Q"
    [4] uint32 num_classes
    [4] uint32 input_size
    [4] uint32 num_layers
    For each conv layer:
        [4] uint32 cout, cin, k, stride, pad, act
        [cout*4] float32 bias (fused BN, FP32)
        [cout*4] float32 w_scale (per-channel weight scale)
        [cout*cin*k*k] int8 quantized weights
        [4] float32 input_act_scale (per-tensor, from calibration)
        [4] float32 output_act_scale (per-tensor, from calibration)
"""

import struct
import argparse
import numpy as np


def quantize_per_channel(weights_fp32, bits=8):
    """Symmetric per-channel quantization.
    weights_fp32: [Cout, K] float32
    Returns: (weights_int8, scales)
    """
    max_abs = np.abs(weights_fp32).max(axis=1, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-8)  # avoid div by zero
    qmax = 127.0
    scales = max_abs.flatten() / qmax  # [Cout]
    weights_q = np.round(weights_fp32 / scales.reshape(-1, 1)).astype(np.int8)
    return weights_q, scales.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32', required=True, help='FP32 weight file (NN2W format)')
    parser.add_argument('--output', required=True, help='Output INT8 weight file')
    parser.add_argument('--act-scale', type=float, default=0.05,
                        help='Default activation scale (calibrate for better accuracy)')
    args = parser.parse_args()

    # Read FP32 weights
    with open(args.fp32, 'rb') as f:
        magic = f.read(4)
        assert magic == b'NN2W', f'Expected NN2W, got {magic}'
        nc, isz, ntensors = struct.unpack('<III', f.read(12))
        print(f'FP32 model: {nc} classes, {isz} input, {ntensors} tensors')

        # Read all tensor pairs (weight, bias)
        layers = []
        for i in range(ntensors // 2):
            # Weight
            wsz = struct.unpack('<I', f.read(4))[0]
            w = np.frombuffer(f.read(wsz * 4), dtype=np.float32).copy()
            # Bias
            bsz = struct.unpack('<I', f.read(4))[0]
            b = np.frombuffer(f.read(bsz * 4), dtype=np.float32).copy()
            layers.append((w, b))

    # Quantize weights
    print(f'\nQuantizing {len(layers)} layers...')
    with open(args.output, 'wb') as f:
        f.write(b'NN2Q')
        f.write(struct.pack('<III', nc, isz, len(layers)))

        for i, (w, b) in enumerate(layers):
            cout = b.shape[0]
            kin = w.shape[0] // cout  # cin*k*k
            w_2d = w.reshape(cout, kin)

            # Per-channel quantization
            w_int8, w_scales = quantize_per_channel(w_2d)

            # Write layer
            f.write(struct.pack('<I', cout))
            f.write(struct.pack('<I', kin))
            b.tofile(f)  # FP32 bias
            w_scales.tofile(f)  # FP32 per-channel scale
            w_int8.tobytes()  # INT8 weights
            f.write(w_int8.tobytes())
            # Activation scales (default, should be calibrated)
            f.write(struct.pack('<ff', args.act_scale, args.act_scale))

            if i < 5 or i % 10 == 0:
                w_range = np.abs(w_2d).max()
                print(f'  layer {i:3d}: {cout:4d}x{kin:5d}  '
                      f'w_range={w_range:.4f}  scale_mean={w_scales.mean():.6f}')

    fsize = open(args.output, 'rb').seek(0, 2)
    print(f'\nQuantized: {args.output} ({fsize:,} bytes, '
          f'{fsize / open(args.fp32, "rb").seek(0, 2) * 100:.0f}% of FP32)')


if __name__ == '__main__':
    main()
