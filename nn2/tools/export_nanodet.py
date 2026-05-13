#!/usr/bin/env python3
"""
export_nanodet.py — Export NanoDet-Plus-m ONNX weights to nn2 binary format.

Reads the ONNX model graph, extracts all Conv weights (with fused BN),
and writes them sequentially in nn2 format.

Weight file format:
    [4]  magic "NN2N"  (NanoDet)
    [4]  uint32 num_classes (80)
    [4]  uint32 input_size (320)
    [4]  uint32 num_conv_layers
    For each conv layer:
        [4]  uint32 cout
        [4]  uint32 cin  (cin_per_group for DW)
        [4]  uint32 k
        [4]  uint32 stride
        [4]  uint32 pad
        [4]  uint32 groups (1=regular, cout=depthwise)
        [4]  uint32 act (0=none, 3=LeakyReLU)
        [cout*4] float32 bias
        [cout*cin_per_group*k*k*4] float32 weights
"""

import struct
import numpy as np
import onnx


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='weights/nanodet_plus_m_320.onnx')
    parser.add_argument('--output', default='weights/nanodet_plus_m_320.bin')
    parser.add_argument('--size', type=int, default=320)
    args = parser.parse_args()

    model = onnx.load(args.onnx)

    # Build initializer lookup
    inits = {}
    for init in model.graph.initializer:
        from onnx.numpy_helper import to_array
        inits[init.name] = to_array(init).copy()

    # Extract conv layers in order
    convs = []
    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue

        attrs = {a.name: a for a in node.attribute}
        group = attrs['group'].i if 'group' in attrs else 1
        ks = list(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else [1, 1]
        stride = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
        pad = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]

        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None

        w = inits[w_name]
        b = inits[b_name] if b_name and b_name in inits else np.zeros(w.shape[0], dtype=np.float32)

        cout = w.shape[0]
        cin_per_g = w.shape[1]

        # Determine activation from the next node
        act = 0  # default none
        # Find output name and check if next node is LeakyRelu
        out_name = node.output[0]
        for next_node in model.graph.node:
            if next_node.op_type == 'LeakyRelu' and out_name in next_node.input:
                act = 3  # LeakyReLU
                break

        convs.append({
            'cout': cout, 'cin_per_g': cin_per_g,
            'k': ks[0], 's': stride[0], 'p': pad[0],
            'group': group, 'act': act,
            'w': w, 'b': b,
            'name': w_name
        })

    nc = 80  # COCO classes
    print(f'NanoDet-Plus-m: {len(convs)} conv layers, input={args.size}')

    with open(args.output, 'wb') as f:
        f.write(b'NN2N')
        f.write(struct.pack('<III', nc, args.size, len(convs)))

        for i, c in enumerate(convs):
            f.write(struct.pack('<IIIIIII',
                c['cout'], c['cin_per_g'], c['k'], c['s'], c['p'],
                c['group'], c['act']))
            c['b'].astype(np.float32).tofile(f)
            c['w'].astype(np.float32).tofile(f)

            dw = ' [DW]' if c['group'] > 1 else ''
            act_str = ' LeakyReLU' if c['act'] == 3 else ''
            if i < 10 or i >= len(convs) - 5 or i % 20 == 0:
                print(f'  {i:3d} {c["cout"]:4d}x{c["cin_per_g"]:4d} k={c["k"]} s={c["s"]} g={c["group"]:3d}{dw}{act_str}')

    fsize = open(args.output, 'rb').seek(0, 2)
    print(f'\nExported: {args.output} ({len(convs)} layers, {fsize:,} bytes, {fsize/1024/1024:.1f} MB)')


if __name__ == '__main__':
    main()
