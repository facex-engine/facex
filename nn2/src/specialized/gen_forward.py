#!/usr/bin/env python3
"""
Generate a compile-time specialized forward pass for YOLOv8n@320.

Every conv layer gets hardcoded dimensions, no runtime dispatch.
GEMM calls use exact tile counts, skip all branches.
"""

S = 320
MR = 6
NR = 32

# YOLOv8n architecture: (name, cin, cout, k, stride, pad, act, H_in, W_in)
layers = []

def add_conv(name, cin, cout, k, stride, pad, act, H, W):
    Hout = (H + 2*pad - k) // stride + 1
    Wout = (W + 2*pad - k) // stride + 1
    spatial = Hout * Wout
    K = cin * k * k
    n_tiles = (spatial + NR - 1) // NR
    m_panels = (cout + MR - 1) // MR
    flops = 2 * cout * K * spatial
    layers.append({
        'name': name, 'cin': cin, 'cout': cout, 'k': k,
        'stride': stride, 'pad': pad, 'act': act,
        'H': H, 'W': W, 'Hout': Hout, 'Wout': Wout,
        'spatial': spatial, 'K': K,
        'n_tiles': n_tiles, 'm_panels': m_panels, 'flops': flops
    })
    return Hout, Wout

def add_c2f(name, cin, cout, n_bn, shortcut, H, W):
    c = cout // 2
    add_conv(f'{name}.cv1', cin, 2*c, 1, 1, 0, 1, H, W)
    for i in range(n_bn):
        add_conv(f'{name}.bn{i}.cv1', c, c, 3, 1, 1, 1, H, W)
        add_conv(f'{name}.bn{i}.cv2', c, c, 3, 1, 1, 1, H, W)
    total_c = (2 + n_bn) * c
    add_conv(f'{name}.cv2', total_c, cout, 1, 1, 0, 1, H, W)

h, w = S, S

# Backbone
h0, w0 = add_conv('bb_conv0', 3, 16, 3, 2, 1, 1, h, w)
h1, w1 = add_conv('bb_conv1', 16, 32, 3, 2, 1, 1, h0, w0)
add_c2f('bb_c2f0', 32, 32, 1, 1, h1, w1)
h2, w2 = add_conv('bb_conv2', 32, 64, 3, 2, 1, 1, h1, w1)
add_c2f('bb_c2f1', 64, 64, 2, 1, h2, w2)
h3, w3 = add_conv('bb_conv3', 64, 128, 3, 2, 1, 1, h2, w2)
add_c2f('bb_c2f2', 128, 128, 2, 1, h3, w3)
h4, w4 = add_conv('bb_conv4', 128, 256, 3, 2, 1, 1, h3, w3)
add_c2f('bb_c2f3', 256, 256, 1, 1, h4, w4)

# SPPF
add_conv('sppf_cv1', 256, 128, 1, 1, 0, 1, h4, w4)
add_conv('sppf_cv2', 512, 256, 1, 1, 0, 1, h4, w4)

# Neck
add_c2f('nk_c2f0', 384, 128, 1, 0, h3, w3)
add_c2f('nk_c2f1', 192, 64, 1, 0, h2, w2)
add_conv('nk_conv0', 64, 64, 3, 2, 1, 1, h2, w2)
add_c2f('nk_c2f2', 192, 128, 1, 0, h3, w3)
add_conv('nk_conv1', 128, 128, 3, 2, 1, 1, h3, w3)
add_c2f('nk_c2f3', 384, 256, 1, 0, h4, w4)

# Head
head_ch = [64, 128, 256]
feat_h = [h2, h3, h4]
feat_w = [w2, w3, w4]
for s in range(3):
    fh, fw = feat_h[s], feat_w[s]
    add_conv(f'head{s}_bbox0', head_ch[s], 64, 3, 1, 1, 1, fh, fw)
    add_conv(f'head{s}_bbox1', 64, 64, 3, 1, 1, 1, fh, fw)
    add_conv(f'head{s}_bbox2', 64, 64, 1, 1, 0, 0, fh, fw)
    add_conv(f'head{s}_cls0', head_ch[s], 64, 3, 1, 1, 1, fh, fw)
    add_conv(f'head{s}_cls1', 64, 64, 3, 1, 1, 1, fh, fw)
    add_conv(f'head{s}_cls2', 64, 80, 1, 1, 0, 0, fh, fw)

# Print summary
print(f"YOLOv8n@{S}: {len(layers)} conv layers")
total_flops = sum(l['flops'] for l in layers)
print(f"Total FLOPs: {total_flops/1e6:.1f}M")
print(f"Total dispatches (1 GEMM + im2col each): ~{len(layers)*2}")
print()

# Categorize by size
small = [l for l in layers if l['flops'] < 1_000_000]
medium = [l for l in layers if 1_000_000 <= l['flops'] < 10_000_000]
large = [l for l in layers if l['flops'] >= 10_000_000]
print(f"Small (<1M FLOPs): {len(small)} layers — single-thread candidates")
print(f"Medium (1-10M): {len(medium)} layers")
print(f"Large (>10M): {len(large)} layers — main compute")
print()

# Show layers that should be single-threaded
print("Layers to run single-threaded (< 500K FLOPs):")
for l in layers:
    if l['flops'] < 500_000:
        print(f"  {l['name']:30s} {l['cout']:3d}x{l['K']:4d}x{l['spatial']:5d} = {l['flops']/1e3:.0f}K FLOPs")

print()
print("Top 10 heaviest layers:")
for l in sorted(layers, key=lambda x: -x['flops'])[:10]:
    print(f"  {l['name']:30s} {l['cout']:3d}x{l['K']:4d}x{l['spatial']:5d} = {l['flops']/1e6:.1f}M FLOPs, {l['n_tiles']} tiles")
