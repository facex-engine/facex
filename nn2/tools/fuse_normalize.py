#!/usr/bin/env python3
"""Fuse 1/255 normalization into first conv weights of nn2 .bin model."""
import sys, struct, numpy as np

def fuse(inp, out):
    with open(inp, 'rb') as f:
        magic = f.read(4)
        assert magic == b'NN2W', f"Bad magic: {magic}"
        nc, isz, nt = struct.unpack('III', f.read(12))
        print(f"Model: nc={nc}, input={isz}, tensors={nt}")

        header = magic + struct.pack('III', nc, isz, nt)
        tensors = []
        for i in range(nt):
            sz = struct.unpack('I', f.read(4))[0]
            data = np.frombuffer(f.read(sz * 4), dtype=np.float32).copy()
            tensors.append(data)

    print(f"Loaded {len(tensors)} tensors")
    # First tensor = conv0 weights [16, 27] (16 filters, 3*3*3)
    # Second tensor = conv0 bias [16]
    w = tensors[0]
    print(f"conv0 weights: {w.shape[0]} values, range [{w.min():.4f}, {w.max():.4f}]")
    tensors[0] = w / 255.0
    print(f"After /255:   range [{tensors[0].min():.6f}, {tensors[0].max():.6f}]")

    with open(out, 'wb') as f:
        f.write(header)
        for t in tensors:
            f.write(struct.pack('I', len(t)))
            f.write(t.astype(np.float32).tobytes())
    print(f"Saved {out}")

if __name__ == '__main__':
    fuse(sys.argv[1], sys.argv[2])
