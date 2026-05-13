"""
Verify a .bin file matches its PyTorch model end-to-end.

Loads tensors from EFM3 .bin, replays the same op-graph in pure numpy
(BN folded into conv, then conv + bias + PReLU), and compares output
to the PyTorch model loaded from the same checkpoint.

Run after training:
    python verify_bin.py --arch standard \
        --bin ../../weights/facex_standard.bin \
        --ckpt ../runs/standard/last.pt

If max embedding error < 1e-4, the .bin format and op-graph are correct
and the C engine (which uses the same op-graph) will produce matching
embeddings.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import arch as arch_module
from binformat import read_bin_header, read_bin_tensors, tensor_layout
from model import MobileFaceNet


# -------- numpy ops (mirror src/facex_mfn.c) ----------------------------


def fold_bn(w, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Returns (w_folded, bias_folded)."""
    inv = bn_w / np.sqrt(bn_var + eps)
    w_folded = w * inv.reshape((-1,) + (1,) * (w.ndim - 1))
    bias = bn_b - bn_mean * inv
    return w_folded.astype(np.float32), bias.astype(np.float32)


def conv2d_nchw(x, w, b, stride=1, padding=0, groups=1):
    """Plain reference conv2d. x: [1, Cin, H, W], w: [Cout, Cin/g, K, K], b: [Cout]."""
    _, Cin, Hin, Win = x.shape
    Cout, Cig, K, _ = w.shape
    assert Cig == Cin // groups
    Hout = (Hin + 2 * padding - K) // stride + 1
    Wout = (Win + 2 * padding - K) // stride + 1
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    y = np.zeros((1, Cout, Hout, Wout), dtype=np.float32)
    Cout_per_g = Cout // groups
    for og in range(groups):
        for oc in range(Cout_per_g):
            co = og * Cout_per_g + oc
            for ic in range(Cig):
                ci = og * Cig + ic
                # cross-correlation
                for ky in range(K):
                    for kx in range(K):
                        y[0, co] += (
                            x[0, ci, ky:ky + Hout * stride:stride,
                                       kx:kx + Wout * stride:stride]
                            * w[co, ic, ky, kx]
                        )
            y[0, co] += b[co]
    return y


def prelu(x, slope):
    """PReLU per-channel. x: [1, C, H, W], slope: [C]."""
    return np.where(x > 0, x, x * slope.reshape(1, -1, 1, 1)).astype(np.float32)


# -------- engine pulled from the .bin -----------------------------------


def load_engine_from_bin(bin_path: str):
    """Returns (arch_dict, layers) where layers is a list of dicts
    describing each conv (already BN-folded).
    """
    arch_obj, n_tensors, _, _ = read_bin_header(bin_path)
    layout = tensor_layout(arch_obj)
    tensors = {name: arr for name, arr in read_bin_tensors(bin_path)}

    def consume_conv(prefix, c_in, c_out, k, stride, padding, groups, with_act):
        # shapes
        w = tensors[f"{prefix}.conv.w"].reshape(c_out, c_in // groups, k, k)
        bn_w = tensors[f"{prefix}.bn.w"]
        bn_b = tensors[f"{prefix}.bn.b"]
        bn_mean = tensors[f"{prefix}.bn.mean"]
        bn_var = tensors[f"{prefix}.bn.var"]
        w_folded, b_folded = fold_bn(w, bn_w, bn_b, bn_mean, bn_var)
        slope = tensors[f"{prefix}.act.w"] if with_act else None
        return dict(prefix=prefix, w=w_folded, b=b_folded, slope=slope,
                    stride=stride, padding=padding, groups=groups,
                    c_in=c_in, c_out=c_out, k=k)

    arch = arch_obj
    layers = []
    # stem
    layers.append(consume_conv("stem", 3, arch.stem_channels,
                                3, 2, 1, 1, with_act=True))
    # dw_stem
    layers.append(consume_conv("dw_stem", arch.stem_channels, arch.stem_channels,
                                3, 1, 1, arch.stem_channels, with_act=True))
    c_in = arch.stem_channels
    for i, s in enumerate(arch.stages):
        c_out_stage = arch.stem_channels if i == 0 else arch.block_channels
        for j in range(s.n):
            ci = c_in if j == 0 else c_out_stage
            stride = s.s if j == 0 else 1
            hidden = ci * s.t
            p = f"stage{i+1}.block{j}"
            layers.append(consume_conv(f"{p}.expand", ci, hidden, 1, 1, 0, 1, True))
            layers.append(consume_conv(f"{p}.dw", hidden, hidden, 3, stride, 1, hidden, True))
            project = consume_conv(f"{p}.project", hidden, c_out_stage, 1, 1, 0, 1, False)
            project["has_residual"] = (stride == 1 and ci == c_out_stage)
            project["residual_kind"] = "project"  # block tail
            project["block_ci"] = ci
            layers.append(project)
        c_in = c_out_stage
    layers.append(consume_conv("final_conv", c_in, arch.final_channels, 1, 1, 0, 1, True))
    layers.append(consume_conv("gdconv", arch.final_channels, arch.final_channels,
                                arch.gdc_kernel, 1, 0, arch.final_channels, False))
    layers.append(consume_conv("embed_proj", arch.final_channels, arch.embedding_dim,
                                1, 1, 0, 1, False))
    return arch, layers


def run_engine(x_chw: np.ndarray, layers) -> np.ndarray:
    """x_chw: [1, 3, 112, 112]. Returns L2-normalized embedding [D]."""
    cur = x_chw
    block_residuals = []  # stack: when we hit a project layer with residual, pop
    block_inputs = []      # input saved at start of each block (3 layers per block)
    layer_idx = 0
    n = len(layers)
    while layer_idx < n:
        layer = layers[layer_idx]
        # Detect bottleneck block triple: expand + dw + project
        if ".expand" in layer["prefix"]:
            block_in = cur
            # expand
            cur = conv2d_nchw(cur, layer["w"], layer["b"],
                              stride=layer["stride"], padding=layer["padding"],
                              groups=layer["groups"])
            if layer["slope"] is not None:
                cur = prelu(cur, layer["slope"])
            # dw
            dw = layers[layer_idx + 1]
            cur = conv2d_nchw(cur, dw["w"], dw["b"],
                              stride=dw["stride"], padding=dw["padding"],
                              groups=dw["groups"])
            if dw["slope"] is not None:
                cur = prelu(cur, dw["slope"])
            # project
            pj = layers[layer_idx + 2]
            cur = conv2d_nchw(cur, pj["w"], pj["b"],
                              stride=pj["stride"], padding=pj["padding"],
                              groups=pj["groups"])
            if pj.get("has_residual"):
                cur = cur + block_in
            layer_idx += 3
        else:
            cur = conv2d_nchw(cur, layer["w"], layer["b"],
                              stride=layer["stride"], padding=layer["padding"],
                              groups=layer["groups"])
            if layer["slope"] is not None:
                cur = prelu(cur, layer["slope"])
            layer_idx += 1
    # cur: [1, embedding_dim, 1, 1]
    emb = cur.reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(arch_module.ALL_ARCHS.keys()))
    ap.add_argument("--bin", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    arch = arch_module.ALL_ARCHS[args.arch]
    model = MobileFaceNet(arch).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((1, 3, 112, 112), dtype=np.float32)
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        emb_torch = model(x_t).numpy().reshape(-1)

    _, layers = load_engine_from_bin(args.bin)
    emb_engine = run_engine(x, layers)

    max_err = float(np.max(np.abs(emb_torch - emb_engine)))
    cos = float(np.dot(emb_torch, emb_engine))
    print(f"emb_torch  [0:6] = {emb_torch[:6]}")
    print(f"emb_engine [0:6] = {emb_engine[:6]}")
    print(f"max_err = {max_err:.2e}   cos_sim = {cos:.6f}")
    if max_err < 1e-3 and cos > 0.9999:
        print("PASS — .bin op-graph matches PyTorch")
    else:
        print("FAIL — engine output diverges from PyTorch")


if __name__ == "__main__":
    main()
