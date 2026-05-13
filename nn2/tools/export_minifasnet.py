#!/usr/bin/env python3
"""
export_minifasnet.py — Convert MinivisionAI Silent-Face MiniFASNet PyTorch
weights (.pth) to nn2 binary format (.bin) for the antispoof port.

Layer write order MUST match `src/minifasnet.c` loader. Each Conv_block
folds its BatchNorm into [weight, bias]; PReLU.weight is appended when
the block uses activation. Project conv has no activation. Final Linear
folds BN1d into [w, b].
"""

import argparse, struct, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def fold_bn_conv(conv, bn):
    """Fold BN(running_mean, running_var, weight, bias, eps) into conv weights/bias."""
    with torch.no_grad():
        w = conv.weight.detach().clone()
        if conv.bias is not None:
            b = conv.bias.detach().clone()
        else:
            b = torch.zeros(w.shape[0], device=w.device)
        mean = bn.running_mean.detach()
        var = bn.running_var.detach()
        gamma = bn.weight.detach()
        beta = bn.bias.detach()
        eps = bn.eps
        std = torch.sqrt(var + eps)
        scale = gamma / std
        if w.dim() == 4:
            w_new = w * scale.reshape(-1, 1, 1, 1)
        else:
            w_new = w * scale.reshape(-1, 1)
        b_new = (b - mean) * scale + beta
        return w_new.cpu().numpy(), b_new.cpu().numpy()


def fold_bn1d_linear(lin, bn1d):
    """Fold BN1d into Linear: y = (W*x+b - μ)/√(v+ε)*γ + β"""
    with torch.no_grad():
        W = lin.weight.detach().clone()
        if lin.bias is not None:
            b = lin.bias.detach().clone()
        else:
            b = torch.zeros(W.shape[0], device=W.device)
        mean = bn1d.running_mean.detach()
        var = bn1d.running_var.detach()
        gamma = bn1d.weight.detach()
        beta = bn1d.bias.detach()
        eps = bn1d.eps
        std = torch.sqrt(var + eps)
        scale = gamma / std
        W_new = W * scale.reshape(-1, 1)
        b_new = (b - mean) * scale + beta
        return W_new.cpu().numpy(), b_new.cpu().numpy()


def write_tensor(f, arr, tag=""):
    flat = np.ascontiguousarray(arr).astype(np.float32).ravel()
    f.write(struct.pack("<I", len(flat)))
    f.write(flat.tobytes())
    return len(flat)


def write_conv_block(f, cb, has_prelu, tag):
    """Conv_block: conv + bn + prelu(optional). Writes [w, b, prelu_alpha?]."""
    w, b = fold_bn_conv(cb.conv, cb.bn)
    write_tensor(f, w, f"{tag}.w")
    write_tensor(f, b, f"{tag}.b")
    if has_prelu:
        write_tensor(f, cb.prelu.weight.detach().cpu().numpy(), f"{tag}.prelu")


def write_linear_block(f, cb, tag):
    """Linear_block: conv + bn, NO activation."""
    w, b = fold_bn_conv(cb.conv, cb.bn)
    write_tensor(f, w, f"{tag}.w")
    write_tensor(f, b, f"{tag}.b")


def write_dw_block(f, dw, has_se, tag):
    """Depth_Wise: expand (Conv_block) + dw (Conv_block) + project (Linear_block) + optional SE."""
    write_conv_block(f, dw.conv, True, f"{tag}.expand")
    write_conv_block(f, dw.conv_dw, True, f"{tag}.dw")
    write_linear_block(f, dw.project, f"{tag}.project")
    if has_se:
        se = dw.se_module
        # fc1: Conv2d 1x1 + BN1 + ReLU + Conv2d 1x1 + BN2 + sigmoid; fuse BN into bias
        # Treat the Conv2d like a 1x1 conv with BN — same fold_bn_conv works.
        w1, b1 = fold_bn_conv(se.fc1, se.bn1)
        w2, b2 = fold_bn_conv(se.fc2, se.bn2)
        # The fc1/fc2 are Conv2d 1x1; weights shape is [out, in, 1, 1] → write as [out*in].
        write_tensor(f, w1, f"{tag}.se.fc1.w")
        write_tensor(f, b1, f"{tag}.se.fc1.b")
        write_tensor(f, w2, f"{tag}.se.fc2.w")
        write_tensor(f, b2, f"{tag}.se.fc2.b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="D:/apps/facex/training/Silent-Face-Anti-Spoofing",
                    help="path to MinivisionAI repo (for src.model_lib.MiniFASNet)")
    ap.add_argument("--ckpt", required=True, help="path to .pth")
    ap.add_argument("--variant", choices=["v2", "v1se"], required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE

    ctor = MiniFASNetV2 if args.variant == "v2" else MiniFASNetV1SE
    variant_id = 0 if args.variant == "v2" else 1
    has_se = (args.variant == "v1se")

    net = ctor(conv6_kernel=(5, 5)).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # strip module. prefix if needed
    if next(iter(sd)).startswith("module."):
        sd = OrderedDict((k[7:], v) for k, v in sd.items())
    net.load_state_dict(sd)
    print(f"loaded {args.variant} from {args.ckpt}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(b"MFN1")
        f.write(struct.pack("<III", variant_id, 80, 0))   # num_tensors placeholder
        tensors_before = f.tell()

        # === Backbone Conv_blocks ===
        write_conv_block(f, net.conv1,    True, "conv1")
        write_conv_block(f, net.conv2_dw, True, "conv2_dw")

        # === Depth_Wise / Residual stages ===
        # conv_23
        write_dw_block(f, net.conv_23, False, "conv_23")
        # conv_3 = Residual of 4 Depth_Wise (last has SE if V1SE)
        for i, dw in enumerate(net.conv_3.model):
            write_dw_block(f, dw, has_se and i == len(net.conv_3.model) - 1, f"conv_3[{i}]")
        write_dw_block(f, net.conv_34, False, "conv_34")
        for i, dw in enumerate(net.conv_4.model):
            write_dw_block(f, dw, has_se and i == len(net.conv_4.model) - 1, f"conv_4[{i}]")
        write_dw_block(f, net.conv_45, False, "conv_45")
        for i, dw in enumerate(net.conv_5.model):
            write_dw_block(f, dw, has_se and i == len(net.conv_5.model) - 1, f"conv_5[{i}]")

        # === Head ===
        write_conv_block(f, net.conv_6_sep, True, "conv_6_sep")
        write_linear_block(f, net.conv_6_dw, "conv_6_dw")

        # Linear 512→128 + BN1d (folded), then Linear 128→3 (no bias)
        w_lin, b_lin = fold_bn1d_linear(net.linear, net.bn)
        write_tensor(f, w_lin, "linear.w")
        write_tensor(f, b_lin, "linear.b")
        # prob: 128→3, no bias
        write_tensor(f, net.prob.weight.detach().cpu().numpy(), "prob.w")

        # Patch num_tensors: count how many [size, data] records were written
        end = f.tell()
        # We don't track tensor count individually; rewind and count.
        # Simpler: read back by scanning. But to avoid that, track tensor writes inline.

    # Re-read to count tensors and patch
    with open(args.output, "rb") as f:
        f.seek(16)
        n = 0
        while True:
            head = f.read(4)
            if len(head) < 4: break
            sz = struct.unpack("<I", head)[0]
            f.seek(sz * 4, 1)
            n += 1
    with open(args.output, "r+b") as f:
        f.seek(12)
        f.write(struct.pack("<I", n))
    size = Path(args.output).stat().st_size
    print(f"wrote {args.output}  {n} tensors  {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
