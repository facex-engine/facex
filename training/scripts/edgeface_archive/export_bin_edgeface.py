"""
Export a trained PyTorch checkpoint to FaceX .bin v2 format.

Usage:
    python export_bin.py --arch nano --ckpt runs/nano/best.pt --out weights/nano.bin
"""

import argparse
from pathlib import Path

import torch

import arch as arch_module
from binformat import write_bin, tensor_layout
from model import FaceXModel


def collect_tensors(model: FaceXModel) -> dict:
    """Map every parameter to the layout name expected by binformat.tensor_layout."""
    arch = model.arch
    out = {}

    out["stem.conv.w"] = model.stem.weight
    out["stem.conv.b"] = model.stem.bias
    out["stem.ln.w"]   = model.stem_ln.weight
    out["stem.ln.b"]   = model.stem_ln.bias

    for i, s in enumerate(arch.stages):
        blocks = list(model.stage_blocks[i])
        # convnext blocks first
        for j in range(s.n_convnext):
            blk = blocks[j]
            p = f"stage{i}.block{j}"
            out[f"{p}.dw.w"] = blk.dw.weight
            out[f"{p}.dw.b"] = blk.dw.bias
            out[f"{p}.ln.w"] = blk.ln.weight
            out[f"{p}.ln.b"] = blk.ln.bias
            out[f"{p}.mlp.fc0.w"] = blk.mlp.fc0.weight
            out[f"{p}.mlp.fc1.w"] = blk.mlp.fc1.weight
            out[f"{p}.mlp.fc1.b"] = blk.mlp.fc1.bias
            out[f"{p}.mlp.fc2.w"] = blk.mlp.fc2.weight
            out[f"{p}.mlp.fc3.w"] = blk.mlp.fc3.weight
            out[f"{p}.mlp.fc3.b"] = blk.mlp.fc3.bias
            out[f"{p}.gamma"]     = blk.gamma
        if s.has_xca:
            xca = blocks[s.n_convnext]
            xp = f"stage{i}.xca"
            for k in range(s.xca_n_convs):
                conv = xca.dw.convs[k]
                out[f"{xp}.dw{k}.w"] = conv.weight
                out[f"{xp}.dw{k}.b"] = conv.bias
            if s.xca_pos_embed:
                out[f"{xp}.pos_const"]   = xca.pos_const
                out[f"{xp}.pos_conv.w"]  = xca.pos_conv.weight
                out[f"{xp}.pos_conv.b"]  = xca.pos_conv.bias
            out[f"{xp}.ln_attn.w"] = xca.ln_attn.weight
            out[f"{xp}.ln_attn.b"] = xca.ln_attn.bias
            out[f"{xp}.qkv.a.w"]   = xca.qkv_lora_a.weight
            out[f"{xp}.qkv.b.w"]   = xca.qkv_lora_b.weight
            out[f"{xp}.qkv.b.b"]   = xca.qkv_lora_b.bias
            out[f"{xp}.temperature"] = xca.temperature
            out[f"{xp}.proj.a.w"]  = xca.proj_lora_a.weight
            out[f"{xp}.proj.b.w"]  = xca.proj_lora_b.weight
            out[f"{xp}.proj.b.b"]  = xca.proj_lora_b.bias
            out[f"{xp}.gamma_xca"] = xca.gamma_xca
            out[f"{xp}.ln_mlp.w"]  = xca.ln_mlp.weight
            out[f"{xp}.ln_mlp.b"]  = xca.ln_mlp.bias
            out[f"{xp}.mlp.fc0.w"] = xca.mlp.fc0.weight
            out[f"{xp}.mlp.fc1.w"] = xca.mlp.fc1.weight
            out[f"{xp}.mlp.fc1.b"] = xca.mlp.fc1.bias
            out[f"{xp}.mlp.fc2.w"] = xca.mlp.fc2.weight
            out[f"{xp}.mlp.fc3.w"] = xca.mlp.fc3.weight
            out[f"{xp}.mlp.fc3.b"] = xca.mlp.fc3.bias
            out[f"{xp}.gamma"]     = xca.gamma
        if s.downsample_to:
            ds = model.downsamples[i]
            out[f"stage{i}.downsample.ln.w"]   = ds.ln.weight
            out[f"stage{i}.downsample.ln.b"]   = ds.ln.bias
            out[f"stage{i}.downsample.conv.w"] = ds.conv.weight
            out[f"stage{i}.downsample.conv.b"] = ds.conv.bias

    out["head.ln.w"]  = model.head_ln.weight
    out["head.ln.b"]  = model.head_ln.bias
    out["head.fc1.w"] = model.head_fc1.weight
    out["head.fc2.w"] = model.head_fc2.weight
    out["head.fc2.b"] = model.head_fc2.bias

    # sanity: every layout name must be present
    expected = set(tensor_layout(arch))
    missing = expected - set(out.keys())
    extra = set(out.keys()) - expected
    if missing or extra:
        raise SystemExit(f"layout mismatch. missing={missing}  extra={extra}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(arch_module.ALL_ARCHS.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arch = arch_module.ALL_ARCHS[args.arch]
    model = FaceXModel(arch)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)  # checkpoint may be {"model": sd, ...} or raw sd
    model.load_state_dict(state)
    model.eval()

    tensors = collect_tensors(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_bytes = write_bin(args.out, arch, tensors)
    print(f"Wrote {args.out}  ({n_bytes/1024:.1f} KB FP32, "
          f"~{n_bytes/4/1024:.1f} KB INT8 equivalent)")


if __name__ == "__main__":
    main()
