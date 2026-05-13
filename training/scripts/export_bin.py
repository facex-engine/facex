"""
Export a trained MobileFaceNet checkpoint to FaceX EFM3 .bin format.

Usage:
    python export_bin.py --arch standard --ckpt runs/standard/last.pt \
                         --out ../weights/facex_standard.bin
"""

import argparse
from pathlib import Path

import torch

import arch as arch_module
from binformat import write_bin, tensor_layout
from model import MobileFaceNet


def collect_tensors(model: MobileFaceNet) -> dict:
    out = {}

    def add_conv_bn(prefix: str, conv_bn_module, with_act_module=None):
        out[f"{prefix}.conv.w"]   = conv_bn_module.conv.weight
        out[f"{prefix}.bn.w"]     = conv_bn_module.bn.weight
        out[f"{prefix}.bn.b"]     = conv_bn_module.bn.bias
        out[f"{prefix}.bn.mean"]  = conv_bn_module.bn.running_mean
        out[f"{prefix}.bn.var"]   = conv_bn_module.bn.running_var
        if with_act_module is not None:
            out[f"{prefix}.act.w"] = with_act_module.weight  # PReLU slope

    add_conv_bn("stem", model.stem, model.stem.act)
    add_conv_bn("dw_stem", model.dw_stem, model.dw_stem.act)

    arch = model.arch
    blk_idx = 0
    for i, s in enumerate(arch.stages):
        for j in range(s.n):
            blk = model.stages[blk_idx]; blk_idx += 1
            p = f"stage{i+1}.block{j}"
            add_conv_bn(f"{p}.expand",  blk.expand,  blk.expand.act)
            add_conv_bn(f"{p}.dw",      blk.dw,      blk.dw.act)
            add_conv_bn(f"{p}.project", blk.project, None)

    add_conv_bn("final_conv", model.final_conv, model.final_conv.act)
    add_conv_bn("gdconv",     model.gdconv,     None)
    add_conv_bn("embed_proj", model.embed_proj, None)

    # sanity
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
    model = MobileFaceNet(arch)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    model.load_state_dict(state)
    model.eval()

    tensors = collect_tensors(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_bytes = write_bin(args.out, arch, tensors)
    print(f"Wrote {args.out}  ({n_bytes/1024:.1f} KB FP32, "
          f"~{n_bytes/4/1024:.1f} KB INT8 equivalent)")


if __name__ == "__main__":
    main()
