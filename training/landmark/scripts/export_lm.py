"""Export trained landmark model to EFL3 .bin."""

import argparse
from pathlib import Path

import torch

from binformat_lm import write_bin, tensor_layout
from model import LandmarkNet


def collect_tensors(model: LandmarkNet) -> dict:
    out = {}

    def add_conv_bn(p, mod, with_act):
        out[f"{p}.conv.w"] = mod.conv.weight
        out[f"{p}.bn.w"]   = mod.bn.weight
        out[f"{p}.bn.b"]   = mod.bn.bias
        out[f"{p}.bn.mean"] = mod.bn.running_mean
        out[f"{p}.bn.var"]  = mod.bn.running_var
        if with_act:
            out[f"{p}.act.w"] = mod.act.weight

    add_conv_bn("stem", model.stem, True)
    for i, blk in enumerate(model.blocks):
        p = f"block{i}"
        add_conv_bn(f"{p}.expand", blk.expand, True)
        add_conv_bn(f"{p}.dw", blk.dw, True)
        add_conv_bn(f"{p}.project", blk.project, False)
    add_conv_bn("final", model.final, True)
    add_conv_bn("gdc", model.gdc, False)
    out["fc.w"] = model.fc.weight
    out["fc.b"] = model.fc.bias
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/apps/facex/training/landmark/runs/best.pt")
    ap.add_argument("--out", default="D:/apps/facex/weights/facex_landmark.bin")
    args = ap.parse_args()

    model = LandmarkNet().eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"Loaded checkpoint (NME={ck.get('nme', '?'):.4f})")

    tensors = collect_tensors(model)
    stages = model.DEFAULT_STAGES
    n_bytes = write_bin(
        args.out, stages,
        stem_out=32, final_c=256, fc_in=256, n_landmarks=98,
        gdc_kernel=7, stem_kernel=3, stem_stride=2,
        tensors_by_name=tensors,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {args.out}  ({n_bytes/1024:.1f} KB FP32, "
          f"~{n_bytes/4/1024:.1f} KB INT8)")


if __name__ == "__main__":
    main()
