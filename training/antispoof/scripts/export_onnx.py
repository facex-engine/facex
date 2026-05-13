"""Export trained anti-spoof model to ONNX for browser use.

Input:  float32 [1, 3, 112, 112] normalized as (rgb - 127.5) / 128
Output: float32 [1, 2] softmax probabilities [P(spoof), P(live)]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AntiSpoofNet


class AntiSpoofExport(nn.Module):
    """Wrapper that applies softmax to logits."""
    def __init__(self, net: AntiSpoofNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/apps/facex/training/antispoof/runs/best.pt")
    ap.add_argument("--out",  default="D:/apps/facex/wasm/facex_antispoof.onnx")
    args = ap.parse_args()

    net = AntiSpoofNet().eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    acc = ck.get("acc", None)
    print(f"loaded ckpt (ep {ck.get('epoch','?')}, acc={acc*100 if acc else '?':.2f}%)")

    model = AntiSpoofExport(net).eval()

    dummy = torch.zeros(1, 3, 112, 112, dtype=torch.float32)
    out = model(dummy)
    print(f"sanity output shape: {tuple(out.shape)}  sum={out.sum().item():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["prob"],
        dynamic_axes=None,
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,                      # use legacy exporter → single inlined .onnx
    )
    sz = Path(args.out).stat().st_size
    print(f"wrote {args.out}  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
