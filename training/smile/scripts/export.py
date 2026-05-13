"""Export trained tongue classifier to ONNX with softmax baked in."""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SmileNet


class Wrap(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return F.softmax(self.net(x), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/apps/facex/training/smile/runs/best.pt")
    ap.add_argument("--out", default="D:/apps/facex/wasm/facex_smile.onnx")
    args = ap.parse_args()

    net = SmileNet().eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    print(f"loaded (ep {ck['epoch']}, F1={ck['f1']:.3f}, acc={ck['acc']:.3f})")

    m = Wrap(net).eval()
    x = torch.zeros(1, 3, 112, 112)
    with torch.no_grad(): y = m(x)
    print(f"sanity: out shape {tuple(y.shape)}, sum={y.sum():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, x, args.out, input_names=["input"], output_names=["prob"],
                      opset_version=13, do_constant_folding=True, dynamo=False)
    sz = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out}  ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
