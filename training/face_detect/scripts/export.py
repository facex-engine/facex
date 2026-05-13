"""Export trained face detector to ONNX. Output includes raw logits per scale.

Output names:
  cls_p3, box_p3, kps_p3       (1/8 stride)
  cls_p4, box_p4, kps_p4       (1/16 stride)
  cls_p5, box_p5, kps_p5       (1/32 stride)
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn

from model import FaceDetector


class ExportWrap(nn.Module):
    """Outputs as concatenated flat tensors per scale to simplify JS decode."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        (c3, b3, k3), (c4, b4, k4), (c5, b5, k5) = self.net(x)
        return c3, b3, k3, c4, b4, k4, c5, b5, k5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/apps/facex/training/face_detect/runs/best.pt")
    ap.add_argument("--out",  default="D:/apps/facex/wasm/facex_detect.onnx")
    args = ap.parse_args()

    net = FaceDetector().eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    print(f"loaded ckpt (ep {ck.get('epoch', '?')}, recall {ck.get('recall', '?'):.4f})")

    m = ExportWrap(net).eval()
    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        outs = m(x)
    for n, o in zip(["cls_p3","box_p3","kps_p3","cls_p4","box_p4","kps_p4","cls_p5","box_p5","kps_p5"], outs):
        print(f"  {n}: {tuple(o.shape)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m, x, args.out,
        input_names=["input"],
        output_names=["cls_p3","box_p3","kps_p3","cls_p4","box_p4","kps_p4","cls_p5","box_p5","kps_p5"],
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    sz = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out}  ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
