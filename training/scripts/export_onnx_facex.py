"""Export trained FaceX MobileFaceNet to ONNX with embedding output named 'embedding'."""

import argparse
from pathlib import Path
import torch

from model import MobileFaceNet
from arch import ALL_ARCHS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ALL_ARCHS), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net = MobileFaceNet(ALL_ARCHS[args.arch]).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    # strip module. prefix or any wrapping
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # also strip arcface head if present in checkpoint
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    net.load_state_dict(sd, strict=False)
    print(f"loaded {args.arch} from {args.ckpt}  emb_dim={net.arch.embedding_dim}")

    x = torch.zeros(1, 3, 112, 112)
    with torch.no_grad():
        y = net(x)
    print(f"  sanity: input {tuple(x.shape)}  output {tuple(y.shape)}  norm={y.norm(dim=-1).item():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net, x, args.out,
        input_names=["input"], output_names=["embedding"],
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    sz = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out}  ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
