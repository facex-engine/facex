"""
End-to-end smoke test for MobileFaceNet:
  model -> .bin -> reload -> verify embeddings match.
"""

import sys
import tempfile
from pathlib import Path

import torch

import arch as arch_module
from binformat import write_bin, read_bin_tensors, tensor_layout
from export_bin import collect_tensors
from model import MobileFaceNet


def reload_into_fresh_model(arch, bin_path: str) -> MobileFaceNet:
    model = MobileFaceNet(arch)
    tensors = collect_tensors(model)
    seen = set()
    for name, arr in read_bin_tensors(bin_path):
        target = tensors[name]
        if target.shape.numel() != arr.size:
            raise SystemExit(f"shape mismatch for {name}: target {tuple(target.shape)} vs arr {arr.size}")
        with torch.no_grad():
            target.copy_(torch.from_numpy(arr.copy()).reshape(target.shape))
        seen.add(name)
    expected = set(tensor_layout(arch))
    missing = expected - seen
    if missing:
        raise SystemExit(f"missing tensors after reload: {missing}")
    return model


def run_arch(arch_name: str):
    arch = arch_module.ALL_ARCHS[arch_name]
    print(f"--- {arch_name} ({arch_module.count_params(arch)['TOTAL']:,} params) ---")
    torch.manual_seed(0)
    model = MobileFaceNet(arch).eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        emb_a = model(x)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        bin_path = tf.name
    try:
        tensors = collect_tensors(model)
        n_bytes = write_bin(bin_path, arch, tensors)
        print(f"  wrote {n_bytes/1024:.1f} KB")
        model2 = reload_into_fresh_model(arch, bin_path).eval()
        with torch.no_grad():
            emb_b = model2(x)
        max_err = (emb_a - emb_b).abs().max().item()
        cos = torch.cosine_similarity(emb_a, emb_b, dim=-1).mean().item()
        ok = max_err < 1e-5
        print(f"  reload max_err={max_err:.2e}  cos_sim={cos:.6f}  {'OK' if ok else 'FAIL'}")
        return ok
    finally:
        Path(bin_path).unlink(missing_ok=True)


def main():
    targets = sys.argv[1:] or list(arch_module.ALL_ARCHS.keys())
    all_ok = True
    for n in targets:
        all_ok &= run_arch(n)
    print("\nAll OK" if all_ok else "\nSOMETHING FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
