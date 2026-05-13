"""
FaceX architectures: 4 size variants of EdgeFace-shaped model.

Each variant keeps the same op-graph topology (Stem → 4 stages with
ConvNeXt+optional XCA → head with LoRA-decomposed FC) so a single
parametric C engine can run all of them.

The parametric .bin v2 format stores arch config in a header so
the engine knows widths/depths/kernels/ranks at load time.

Parameter budget targets (INT8 weights, ~1 byte/param):
    Nano     : 200 KB  →  ~200K params
    Tiny     : 500 KB  →  ~500K params
    Standard :   1 MB  →    ~1M params
    XS       : 1.8 MB  → ~1.77M params  (current EdgeFace-XS)
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class StageCfg:
    channels: int
    n_convnext: int
    dw_kernel: int
    rank: int          # LoRA rank inside MLP
    hidden: int        # MLP hidden dim
    has_xca: bool = False
    xca_splits: Tuple[int, ...] = ()   # channel splits for XCA cascaded DW
    xca_n_convs: int = 0               # how many of the splits get a DW conv
    xca_dw_kernel: int = 3
    xca_n_heads: int = 4
    xca_pos_embed: bool = False        # only stage 1 in current XS uses it
    xca_pos_hw: int = 0                # if pos_embed: H=W of cached pos const
    downsample_to: int = 0             # next stage channels (0 = no downsample)
    downsample_kernel: int = 2
    downsample_stride: int = 2


@dataclass
class FaceXArch:
    name: str
    stem_out: int
    stem_kernel: int
    stem_stride: int
    stages: List[StageCfg]
    head_rank: int          # LoRA rank for head FC1
    embedding_dim: int


# ---------- Param counter ---------------------------------------------------

def _convnext_params(C: int, K: int, r: int, hidden: int) -> int:
    """LoRA-style ConvNeXt block parameter count."""
    gamma = C
    dw = C * K * K + C
    ln = 2 * C
    # 2-stage LoRA MLP: (C→r→hidden) + bias(hidden) + GELU + (hidden→r→C) + bias(C) + gamma
    mlp = (C * r) + (r * hidden) + hidden + (hidden * r) + (r * C) + C
    return gamma + dw + ln + mlp


def _xca_params(C: int, r: int, hidden: int, splits: Tuple[int, ...],
                n_convs: int, dw_kernel: int, n_heads: int,
                pos_embed: bool, pos_hw: int) -> int:
    total = 0
    # cascaded DW convs (each on `splits[i]` channels)
    for i in range(n_convs):
        ci = splits[i]
        total += ci * dw_kernel * dw_kernel + ci
    # positional embedding (constant + 1x1 conv)
    if pos_embed and pos_hw > 0:
        total += C * pos_hw * pos_hw       # cached const NCHW
        total += C * C + C                  # 1x1 conv
    # LN before QKV
    total += 2 * C
    # QKV LoRA: C→r → r→3C + bias(3C)
    total += C * r + r * (3 * C) + 3 * C
    # temperature: n_heads
    total += n_heads
    # output projection LoRA: C→r → r→C + bias(C)
    total += C * r + r * C + C
    # post-attn LN
    total += 2 * C
    # MLP same shape as ConvNeXt MLP
    total += (C * r) + (r * hidden) + hidden + (hidden * r) + (r * C) + C
    # gamma_xca + gamma
    total += C + C
    return total


def _downsample_params(c_in: int, c_out: int, k: int) -> int:
    return 2 * c_in + c_in * c_out * k * k + c_out  # LN + conv + bias


def count_params(arch: FaceXArch) -> dict:
    parts = {}
    # stem: conv (in_c=3) + bias + LN(out)
    stem = 3 * arch.stem_out * arch.stem_kernel * arch.stem_kernel + arch.stem_out + 2 * arch.stem_out
    parts["stem"] = stem
    total = stem
    for i, s in enumerate(arch.stages):
        sp = 0
        sp += s.n_convnext * _convnext_params(s.channels, s.dw_kernel, s.rank, s.hidden)
        if s.has_xca:
            sp += _xca_params(s.channels, s.rank, s.hidden, s.xca_splits,
                              s.xca_n_convs, s.xca_dw_kernel, s.xca_n_heads,
                              s.xca_pos_embed, s.xca_pos_hw)
        if s.downsample_to:
            sp += _downsample_params(s.channels, s.downsample_to, s.downsample_kernel)
        parts[f"stage{i}"] = sp
        total += sp
    # head: LN(C_last) + FC(C_last → head_rank) + FC(head_rank → emb) + bias
    last_c = arch.stages[-1].channels
    head = 2 * last_c + last_c * arch.head_rank + arch.head_rank * arch.embedding_dim + arch.embedding_dim
    parts["head"] = head
    total += head
    parts["TOTAL"] = total
    return parts


# ---------- 4 architectures -------------------------------------------------

# XS: matches current edgeface_xs_fp32.bin (1.77M params, validated against
# C engine's mm_shapes table)
ARCH_XS = FaceXArch(
    name="xs",
    stem_out=32, stem_kernel=4, stem_stride=4,
    stages=[
        StageCfg(channels=32, n_convnext=3, dw_kernel=3, rank=19, hidden=128,
                 downsample_to=64),
        StageCfg(channels=64, n_convnext=2, dw_kernel=5, rank=38, hidden=256,
                 has_xca=True, xca_splits=(32, 32), xca_n_convs=1,
                 xca_pos_embed=True, xca_pos_hw=14,
                 downsample_to=100),
        StageCfg(channels=100, n_convnext=8, dw_kernel=7, rank=60, hidden=400,
                 has_xca=True, xca_splits=(34, 34, 32), xca_n_convs=2,
                 downsample_to=192),
        StageCfg(channels=192, n_convnext=2, dw_kernel=9, rank=115, hidden=768,
                 has_xca=True, xca_splits=(48, 48, 48, 48), xca_n_convs=3),
    ],
    head_rank=115, embedding_dim=512,
)

# Standard: ~1M params.
ARCH_STANDARD = FaceXArch(
    name="standard",
    stem_out=32, stem_kernel=4, stem_stride=4,
    stages=[
        StageCfg(channels=32, n_convnext=2, dw_kernel=3, rank=18, hidden=112,
                 downsample_to=64),
        StageCfg(channels=64, n_convnext=2, dw_kernel=5, rank=36, hidden=224,
                 has_xca=True, xca_splits=(32, 32), xca_n_convs=1,
                 xca_pos_embed=True, xca_pos_hw=14,
                 downsample_to=96),
        StageCfg(channels=96, n_convnext=5, dw_kernel=7, rank=56, hidden=384,
                 has_xca=True, xca_splits=(32, 32, 32), xca_n_convs=2,
                 downsample_to=176),
        StageCfg(channels=176, n_convnext=1, dw_kernel=9, rank=96, hidden=576,
                 has_xca=True, xca_splits=(44, 44, 44, 44), xca_n_convs=3),
    ],
    head_rank=96, embedding_dim=512,
)

# Tiny: ~500K.
ARCH_TINY = FaceXArch(
    name="tiny",
    stem_out=24, stem_kernel=4, stem_stride=4,
    stages=[
        StageCfg(channels=24, n_convnext=2, dw_kernel=3, rank=14, hidden=72,
                 downsample_to=48),
        StageCfg(channels=48, n_convnext=2, dw_kernel=5, rank=26, hidden=144,
                 downsample_to=80),
        StageCfg(channels=80, n_convnext=3, dw_kernel=7, rank=44, hidden=272,
                 has_xca=True, xca_splits=(40, 40), xca_n_convs=1,
                 downsample_to=136),
        StageCfg(channels=136, n_convnext=1, dw_kernel=7, rank=68, hidden=400,
                 has_xca=True, xca_splits=(34, 34, 34, 34), xca_n_convs=3),
    ],
    head_rank=72, embedding_dim=512,
)

# Nano: ~200K. Aggressive shrink.
ARCH_NANO = FaceXArch(
    name="nano",
    stem_out=16, stem_kernel=4, stem_stride=4,
    stages=[
        StageCfg(channels=16, n_convnext=1, dw_kernel=3, rank=10, hidden=56,
                 downsample_to=32),
        StageCfg(channels=32, n_convnext=2, dw_kernel=5, rank=20, hidden=112,
                 downsample_to=64),
        StageCfg(channels=64, n_convnext=3, dw_kernel=5, rank=32, hidden=192,
                 has_xca=True, xca_splits=(32, 32), xca_n_convs=1,
                 downsample_to=104),
        StageCfg(channels=104, n_convnext=1, dw_kernel=7, rank=48, hidden=288,
                 has_xca=False),
    ],
    head_rank=56, embedding_dim=256,
)


ALL_ARCHS = {
    "nano": ARCH_NANO,
    "tiny": ARCH_TINY,
    "standard": ARCH_STANDARD,
    "xs": ARCH_XS,
}


def main():
    print(f"{'arch':>10}  {'params':>10}  {'INT8 size':>12}  target")
    print("-" * 55)
    targets = {"nano": 200_000, "tiny": 500_000, "standard": 1_000_000, "xs": 1_770_000}
    for name, arch in ALL_ARCHS.items():
        parts = count_params(arch)
        total = parts["TOTAL"]
        size_kb = total / 1024
        tgt = targets[name]
        delta = (total - tgt) / tgt * 100
        flag = "OK" if abs(delta) < 8 else "TUNE"
        print(f"{name:>10}  {total:>10,}  {size_kb:>10.1f}KB  "
              f"target {tgt:>9,}  d{delta:+5.1f}%  {flag}")
        if name == "xs":
            print("    breakdown:", {k: f"{v:,}" for k, v in parts.items() if k != "TOTAL"})


if __name__ == "__main__":
    main()
