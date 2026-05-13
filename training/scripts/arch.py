"""
MobileFaceNet architectures: 4 size variants for FaceX.

Reference: Chen et al. 2018, "MobileFaceNets: Efficient CNNs for Accurate
Real-Time Face Verification on Mobile Devices."

Standard MFN topology:
    Stem      : Conv 3x3 s=2, C=64
    DW        : DW Conv 3x3 s=1, C=64                     (one extra DW after stem)
    Stage 1   : 5x InvertedResidual (t=2, c=64, s=2 first)
    Stage 2   : 1x InvertedResidual (t=4, c=128, s=2)
    Stage 3   : 6x InvertedResidual (t=2, c=128, s=1)
    Stage 4   : 1x InvertedResidual (t=4, c=128, s=2)
    Stage 5   : 2x InvertedResidual (t=2, c=128, s=1)
    Conv 1x1  : 128 -> 512
    GDConv    : DW Conv 7x7 s=1 (no padding) -> 512x1x1     (Linear-GDC)
    Linear    : 1x1 conv 512 -> embedding_dim, no activation
    BN        : final BN over embedding (no L2 norm — InsightFace standard)

Param scaling for FaceX 4 budgets is via width multiplier (channels).
We use {0.25, 0.5, 1.0, 1.5} to get roughly {200K, 500K, 1M, 2M}.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class StageCfg:
    """One MFN inverted-residual stage."""
    t: int       # expansion ratio
    n: int       # number of blocks
    s: int       # stride of first block (rest are 1)


@dataclass
class FaceXArch:
    name: str
    width_mult: float          # multiplies all channel counts
    stem_channels: int         # base 64
    block_channels: int        # base 128 (conv channels in bottleneck stages)
    final_channels: int        # base 512 (after stage 5, before GDConv)
    embedding_dim: int         # output dim (we use 256 for nano, 512 for others)
    stages: List[StageCfg]
    gdc_kernel: int = 7        # GDConv kernel — equals feature map size after stages


def _round_ch(c: float) -> int:
    """Round channels to nearest 8 for tensor-core friendliness."""
    return max(8, int(round(c / 8) * 8))


def _make_arch(name: str, width: float, embedding_dim: int) -> FaceXArch:
    """Build a width-scaled MFN config matching the standard topology."""
    return FaceXArch(
        name=name,
        width_mult=width,
        stem_channels=_round_ch(64 * width),
        block_channels=_round_ch(128 * width),
        final_channels=_round_ch(512 * width),
        embedding_dim=embedding_dim,
        stages=[
            StageCfg(t=2, n=5, s=2),   # 7x7 -> after stem (s=2): 56 -> 28
            StageCfg(t=4, n=1, s=2),   # 28 -> 14
            StageCfg(t=2, n=6, s=1),   # 14 -> 14
            StageCfg(t=4, n=1, s=2),   # 14 -> 7
            StageCfg(t=2, n=2, s=1),   # 7 -> 7  (final feature map)
        ],
        gdc_kernel=7,  # feature map is 7x7 after stages
    )


# Width factors chosen so weights count lands near each target.
ARCH_NANO     = _make_arch("nano",     width=0.36, embedding_dim=256)
ARCH_TINY     = _make_arch("tiny",     width=0.55, embedding_dim=512)
ARCH_STANDARD = _make_arch("standard", width=0.90, embedding_dim=512)
ARCH_XS       = _make_arch("xs",       width=1.35, embedding_dim=512)


ALL_ARCHS = {
    "nano": ARCH_NANO,
    "tiny": ARCH_TINY,
    "standard": ARCH_STANDARD,
    "xs": ARCH_XS,
}


# ---------- Param counter ---------------------------------------------------


def _conv_bn_params(c_in: int, c_out: int, k: int, groups: int = 1) -> int:
    """Conv2d (no bias when followed by BN) + BN(weight, bias, running_mean, running_var)."""
    cw = c_in * (c_out // groups) * k * k
    bn = 4 * c_out  # weight + bias + running_mean + running_var
    return cw + bn


def _bottleneck_params(c_in: int, c_out: int, t: int, dw_k: int = 3) -> int:
    """One inverted residual block."""
    expand = t * c_in
    p = 0
    p += _conv_bn_params(c_in, expand, 1)            # 1x1 expand
    p += _conv_bn_params(expand, expand, dw_k, groups=expand)  # DW
    p += _conv_bn_params(expand, c_out, 1)            # 1x1 project (linear, no PReLU)
    # PReLU has 1 learnable per channel (or per-tensor); standard MFN uses per-channel
    p += expand + expand  # 2 PReLUs (after expand, after dw)
    return p


def count_params(arch: FaceXArch) -> dict:
    parts = {}
    p = 0
    # Stem: Conv 3x3 + BN + PReLU(per-channel)
    stem_p = _conv_bn_params(3, arch.stem_channels, 3) + arch.stem_channels
    parts["stem"] = stem_p; p += stem_p
    # DW after stem: DW 3x3 + BN + PReLU
    dw_p = _conv_bn_params(arch.stem_channels, arch.stem_channels, 3, groups=arch.stem_channels) + arch.stem_channels
    parts["dw_stem"] = dw_p; p += dw_p
    # Bottleneck stages
    c_in = arch.stem_channels
    for i, s in enumerate(arch.stages):
        c_out = arch.block_channels if i > 0 else arch.stem_channels  # stage 1 keeps stem_c
        # Actually MFN stage 1 outputs same as input (64). Stage 2+ outputs block_channels.
        if i == 0:
            c_out = arch.stem_channels
        else:
            c_out = arch.block_channels
        sp = 0
        for j in range(s.n):
            ci = c_in if j == 0 else c_out
            sp += _bottleneck_params(ci, c_out, s.t)
        parts[f"stage{i+1}"] = sp; p += sp
        c_in = c_out
    # Conv 1x1 -> final_channels
    final_p = _conv_bn_params(c_in, arch.final_channels, 1) + arch.final_channels  # PReLU
    parts["final_conv"] = final_p; p += final_p
    # GDConv DW kxk: just DW conv (no BN often, but we keep BN for stability)
    gdc_p = _conv_bn_params(arch.final_channels, arch.final_channels, arch.gdc_kernel, groups=arch.final_channels)
    parts["gdconv"] = gdc_p; p += gdc_p
    # Final linear: 1x1 conv to embedding_dim, BN
    out_p = _conv_bn_params(arch.final_channels, arch.embedding_dim, 1)
    parts["embed_proj"] = out_p; p += out_p
    parts["TOTAL"] = p
    return parts


def main():
    targets = {"nano": 200_000, "tiny": 500_000, "standard": 1_000_000, "xs": 2_000_000}
    print(f"{'arch':>10}  {'params':>10}  {'INT8 size':>12}  target")
    print("-" * 55)
    for name, arch in ALL_ARCHS.items():
        parts = count_params(arch)
        total = parts["TOTAL"]
        kb = total / 1024
        tgt = targets[name]
        delta = (total - tgt) / tgt * 100
        flag = "OK" if abs(delta) < 30 else "TUNE"
        print(f"{name:>10}  {total:>10,}  {kb:>10.1f}KB  target {tgt:>9,}  d{delta:+5.1f}%  {flag}")
        if name == "standard":
            print(f"    breakdown: {dict((k, f'{v:,}') for k, v in parts.items() if k != 'TOTAL')}")


if __name__ == "__main__":
    main()
