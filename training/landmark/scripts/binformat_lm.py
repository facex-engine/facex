"""
FaceX EFL3 .bin format — landmark variant.

Same structure as EFM3 (MobileFaceNet) but the head is a single FC layer
producing 196 raw outputs (98 landmarks as x,y), not L2-normalized.

Layout:
    "EFL3"          (4 bytes)
    version u32 = 3
    binary header (32 bytes, see below)
    json_len u32 + JSON
    n_tensors u32
    [u32 size + FP32 bytes] x n_tensors

Binary header (32 bytes, fixed):
    u8  hdr_version      (=1)
    u8  n_stages
    u8  stem_kernel
    u8  stem_stride
    u16 stem_out
    u16 final_channels
    u16 n_landmarks      (=98 for WFLW)
    u16 fc_in            (channels feeding final FC, =256 here)
    u8  gdc_kernel
    u8  pad[3]
    u32 reserved[4]
    per-stage records (8 bytes each, up to MAX_STAGES=8 — same as EFM3):
        u8 expand, u8 n_blocks, u8 stride, u8 pad, u32 reserved
"""

import json
import struct

MAX_STAGES = 8
STAGE_REC_SIZE = 8
HDR_FIXED_SIZE = 32
ARCH_HDR_SIZE = HDR_FIXED_SIZE + MAX_STAGES * STAGE_REC_SIZE   # 96

_HDR_FIXED = struct.Struct("<BBBBHHHHB3sIIII")    # 32 bytes
_STAGE_REC = struct.Struct("<BBBBI")              # 8 bytes


def tensor_layout(stages):
    """Walk model in forward order, name each weight."""
    def conv_bn(p, with_act):
        out = [f"{p}.conv.w",
               f"{p}.bn.w", f"{p}.bn.b",
               f"{p}.bn.mean", f"{p}.bn.var"]
        if with_act:
            out.append(f"{p}.act.w")
        return out

    names = []
    names += conv_bn("stem", with_act=True)
    blk_idx = 0
    for (t, c, n, s) in stages:
        for _ in range(n):
            p = f"block{blk_idx}"
            names += conv_bn(f"{p}.expand", True)
            names += conv_bn(f"{p}.dw", True)
            names += conv_bn(f"{p}.project", False)
            blk_idx += 1
    names += conv_bn("final", True)
    names += conv_bn("gdc", False)
    names += ["fc.w", "fc.b"]
    return names


def write_bin(path: str, stages, stem_out, final_c, fc_in, n_landmarks,
              gdc_kernel, stem_kernel, stem_stride, tensors_by_name) -> int:
    layout = tensor_layout(stages)
    arch_json = json.dumps({
        "head": "fc_landmark",
        "n_landmarks": n_landmarks,
        "stem_out": stem_out, "final_channels": final_c,
        "fc_in": fc_in, "gdc_kernel": gdc_kernel,
        "stages": [{"t": t, "c": c, "n": n, "s": s} for (t, c, n, s) in stages],
    }, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf += b"EFL3"
    buf += struct.pack("<I", 3)
    buf += _HDR_FIXED.pack(
        1, len(stages), stem_kernel, stem_stride,
        stem_out, final_c, n_landmarks, fc_in,
        gdc_kernel, b"\x00\x00\x00",
        0, 0, 0, 0,
    )
    for (t, c, n, s) in stages:
        buf += _STAGE_REC.pack(t, n, s, 0, c)  # encode channels in the reserved u32 (per-stage out_c)
    # pad stages
    while len(buf) - 8 < ARCH_HDR_SIZE:
        buf += _STAGE_REC.pack(0, 0, 0, 0, 0)
    buf += struct.pack("<I", len(arch_json))
    buf += arch_json
    buf += struct.pack("<I", len(layout))
    for name in layout:
        t = tensors_by_name[name]
        blob = t.detach().cpu().contiguous().numpy().astype("float32").tobytes()
        buf += struct.pack("<I", len(blob))
        buf += blob
    with open(path, "wb") as f:
        f.write(buf)
    import os
    return os.path.getsize(path)
