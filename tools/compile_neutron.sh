#!/usr/bin/env bash
# compile_neutron.sh — Run NXP's neutron-converter on an INT8 tflite to
# produce a Neutron-specialised .tflite for the i.MX 95 eIQ Neutron N3 NPU.
#
# neutron-converter ships with NXP's eIQ Toolkit. See README at the top of
# this script (or docs/imx_npu.md §1) for how to obtain it.
#
# Output is `<basename>_neutron.tflite` in the same directory — a tflite
# that contains Neutron custom subgraph nodes alongside the original model
# graph. Loading it with the standard TFLite C API + libneutron_delegate.so
# offloads the matched ops to the NPU; anything the converter rejected
# stays on the CPU side (XNNPACK).
#
# Usage:
#   tools/compile_neutron.sh weights/edgeface_xs_int8.tflite
#   tools/compile_neutron.sh weights/yunet_int8.tflite imx95
#
# Target defaults to imx95 (the only Neutron N3 SoC today).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.tflite> [target]" >&2
    echo "  target defaults to imx95 (eIQ Neutron N3)" >&2
    exit 1
fi

INPUT="$1"
TARGET="${2:-imx95}"

if [[ ! -f "$INPUT" ]]; then
    echo "error: $INPUT not found" >&2
    exit 1
fi

# neutron-converter is the binary name in eIQ Toolkit ≥ 1.x. Older betas
# called it `neutron_converter` — accept both.
CONV=""
for cand in neutron-converter neutron_converter; do
    if command -v "$cand" >/dev/null 2>&1; then
        CONV="$cand"
        break
    fi
done

if [[ -z "$CONV" ]]; then
    echo "error: neutron-converter not on PATH" >&2
    echo "       install NXP eIQ Toolkit and source its env script;" >&2
    echo "       see docs/imx_npu.md §1 for the download link." >&2
    exit 1
fi

OUTDIR="$(dirname "$INPUT")"
BASE="$(basename "${INPUT%.tflite}")"
OUT="$OUTDIR/${BASE}_neutron.tflite"

echo "compiling $INPUT for $TARGET → $OUT"

# Flag names track eIQ Toolkit ≥ 1.10. Older releases used --input/--output
# instead of positional args; if your install rejects this invocation, run
# `$CONV --help` and adjust. We pin --target so the artefact is built for
# the right NPU revision.
"$CONV" \
    --target "$TARGET" \
    --output "$OUT" \
    "$INPUT"

if [[ ! -f "$OUT" ]]; then
    echo "warn: expected $OUT but it wasn't produced — check converter log above" >&2
    exit 2
fi

echo "ok: $OUT ($(wc -c <"$OUT") bytes)"

# neutron-converter typically prints op-coverage ("X/Y ops mapped to NPU")
# inline. Anything left on the CPU side runs via XNNPACK at runtime —
# common culprits are unsupported activations (GELU), dynamic shapes, and
# ops needing FP32. Decompose / replace, re-quantise, re-convert.
