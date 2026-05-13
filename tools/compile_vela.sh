#!/usr/bin/env bash
# compile_vela.sh — Run Arm's Vela compiler on an INT8 tflite to produce
# an Ethos-U65 command stream for i.MX 93 / 95.
#
# Vela docs:    https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/
# Install:      pip install ethos-u-vela
#
# Output is `<basename>_vela.tflite` in the same directory — a tflite file
# that contains the Ethos-U custom operator alongside the original model
# graph. Loading this with the standard TFLite C API + the Arm Ethos-U
# external delegate dispatches the heavy ops to the NPU; anything Vela
# refused stays on the CPU side.
#
# Usage:
#   tools/compile_vela.sh weights/edgeface_xs_int8.tflite
#   tools/compile_vela.sh weights/yunet_int8.tflite ethos-u65-256
#
# Accelerator config defaults to ethos-u65-256 (i.MX 93 / 95). Other valid
# options Vela understands: ethos-u65-512, ethos-u55-128, ethos-u55-256.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.tflite> [accel-config]" >&2
    echo "  accel-config defaults to ethos-u65-256 (i.MX 93 / 95)" >&2
    exit 1
fi

INPUT="$1"
ACCEL="${2:-ethos-u65-256}"

if [[ ! -f "$INPUT" ]]; then
    echo "error: $INPUT not found" >&2
    exit 1
fi

if ! command -v vela >/dev/null 2>&1; then
    echo "error: vela not on PATH — install with: pip install ethos-u-vela" >&2
    exit 1
fi

OUTDIR="$(dirname "$INPUT")"
echo "compiling $INPUT for $ACCEL → $OUTDIR/"

vela \
    --accelerator-config "$ACCEL" \
    --system-config Ethos_U65_High_End \
    --memory-mode Shared_Sram \
    --output-dir "$OUTDIR" \
    "$INPUT"

# Vela emits <basename>_vela.tflite + a summary CSV. Print the summary so
# the user can sanity-check op coverage (anything not on the NPU stays on CPU).
SUM=$(ls "$OUTDIR"/*summary*.csv 2>/dev/null | head -n1 || true)
if [[ -n "$SUM" ]]; then
    echo "---- vela summary ($SUM) ----"
    head -3 "$SUM"
fi

OUT="$OUTDIR/$(basename "${INPUT%.tflite}")_vela.tflite"
if [[ -f "$OUT" ]]; then
    echo "ok: $OUT ($(wc -c <"$OUT") bytes)"
else
    echo "warn: expected $OUT but it wasn't produced — check vela log above" >&2
    exit 2
fi
