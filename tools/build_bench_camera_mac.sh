#!/usr/bin/env bash
# Build the macOS camera benchmark (Swift + libfacex.a).
#
# Prereq:  `make` from repo root has produced libfacex.a.
# Output:  ./facex-camera-bench (in repo root).
#
# Modes (BUILD env var or first positional arg):
#   release   — swiftc -O           (default; what `make bench-camera` runs)
#   debug     — swiftc -Onone -g    (with debug symbols, suitable for lldb)
#   profile   — swiftc -O  -g       (release optimisation + symbols, for Instruments)

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD="${BUILD:-${1:-release}}"
case "$BUILD" in
    release) SWIFT_FLAGS=(-O)         ;;
    debug)   SWIFT_FLAGS=(-Onone -g)  ;;
    profile) SWIFT_FLAGS=(-O -g)      ;;
    *) echo "unknown BUILD=$BUILD (use release|debug|profile)" >&2; exit 1 ;;
esac

if [[ ! -f libfacex.a ]]; then
    echo "libfacex.a missing — running make first" >&2
    make
fi

# Detect optional libfacex contents and link the matching frameworks.
# This avoids "Undefined _cblas_sgemm" if the user previously built
# with ACCELERATE=1 (etc.).
EXTRA_FRAMEWORKS=()
if nm libfacex.a 2>/dev/null | grep -q '_matmul_fp32_packed_accelerate'; then
    EXTRA_FRAMEWORKS+=(-framework Accelerate)
fi
if nm libfacex.a 2>/dev/null | grep -q '_facex_coreml_init'; then
    EXTRA_FRAMEWORKS+=(-framework CoreML)
fi

# Bridging header so Swift sees facex.h directly.
BRIDGE_HEADER="$(mktemp -t facex_bridge_XXXX.h)"
trap 'rm -f "$BRIDGE_HEADER"' EXIT
cat > "$BRIDGE_HEADER" <<'EOF'
#include "facex.h"
EOF

swiftc "${SWIFT_FLAGS[@]}" \
    -import-objc-header "$BRIDGE_HEADER" \
    -I include \
    tools/bench_camera_mac.swift \
    -L . -lfacex \
    -framework AVFoundation \
    -framework CoreMedia \
    -framework CoreVideo \
    -framework CoreImage \
    -framework Foundation \
    ${EXTRA_FRAMEWORKS[@]+"${EXTRA_FRAMEWORKS[@]}"} \
    -o facex-camera-bench

echo "built: ./facex-camera-bench  (mode: $BUILD)"
echo "run:   ./facex-camera-bench --frames 200"
