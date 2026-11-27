#!/bin/bash
# Build release archives for Linux x86-64.
# Run in WSL or on Linux.
# Output: deploy/facex-linux-x64.tar.gz
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)
VERSION="2.0.0"
OUT="$ROOT/deploy/facex-${VERSION}-linux-x64"

echo "=== Building FaceX ${VERSION} release ==="

rm -rf "$OUT"
mkdir -p "$OUT/lib" "$OUT/include" "$OUT/bin"

# 1. Build static library
echo "[1/4] Building libfacex.a..."
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static -DFACEX_LIB \
    -c src/facex.c -o /tmp/fx.o \
    -Iinclude
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static \
    -c src/transformer_ops.c -o /tmp/tops.o
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static \
    -c src/gemm_int8_4x8c8.c -o /tmp/gemm.o
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static \
    -c src/threadpool.c -o /tmp/tp.o
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static \
    -c src/weight_crypto.c -o /tmp/wc.o
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static \
    -c src/detect.c -o /tmp/det.o -Iinclude
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static \
    -c src/align.c -o /tmp/align.o
ar rcs "$OUT/lib/libfacex.a" /tmp/fx.o /tmp/tops.o /tmp/gemm.o /tmp/tp.o /tmp/wc.o /tmp/det.o /tmp/align.o
echo "   libfacex.a: $(stat -c%s "$OUT/lib/libfacex.a") bytes"

# 2. Build CLI binary
echo "[2/4] Building facex-cli..."
make -B cli CFLAGS="-O3 -march=x86-64-v3 -funroll-loops" LDFLAGS="-static -lm -lpthread"
cp facex-cli "$OUT/bin/facex-cli"
echo "   facex-cli: $(stat -c%s "$OUT/bin/facex-cli") bytes"

# 3. Build encrypt tool
echo "[3/4] Building facex-encrypt..."
gcc -O3 -march=x86-64-v3 -static -DWEIGHT_CRYPTO_MAIN \
    -o "$OUT/bin/facex-encrypt" src/weight_crypto.c -lm
echo "   facex-encrypt: $(stat -c%s "$OUT/bin/facex-encrypt") bytes"

# 4. Copy headers + docs
echo "[4/4] Packaging..."
cp include/facex.h include/detect.h "$OUT/include/"
cp LICENSE README.md "$OUT/"

# Create archive
cd deploy
tar czf "facex-${VERSION}-linux-x64.tar.gz" "facex-${VERSION}-linux-x64/"
SIZE=$(stat -c%s "facex-${VERSION}-linux-x64.tar.gz")
echo ""
echo "=== Done ==="
echo "Archive: deploy/facex-${VERSION}-linux-x64.tar.gz ($((SIZE/1024)) KB)"
echo ""
echo "Contents:"
echo "  lib/libfacex.a     — static library"
echo "  include/facex.h    — unified API"
echo "  include/detect.h   — detector API"
echo "  bin/facex-cli      — CLI tool"
echo "  bin/facex-encrypt   — weight encryption"
