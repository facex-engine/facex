#!/bin/bash
# Build the MiniFASNet anti-spoof port that lives alongside YOLO in nn2.
set -e
CC=${CC:-/c/mingw64/bin/gcc}
FL="-O3 -march=native -funroll-loops -Wno-unused-function -Wno-unused-variable -Iinclude -Isrc"
LD="-lm -lsynchronization"

echo "Building nn2 anti-spoof (MiniFASNet port)"

# Core objects reused from YOLO build (compile if missing)
NEED="src/gemm.o src/gemm_res.o src/gemm_int8.o src/conv.o src/conv_implicit.o src/conv_fused.o src/conv_tiled.o src/winograd.o src/ops.o src/pool.o"
for o in $NEED; do
    if [ ! -f "$o" ]; then
        src=${o%.o}.c
        echo "  CC $src"
        $CC $FL -c "$src" -o "$o" || exit 1
    fi
done

# New objects
for f in src/antispoof_ops.c src/minifasnet.c src/main_antispoof.c; do
    echo "  CC $f"
    $CC $FL -c "$f" -o "${f%.c}.o" || exit 1
done

echo "  LD nn2_antispoof.exe"
$CC $FL -o nn2_antispoof.exe \
    src/antispoof_ops.o src/minifasnet.o src/main_antispoof.o \
    $NEED $LD || exit 1

echo "  → nn2_antispoof.exe ($(du -k nn2_antispoof.exe | cut -f1)KB)"
