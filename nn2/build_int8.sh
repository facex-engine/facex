#!/bin/bash
# Build INT8 NHWC pipeline (separate from main build)
CC=${CC:-/c/mingw64/bin/gcc}
FL="-O3 -march=native -funroll-loops -mavx512vnni -Wno-unused-function -Wno-unused-variable -Iinclude -Isrc"
LD="-lm -lsynchronization"

echo "Building INT8 NHWC Pipeline"
echo ""

# Core objects from main build (reuse)
echo "  Using existing core objects..."

CORE="src/gemm.o src/gemm_res.o src/gemm_int8.o src/conv.o src/conv_fused.o src/conv_implicit.o src/conv_tiled.o \
    src/winograd.o src/ops.o src/decode.o src/pool.o src/net.o src/nn2.o"
I8="src/int8_nhwc/gemm_vnni.o src/int8_nhwc/conv_int8.o src/int8_nhwc/ops_int8.o src/int8_nhwc/c2f_int8.o"

# INT8 NHWC objects
for f in src/int8_nhwc/gemm_vnni.c src/int8_nhwc/conv_int8.c src/int8_nhwc/ops_int8.c \
         src/int8_nhwc/c2f_int8.c src/int8_nhwc/net_int8.c src/int8_nhwc/forward_hybrid.c; do
    echo "  CC $f"
    $CC $FL -c $f -o ${f%.c}.o || exit 1
done

# Link per-layer benchmark
echo "  Linking nn2_int8_bench..."
$CC $FL -o nn2_int8_bench.exe $I8 src/int8_nhwc/net_int8.o $CORE $LD || exit 1

# Link hybrid pipeline benchmark
echo "  Linking nn2_hybrid..."
$CC $FL -o nn2_hybrid.exe $I8 src/int8_nhwc/forward_hybrid.o $CORE $LD || exit 1

echo "  → nn2_int8_bench.exe (per-layer INT8 benchmark)"
echo "  → nn2_hybrid.exe (full hybrid pipeline benchmark)"
echo ""
echo "Run: ./nn2_hybrid.exe [weights/yolov8n_320.bin]"
