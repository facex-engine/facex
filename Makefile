# FaceX — Fast face embedding library
# 3ms inference, 7MB binary, zero dependencies
#
# Build:  make
# Test:   make example && ./facex-example
#         make test     (golden embedding test)
#         make bench    (synthetic latency benchmark)
#
# Auto-detects host arch:
#   x86-64 Linux/macOS/Windows  → AVX2 (+ AVX-512 + VNNI when present)
#   arm64 macOS / Linux         → NEON kernels, links gemm_stub +
#                                  threadpool_pthread, defines FACEX_NO_INT8

CC ?= cc
LDFLAGS = -lm -lpthread

UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

ifeq ($(OS),Windows_NT)
  ARCH := x86_64
else ifneq (,$(filter arm64 aarch64,$(UNAME_M)))
  ARCH := arm64
else
  ARCH := x86_64
endif

ifeq ($(ARCH),arm64)
  # Apple Silicon / generic AArch64
  CFLAGS = -O3 -funroll-loops -DFACEX_NO_INT8
  ifeq ($(UNAME_S),Darwin)
    CFLAGS += -mcpu=apple-m1
  else
    CFLAGS += -march=armv8-a+simd
  endif
  GEMM_SRC = src/gemm_stub.c
  THREADPOOL_SRC = src/threadpool_pthread.c
  # Opt-in SME path: `make SME=1` adds the M4+ Streaming-Matrix-Extension
  # kernel + runtime detection. Default build stays portable (works on
  # older Xcode / M1-M3 / non-Apple aarch64 boards). Requires Apple Clang
  # 16+ (Xcode 16+) or upstream Clang 18+ for the ACLE 2024 SME intrinsics.
  #
  # IMPORTANT: -march=armv9-a+sme must NOT be applied to the dispatcher
  # source (transformer_ops.c) — doing so lets clang auto-vectorize plain
  # C using SVE/SME instructions that trap on M1/M2/M3. SME flags are
  # applied per-file to transformer_ops_sme.c only (see SME_FLAGS below).
  ifeq ($(SME),1)
    CFLAGS  += -DFACEX_HAVE_SME
    SME_FLAGS = -march=armv9-a+sme
    SME_SRCS = src/transformer_ops_sme.c src/cpu_features.c
  else
    SME_FLAGS =
    SME_SRCS =
  endif
else
  # x86-64 path (original)
  CFLAGS = -O3 -march=native -mfma -funroll-loops
  AVX512 := $(shell $(CC) -mavx512f -dM -E - < /dev/null 2>/dev/null | grep -c AVX512F)
  ifeq ($(AVX512),1)
    CFLAGS += -mavx512f -mavx512vnni -mprefer-vector-width=512
  endif
  GEMM_SRC = src/gemm_int8_4x8c8.c
  ifeq ($(UNAME_S),Darwin)
    THREADPOOL_SRC = src/threadpool_pthread.c
  else
    THREADPOOL_SRC = src/threadpool.c
  endif
  SME_SRCS =
  SME_FLAGS =
endif

ifeq ($(OS),Windows_NT)
  LDFLAGS += -lsynchronization
  EXT = .exe
endif

# Opt-in Apple Accelerate.framework path. Adds an AMX-backed matmul
# dispatch via cblas_sgemm; falls back to NEON / AVX2 for the shapes
# Accelerate refuses (tiny M*K*N) and disables itself at startup if
# its self-check disagrees with a scalar reference.
ifeq ($(ACCELERATE),1)
  ifneq ($(UNAME_S),Darwin)
    $(error ACCELERATE=1 requires macOS — Accelerate.framework is Apple-only)
  endif
  CFLAGS  += -DFACEX_HAVE_ACCELERATE
  LDFLAGS += -framework Accelerate
  ACC_SRCS = src/backend_accelerate.c
else
  ACC_SRCS =
endif

# Opt-in Core ML / Apple Neural Engine path. Builds the Objective-C
# bridge in src/backend_coreml.m and links CoreML.framework. The bridge
# loads a precompiled `.mlpackage` (produced by tools/export_coreml.py
# from an EdgeFace ONNX) and routes prediction to the ANE.
ifeq ($(COREML),1)
  ifneq ($(UNAME_S),Darwin)
    $(error COREML=1 requires macOS — Core ML is Apple-only)
  endif
  CFLAGS  += -DFACEX_HAVE_COREML
  COREML_FLAGS = -fobjc-arc
  LDFLAGS += -framework CoreML -framework Foundation
  COREML_SRCS = src/backend_coreml.m
else
  COREML_FLAGS =
  COREML_SRCS =
endif

SRCS = src/facex.c src/transformer_ops.c $(GEMM_SRC) $(THREADPOOL_SRC) $(SME_SRCS) $(ACC_SRCS) $(COREML_SRCS)

.PHONY: all clean example lib cli encrypt test mac-test bench detect-lib \
        bench-camera bench-camera-debug bench-camera-profile \
        mac-sme mac-universal mac-universal-arm64 mac-universal-x86_64 \
        imx-npu imx93 imx95 imx8mp

all: lib cli detect-lib

# Static library
lib: libfacex.a

SME_OBJS =
ifeq ($(SME),1)
  SME_OBJS = transformer_ops_sme.o cpu_features.o
endif
ACC_OBJS =
ifeq ($(ACCELERATE),1)
  ACC_OBJS = backend_accelerate.o
endif
COREML_OBJS =
ifeq ($(COREML),1)
  COREML_OBJS = backend_coreml.o
endif

libfacex.a: $(SRCS) src/detect.c src/align.c src/weight_crypto.c
	$(CC) $(CFLAGS) -Iinclude -DFACEX_LIB -c src/facex.c -o facex.o
	$(CC) $(CFLAGS) -Iinclude -c src/transformer_ops.c -o transformer_ops.o
	$(CC) $(CFLAGS) -Iinclude -c $(GEMM_SRC) -o gemm.o
	$(CC) $(CFLAGS) -Iinclude -c $(THREADPOOL_SRC) -o threadpool.o
	$(CC) $(CFLAGS) -Iinclude -c src/detect.c -o detect.o
	$(CC) $(CFLAGS) -Iinclude -c src/align.c -o align.o
	$(CC) $(CFLAGS) -Iinclude -c src/weight_crypto.c -o weight_crypto.o
ifeq ($(SME),1)
	$(CC) $(CFLAGS) -Iinclude -c src/cpu_features.c -o cpu_features.o
	$(CC) $(CFLAGS) $(SME_FLAGS) -Iinclude -c src/transformer_ops_sme.c -o transformer_ops_sme.o
endif
ifeq ($(ACCELERATE),1)
	$(CC) $(CFLAGS) -Iinclude -c src/backend_accelerate.c -o backend_accelerate.o
endif
ifeq ($(COREML),1)
	$(CC) $(CFLAGS) $(COREML_FLAGS) -Iinclude -c src/backend_coreml.m -o backend_coreml.o
endif
	ar rcs $@ facex.o transformer_ops.o gemm.o threadpool.o detect.o align.o weight_crypto.o $(SME_OBJS) $(ACC_OBJS) $(COREML_OBJS)
	@rm -f facex.o transformer_ops.o gemm.o threadpool.o detect.o align.o weight_crypto.o $(SME_OBJS) $(ACC_OBJS) $(COREML_OBJS)
	@echo "Built libfacex.a ($(ARCH)$(if $(filter 1,$(SME)), +SME,)$(if $(filter 1,$(ACCELERATE)), +Accelerate,)$(if $(filter 1,$(COREML)), +CoreML,))"

# Standalone CLI (for Go subprocess / testing)
cli: facex-cli$(EXT)

facex-cli$(EXT): src/edgeface_engine.c src/transformer_ops.c $(GEMM_SRC) $(THREADPOOL_SRC) src/weight_crypto.c src/detect.c src/align.c
	$(CC) $(CFLAGS) -Iinclude -c src/edgeface_engine.c   -o cli_engine.o
	$(CC) $(CFLAGS) -Iinclude -c src/transformer_ops.c   -o cli_ops.o
	$(CC) $(CFLAGS) -Iinclude -c $(GEMM_SRC)             -o cli_gemm.o
	$(CC) $(CFLAGS) -Iinclude -c $(THREADPOOL_SRC)       -o cli_tp.o
	$(CC) $(CFLAGS) -Iinclude -c src/weight_crypto.c     -o cli_wc.o
	$(CC) $(CFLAGS) -Iinclude -c src/detect.c            -o cli_det.o
	$(CC) $(CFLAGS) -Iinclude -c src/align.c             -o cli_align.o
ifeq ($(SME),1)
	$(CC) $(CFLAGS) -Iinclude -c src/cpu_features.c                      -o cli_cpuf.o
	$(CC) $(CFLAGS) $(SME_FLAGS) -Iinclude -c src/transformer_ops_sme.c  -o cli_sme.o
endif
ifeq ($(ACCELERATE),1)
	$(CC) $(CFLAGS) -Iinclude -c src/backend_accelerate.c                -o cli_acc.o
endif
ifeq ($(COREML),1)
	$(CC) $(CFLAGS) $(COREML_FLAGS) -Iinclude -c src/backend_coreml.m    -o cli_coreml.o
endif
	$(CC) $(CFLAGS) -Iinclude -o $@ \
	    cli_engine.o cli_ops.o cli_gemm.o cli_tp.o cli_wc.o cli_det.o cli_align.o \
	    $(if $(filter 1,$(SME)),cli_cpuf.o cli_sme.o,) \
	    $(if $(filter 1,$(ACCELERATE)),cli_acc.o,) \
	    $(if $(filter 1,$(COREML)),cli_coreml.o,) \
	    $(LDFLAGS)
	@rm -f cli_*.o
	@echo "Built facex-cli$(EXT) ($(ARCH)$(if $(filter 1,$(SME)), +SME,)$(if $(filter 1,$(ACCELERATE)), +Accelerate,)$(if $(filter 1,$(COREML)), +CoreML,))"

# Example program
example: facex-example$(EXT)

facex-example$(EXT): examples/example.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Encryption tool
encrypt: facex-encrypt$(EXT)

facex-encrypt$(EXT): src/weight_crypto.c
	$(CC) $(CFLAGS) -DWEIGHT_CRYPTO_MAIN -o $@ $< $(LDFLAGS)

# Golden test
test: golden-test$(EXT)
	@echo "Running golden test..."
	@./golden-test$(EXT) data/edgeface_xs_fp32.bin

golden-test$(EXT): tests/golden_test.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# macOS arm64 smoke test (also runs on x86-64 macOS / Linux)
mac-test: facex-mac-test$(EXT)
	@./facex-mac-test$(EXT)

facex-mac-test$(EXT): tests/test_mac.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Convenience: `make mac-sme` builds the M4+ SME-enabled libfacex.a + cli.
mac-sme:
	@$(MAKE) clean
	@$(MAKE) SME=1

# Universal Mac dylib (arm64 + x86_64) for distribution. Each slice is
# built by re-invoking make with target-specific flags, then `lipo`
# combines them.
mac-universal:
	@if [ "$(UNAME_S)" != "Darwin" ]; then \
	    echo "mac-universal is macOS-only" ; exit 1 ; fi
	@$(MAKE) clean
	@$(MAKE) mac-universal-arm64
	@cp libfacex.a /tmp/libfacex-mac-arm64.a
	@$(MAKE) clean
	@$(MAKE) mac-universal-x86_64
	@cp libfacex.a /tmp/libfacex-mac-x86_64.a
	@lipo -create /tmp/libfacex-mac-arm64.a /tmp/libfacex-mac-x86_64.a \
	      -output libfacex-universal.a
	@rm -f /tmp/libfacex-mac-arm64.a /tmp/libfacex-mac-x86_64.a
	@echo "Built libfacex-universal.a:"
	@lipo -info libfacex-universal.a

mac-universal-arm64:
	$(MAKE) ARCH=arm64 \
	        CFLAGS="-O3 -funroll-loops -DFACEX_NO_INT8 -arch arm64 -mmacosx-version-min=11.0" \
	        GEMM_SRC=src/gemm_stub.c \
	        THREADPOOL_SRC=src/threadpool_pthread.c \
	        SME_SRCS= ACC_SRCS= COREML_SRCS= SME_OBJS= ACC_OBJS= COREML_OBJS= SME_FLAGS= COREML_FLAGS= \
	        libfacex.a

mac-universal-x86_64:
	$(MAKE) ARCH=x86_64 \
	        CFLAGS="-O3 -funroll-loops -mfma -arch x86_64 -mmacosx-version-min=11.0 -mavx2" \
	        GEMM_SRC=src/gemm_int8_4x8c8.c \
	        THREADPOOL_SRC=src/threadpool_pthread.c \
	        SME_SRCS= ACC_SRCS= COREML_SRCS= SME_OBJS= ACC_OBJS= COREML_OBJS= SME_FLAGS= COREML_FLAGS= \
	        libfacex.a

# Unified latency bench. Same source / same output schema across every
# build flavour — see scripts/bench_all.sh for the sweep that produces
# a single comparison table.
bench: facex-bench$(EXT)

facex-bench$(EXT): tools/bench.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# macOS camera benchmark (Swift, AVFoundation). Requires Xcode CLT swiftc.
# The build script handles the swiftc invocation + bridging header.
bench-camera: libfacex.a
	@command -v swiftc >/dev/null || { echo "swiftc not found — install Xcode Command Line Tools"; exit 1; }
	@bash tools/build_bench_camera_mac.sh release

bench-camera-debug: libfacex.a
	@command -v swiftc >/dev/null || { echo "swiftc not found — install Xcode Command Line Tools"; exit 1; }
	@bash tools/build_bench_camera_mac.sh debug

bench-camera-profile: libfacex.a
	@command -v swiftc >/dev/null || { echo "swiftc not found — install Xcode Command Line Tools"; exit 1; }
	@bash tools/build_bench_camera_mac.sh profile

# Detector static library
detect-lib: libdetect.a

libdetect.a: src/detect.c include/detect.h
	$(CC) $(CFLAGS) -Iinclude -c src/detect.c -o detect.o
	ar rcs $@ detect.o
	@rm -f detect.o
	@echo "Built libdetect.a"

clean:
	rm -f libfacex.a libfacex-arm64.a libfacex-x86_64.a libfacex-universal.a \
	      libdetect.a libfacex_npu.so libfacex_npu.dylib \
	      facex-cli$(EXT) facex-example$(EXT) facex-encrypt$(EXT) \
	      golden-test$(EXT) facex-mac-test$(EXT) facex-bench$(EXT) facex-camera-bench \
	      imx_npu_compile_test facex-bench-npu *.o

# ---------------------------------------------------------------------------
# i.MX NPU build (TFLite C API + runtime-loaded delegate).
#
#   make imx-npu                    # host-side dev build (links host libtensorflowlite_c)
#   make imx93   SDK=/opt/imx-yocto # cross-compile for i.MX 93   (Cortex-A55 + Ethos-U65)
#   make imx95   SDK=/opt/imx-yocto # cross-compile for i.MX 95   (Cortex-A55 + Neutron N3)
#   make imx8mp  SDK=/opt/imx-yocto # cross-compile for i.MX 8M Plus (VxDelegate / VIP9000)
#
# SDK= points at an NXP Yocto toolchain root. The `environment-setup-…`
# script there sets CC, CFLAGS, LDFLAGS — we just source it.
#
# Output: libfacex_npu.{so,dylib} — a TFLite-backed engine that auto-selects
# eIQ Neutron → NXP VxDelegate → Arm Ethos-U external delegate → XNNPACK
# fallback at runtime. See docs/imx_npu.md.
# ---------------------------------------------------------------------------

# Optional build inputs:
TFLITE_INCLUDE ?=
TFLITE_LIB     ?=

NPU_CFLAGS = -O3 -fPIC -DFACEX_BACKEND_TFLITE -Iinclude
ifneq ($(TFLITE_INCLUDE),)
  NPU_CFLAGS += -I$(TFLITE_INCLUDE)
endif
NPU_LDFLAGS = -ltensorflowlite_c -ldl -lm -lpthread
ifneq ($(TFLITE_LIB),)
  NPU_LDFLAGS := -L$(TFLITE_LIB) $(NPU_LDFLAGS)
endif

ifeq ($(UNAME_S),Darwin)
  NPU_LIB = libfacex_npu.dylib
  NPU_LDFLAGS += -Wl,-undefined,dynamic_lookup
else
  NPU_LIB = libfacex_npu.so
endif

imx-npu: $(NPU_LIB)

$(NPU_LIB): src/backend_tflite.c include/facex_npu.h include/facex_backend.h
	@command -v $(CC) >/dev/null || { echo "no compiler"; exit 1; }
	$(CC) $(NPU_CFLAGS) -shared -o $@ src/backend_tflite.c $(NPU_LDFLAGS)
	@echo "Built $@"

# i.MX 93 — Cortex-A55 + Ethos-U65, prefers Arm Ethos-U external delegate.
imx93:
	@if [ -z "$(SDK)" ]; then echo "set SDK=/path/to/imx-yocto-sdk"; exit 1; fi
	@echo "sourcing $(SDK)/environment-setup-aarch64-poky-linux"
	@bash -c '. $(SDK)/environment-setup-aarch64-poky-linux && \
	          $$CC -O3 -fPIC -DFACEX_BACKEND_TFLITE -Iinclude \
	          -mcpu=cortex-a55 -march=armv8.2-a+dotprod+fp16 \
	          -shared -o libfacex_npu.so src/backend_tflite.c \
	          -ltensorflowlite_c -ldl -lm -lpthread'
	@echo "Built libfacex_npu.so for i.MX 93"

# i.MX 95 — Cortex-A55 + eIQ Neutron N3 NPU (NOT Ethos-U65). Same source
# artifact as imx93; runtime picks libneutron_delegate.so on this board.
imx95:
	@if [ -z "$(SDK)" ]; then echo "set SDK=/path/to/imx-yocto-sdk"; exit 1; fi
	@bash -c '. $(SDK)/environment-setup-aarch64-poky-linux && \
	          $$CC -O3 -fPIC -DFACEX_BACKEND_TFLITE -Iinclude \
	          -mcpu=cortex-a55 -march=armv8.2-a+dotprod+fp16 \
	          -shared -o libfacex_npu.so src/backend_tflite.c \
	          -ltensorflowlite_c -ldl -lm -lpthread'
	@echo "Built libfacex_npu.so for i.MX 95"

# i.MX 8M Plus — Cortex-A53 + VIP9000 NPU via NXP VxDelegate.
imx8mp:
	@if [ -z "$(SDK)" ]; then echo "set SDK=/path/to/imx-yocto-sdk"; exit 1; fi
	@bash -c '. $(SDK)/environment-setup-aarch64-poky-linux && \
	          $$CC -O3 -fPIC -DFACEX_BACKEND_TFLITE -Iinclude \
	          -mcpu=cortex-a53 -march=armv8-a+crc \
	          -shared -o libfacex_npu.so src/backend_tflite.c \
	          -ltensorflowlite_c -ldl -lm -lpthread'
	@echo "Built libfacex_npu.so for i.MX 8M Plus"

# Compile-only smoke test for the NPU API surface (runs anywhere TFLite
# headers/libs are installed; doesn't need an actual NPU device).
imx_npu_compile_test: tests/test_imx_npu_compile.c $(NPU_LIB)
	$(CC) $(NPU_CFLAGS) -o $@ tests/test_imx_npu_compile.c \
	    -L. -lfacex_npu $(NPU_LDFLAGS)
	@echo "Built imx_npu_compile_test (run with: ./imx_npu_compile_test [embed.tflite [detect.tflite]])"

# Latency benchmark for the NPU path. Mirrors facex-bench's CSV schema so
# rows from both can be concatenated and ingested by scripts/bench_all.sh
# or any spreadsheet tool. Requires libfacex_npu.so to have been built.
facex-bench-npu: tools/bench_npu.c $(NPU_LIB)
	$(CC) $(NPU_CFLAGS) -o $@ tools/bench_npu.c \
	    -L. -lfacex_npu $(NPU_LDFLAGS)
	@echo "Built facex-bench-npu (run with: ./facex-bench-npu --embed PATH.tflite [--delegate NAME] [--external-delegate /usr/lib/libneutron_delegate.so])"
