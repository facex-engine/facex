# FaceX — Fast face embedding library
# 3ms inference, 7MB binary, zero dependencies
#
# Build:  make
# Test:   make example && ./facex-example

CC ?= gcc
CFLAGS = -O3 -march=native -mfma -funroll-loops
LDFLAGS = -lm -lpthread

# Detect AVX-512
AVX512 := $(shell $(CC) -mavx512f -dM -E - < /dev/null 2>/dev/null | grep -c AVX512F)
ifeq ($(AVX512),1)
  CFLAGS += -mavx512f -mavx512vnni -mprefer-vector-width=512
endif

ifeq ($(OS),Windows_NT)
  LDFLAGS += -lsynchronization
  EXT = .exe
endif

SRCS = src/facex.c src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c

.PHONY: all clean example lib cli encrypt test detect-lib v2-cli

all: lib cli detect-lib v2-cli

# Static library
lib: libfacex.a

libfacex.a: $(SRCS)
	$(CC) $(CFLAGS) -DFACEX_LIB -c src/facex.c -o facex.o
	$(CC) $(CFLAGS) -c src/transformer_ops.c -o transformer_ops.o
	$(CC) $(CFLAGS) -c src/gemm_int8_4x8c8.c -o gemm_int8_4x8c8.o
	$(CC) $(CFLAGS) -c src/threadpool.c -o threadpool.o
	ar rcs $@ facex.o transformer_ops.o gemm_int8_4x8c8.o threadpool.o
	@rm -f *.o
	@echo "Built libfacex.a"

# Standalone CLI (for Go subprocess / testing)
cli: facex-cli$(EXT)

facex-cli$(EXT): src/edgeface_engine.c src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c src/weight_crypto.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built facex-cli$(EXT)"

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

# Parametric v2 engine (loads EFX2 .bin, supports nano/tiny/standard/xs)
v2-cli: facex-v2$(EXT)

facex-v2$(EXT): src/edgeface_engine_v2.c src/edgeface_engine.c src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c
	$(CC) $(CFLAGS) -DV2_STANDALONE -DFACEX_LIB -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built facex-v2$(EXT)"

# MobileFaceNet engine (loads EFM3 .bin, supports nano/tiny/standard/xs)
mfn-cli: facex-mfn$(EXT)

facex-mfn$(EXT): src/facex_mfn.c
	$(CC) $(CFLAGS) -DMFN_STANDALONE -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built facex-mfn$(EXT)"

# Minimal MFN example
mfn-example: example_mfn$(EXT)

example_mfn$(EXT): examples/example_mfn.c src/facex_mfn.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built example_mfn$(EXT)"

# Detector static library (Sprint 1+: scaffold only, real engine arrives in
# later sprints — see docs/plan/detector_plan.md).
detect-lib: libdetect.a

libdetect.a: src/detect.c include/detect.h
	$(CC) $(CFLAGS) -Iinclude -c src/detect.c -o detect.o
	ar rcs $@ detect.o
	@rm -f detect.o
	@echo "Built libdetect.a"

clean:
	rm -f libfacex.a libdetect.a facex-cli$(EXT) facex-example$(EXT) facex-encrypt$(EXT) golden-test$(EXT) facex-v2$(EXT) facex-mfn$(EXT) *.o
