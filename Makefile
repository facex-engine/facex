# FaceX — Fast face embedding library
# 3ms inference, 7MB binary, zero dependencies
#
# Build:  make
# Test:   make example && ./facex-example

CC ?= gcc
CFLAGS = -O3 -march=native -mfma -funroll-loops
LDFLAGS = -lm -lpthread

ifeq ($(OS),Windows_NT)
  LDFLAGS += -lsynchronization
  EXT = .exe
endif


.PHONY: all clean example lib cli cli-fp32 encrypt test detect-lib v2-cli

all: lib cli detect-lib v2-cli

# Static library
lib: libfacex.a

LIBOBJ = facex.o transformer_ops.o gemm_int8_4x8c8.o threadpool.o detect.o align.o weight_crypto.o

libfacex.a: $(LIBOBJ)
	$(AR) rcs $@ $^

%.o: src/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iinclude -MMD -MP -c $< -o $@

-include $(LIBOBJ:.o=.d)

# Standalone CLI (for Go subprocess / testing)
cli: facex-cli$(EXT)

CLI_SRC = src/edgeface_engine.c src/transformer_ops.c src/threadpool.c src/weight_crypto.c

facex-cli$(EXT): $(CLI_SRC) src/gemm_int8_4x8c8.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	@echo "Built facex-cli$(EXT)"

# FP32 reference for comparing embeddings across CPU profiles.
cli-fp32: facex-cli-fp32$(EXT)

facex-cli-fp32$(EXT): $(CLI_SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFACEX_FP32_ONLY -Iinclude -o $@ $^ $(LDFLAGS)

# Example program
example: facex-example$(EXT)

facex-example$(EXT): examples/example.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Encryption tool
encrypt: facex-encrypt$(EXT)

facex-encrypt$(EXT): src/weight_crypto.c
	$(CC) $(CFLAGS) -DWEIGHT_CRYPTO_MAIN -o $@ $< $(LDFLAGS)

.PHONY: check
check: kernel-test$(EXT) mfn-depthwise-test$(EXT) geometry-test$(EXT) motion-phase-test$(EXT)
	./kernel-test$(EXT)
	./mfn-depthwise-test$(EXT)
	./geometry-test$(EXT)
	./motion-phase-test$(EXT)

motion-phase-test$(EXT): tests/motion_phase.c nn2/src/motion_gate.c nn2/src/nn2_internal.h nn2/include/nn2.h
	$(CC) $(CFLAGS) -Inn2/src -Inn2/include -o $@ tests/motion_phase.c nn2/src/motion_gate.c

geometry-test$(EXT): tests/geometry_math.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)

mfn-depthwise-test$(EXT): tests/mfn_depthwise.c src/facex_mfn.c include/facex_mfn.h
	$(CC) $(CFLAGS) -Iinclude -o $@ $< $(LDFLAGS)

kernel-test$(EXT): tests/packed_math.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)

# Golden test
test: check golden-test$(EXT)
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

libdetect.a: detect.o
	$(AR) rcs $@ $^

clean:
	rm -f libfacex.a libdetect.a facex-cli$(EXT) facex-cli-fp32$(EXT) facex-example$(EXT) facex-encrypt$(EXT) golden-test$(EXT) facex-v2$(EXT) facex-mfn$(EXT) kernel-test$(EXT) mfn-depthwise-test$(EXT) geometry-test$(EXT) motion-phase-test$(EXT) *.o $(LIBOBJ:.o=.d)
