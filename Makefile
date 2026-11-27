# FaceX native detection and recognition
#
# Build:  make
# Test:   make example && ./facex-example

CC ?= gcc
CFLAGS ?= -O3 -march=native -funroll-loops
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

CLI_COMMON = src/facex.c src/cli_detect.c src/detect.c src/align.c src/transformer_ops.c src/weight_crypto.c
CLI_HEADERS = src/edgeface_engine.c src/edgeface_packed.h src/threadpool.h include/facex.h include/detect.h
CLI_SRC = $(CLI_COMMON) src/threadpool.c src/gemm_int8_4x8c8.c
FP32_SRC = $(CLI_COMMON) src/threadpool_stub.c

facex-cli$(EXT): $(CLI_SRC) $(CLI_HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFACEX_BUILD_CLI -Iinclude -o $@ $(CLI_SRC) $(LDFLAGS)

cli-fp32: facex-cli-fp32$(EXT)

facex-cli-fp32$(EXT): $(FP32_SRC) $(CLI_HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DFACEX_BUILD_CLI -DFACEX_FP32_ONLY -Iinclude -o $@ $(FP32_SRC) $(LDFLAGS)

libfacex.so: $(addprefix src/,$(LIBOBJ:.o=.c)) $(CLI_HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared -fPIC -Iinclude -o $@ $(addprefix src/,$(LIBOBJ:.o=.c)) $(LDFLAGS)

# Rebuild the distributed detector from the native YuNet implementation.
.PHONY: wasm-detector
wasm-detector:
	emcc -O3 -msimd128 -Iinclude src/detect.c -o wasm/detect.js \
	    -sMODULARIZE=1 -sEXPORT_NAME=DetectModule -sALLOW_MEMORY_GROWTH=1 -sFORCE_FILESYSTEM=1 \
	    -sEXPORTED_FUNCTIONS='["_detect_init","_detect_run","_detect_free","_detect_version","_malloc","_free"]' \
	    -sEXPORTED_RUNTIME_METHODS='["FS","cwrap"]' -sENVIRONMENT=web,worker,node

# Example program
example: facex-example$(EXT)

facex-example$(EXT): examples/example.c libfacex.a
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)

# Encryption tool
encrypt: facex-encrypt$(EXT)

facex-encrypt$(EXT): src/weight_crypto.c
	$(CC) $(CFLAGS) -DWEIGHT_CRYPTO_MAIN -o $@ $< $(LDFLAGS)

.PHONY: check
check: nn2-gemm-test$(EXT) kernel-test$(EXT) mfn-depthwise-test$(EXT) geometry-test$(EXT) motion-phase-test$(EXT)
	./nn2-gemm-test$(EXT)
	./kernel-test$(EXT)
	./mfn-depthwise-test$(EXT)
	./geometry-test$(EXT)
	./motion-phase-test$(EXT)

nn2-gemm-test$(EXT): tests/nn2_gemm.c nn2/src/gemm.c nn2/src/gemm_res.c nn2/src/pool.c nn2/src/gemm_vector.h nn2/src/fast_exp.h
	$(CC) $(CFLAGS) -Inn2/include -Inn2/src -o $@ $(filter %.c,$^) $(LDFLAGS)

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
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)

# Parametric v2 engine (loads EFX2 .bin, supports nano/tiny/standard/xs)
v2-cli: facex-v2$(EXT)

facex-v2$(EXT): src/edgeface_packed.h src/edgeface_engine_v2.c src/edgeface_engine.c src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c
	$(CC) $(CFLAGS) -DV2_STANDALONE -DFACEX_LIB -Iinclude -o $@ $(filter %.c,$^) $(LDFLAGS)
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

# Standalone YuNet detector library.
detect-lib: libdetect.a

libdetect.a: detect.o
	$(AR) rcs $@ $^

clean:
	rm -f libfacex.so libfacex.a libdetect.a facex-cli$(EXT) facex-cli-fp32$(EXT) facex-example$(EXT) facex-encrypt$(EXT) golden-test$(EXT) facex-v2$(EXT) facex-mfn$(EXT) kernel-test$(EXT) mfn-depthwise-test$(EXT) geometry-test$(EXT) motion-phase-test$(EXT) nn2-gemm-test$(EXT) example_mfn$(EXT) *.o $(LIBOBJ:.o=.d)
