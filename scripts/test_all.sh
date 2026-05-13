#!/usr/bin/env bash
# test_all.sh — run every test that's runnable in this environment.
#
# Exits non-zero on the first failure. Prints a summary at the end so
# the coverage_matrix.md can be filled from the actual output.
#
# Usage:
#   scripts/test_all.sh                # default: all tests
#   scripts/test_all.sh --skip-camera  # CI mode (no camera permission)
#   VERBOSE=1 scripts/test_all.sh      # echo every command
#
# Subsequent topic commits (Mac, i.MX, ESP32) append their own checks
# below the foundation sections.

set -u
cd "$(dirname "$0")/.."

SKIP_CAMERA=0
for a in "$@"; do
    case "$a" in
        --skip-camera) SKIP_CAMERA=1 ;;
        --help|-h)
            sed -n '2,15p' "$0"; exit 0 ;;
    esac
done

ARCH="$(uname -m)"
OS="$(uname -s)"
PASS=()
FAIL=()
SKIP=()

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
hdr()   { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

# Run a labelled test. $1 = label, $2... = command.
run() {
    local label="$1"; shift
    printf '\n→ %-40s ' "$label"
    if [[ "${VERBOSE:-0}" = 1 ]]; then echo; echo "    cmd: $*"; fi
    local out
    if out=$("$@" 2>&1); then
        green "PASS"
        PASS+=("$label")
    else
        red "FAIL"
        echo "$out" | sed 's/^/    | /' | head -20
        FAIL+=("$label")
    fi
}

skip() {
    printf '\n→ %-40s ' "$1"; yellow "SKIP ($2)"
    SKIP+=("$1 ($2)")
}

# ---------------------------------------------------------------------------
hdr "Environment"
echo "host:     $OS / $ARCH"
echo "compiler: $(${CC:-cc} --version | head -1)"
echo "branch:   $(git branch --show-current 2>/dev/null || echo n/a)"
echo "head:     $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

# ---------------------------------------------------------------------------
hdr "Default build (host arch)"
run "make clean"                     bash -c 'make clean >/dev/null 2>&1 || true; true'
run "make (default)"                 make
run "libfacex.a exists"              test -f libfacex.a
run "facex-cli exists"               test -x ./facex-cli
run "libdetect.a exists"             test -f libdetect.a

# ---------------------------------------------------------------------------
hdr "Smoke tests on the default build"
if [[ -f data/edgeface_xs_fp32.bin ]]; then
    run "make test (golden)"         make test
else
    skip "make test (golden)" "data/edgeface_xs_fp32.bin missing"
fi

if [[ -f data/edgeface_xs_fp32.bin && -f weights/yunet_fp32.bin ]]; then
    run "make mac-test"              make mac-test
elif [[ -f data/edgeface_xs_fp32.bin ]]; then
    run "make mac-test (embed-only)" make mac-test
else
    skip "make mac-test" "weights missing"
fi

# ---------------------------------------------------------------------------
hdr "External dependency audit (default build)"
if [[ "$OS" = "Darwin" ]]; then
    run "facex-cli has only system deps" \
        bash -c '! otool -L facex-cli | tail -n +2 | grep -vqE "/usr/lib/libSystem|/System/Library"'
    run "libfacex.a is self-contained" \
        bash -c 'test "$(ar t libfacex.a | grep -cE "\.o$")" -ge 5'
fi

# ---------------------------------------------------------------------------
hdr "Apple Silicon variants (Mac perf paths)"
if [[ "$ARCH" = "arm64" && "$OS" = "Darwin" ]]; then
    # SME=1 build (compile-only validation)
    run "make clean"                 bash -c 'make clean >/dev/null 2>&1 || true; true'
    run "make SME=1"                 make SME=1
    run "fmopa is in libfacex.a" \
        bash -c 'ar x libfacex.a transformer_ops_sme.o && \
                 otool -tv transformer_ops_sme.o | grep -q fmopa && rm -f transformer_ops_sme.o'
    run "rdvl NOT in transformer_ops.o (M1-M3 safe)" \
        bash -c 'ar x libfacex.a transformer_ops.o && \
                 ! otool -tv transformer_ops.o | grep -qE "rdvl|smstart|fmopa" && \
                 rm -f transformer_ops.o'
    run "mac-test still passes with SME-built lib" \
        bash -c 'make mac-test 2>&1 | grep -q "PASS: macOS arm64 smoke test"'

    # ACCELERATE=1 build — Apple Accelerate / AMX path
    run "make clean"                 bash -c 'make clean >/dev/null 2>&1 || true; true'
    run "make ACCELERATE=1"          make ACCELERATE=1
    run "Accelerate symbol present in libfacex.a" \
        bash -c 'ar x libfacex.a backend_accelerate.o 2>/dev/null && \
                 nm backend_accelerate.o | grep -q matmul_fp32_packed_accelerate && \
                 rm -f backend_accelerate.o'
    run "facex-cli links Accelerate.framework" \
        bash -c 'otool -L facex-cli | grep -q Accelerate.framework'
    run "mac-test passes with Accelerate" \
        bash -c 'make ACCELERATE=1 mac-test 2>&1 | grep -q "PASS: macOS arm64 smoke test"'

    # SME=1 ACCELERATE=1 combo
    run "make clean"                 bash -c 'make clean >/dev/null 2>&1 || true; true'
    run "make SME=1 ACCELERATE=1"    make SME=1 ACCELERATE=1
    run "mac-test passes with SME+Accelerate" \
        bash -c 'make SME=1 ACCELERATE=1 mac-test 2>&1 | grep -q "PASS: macOS arm64 smoke test"'

    # COREML=1 build — Core ML / ANE bridge (compile + link only;
    # runtime ANE dispatch needs an .mlpackage we can't produce here).
    run "make clean"                 bash -c 'make clean >/dev/null 2>&1 || true; true'
    run "make COREML=1"              make COREML=1
    run "Core ML symbols present in libfacex.a" \
        bash -c 'ar x libfacex.a backend_coreml.o 2>/dev/null && \
                 nm backend_coreml.o | grep -q facex_coreml_init && \
                 rm -f backend_coreml.o'
    run "facex-cli links CoreML.framework" \
        bash -c 'otool -L facex-cli | grep -q CoreML.framework'
    run "facex_coreml_init handles missing .mlpackage gracefully" \
        bash -c '
            cat > /tmp/_cm_smoke.c <<EOF
#include "facex_coreml.h"
#include <stdio.h>
int main(void){
    FaceXCoreMLOptions o = {0};
    FaceXCoreML* fx = facex_coreml_init("/tmp/__nope__.mlpackage", &o);
    return fx ? 1 : 0;
}
EOF
            cc -O2 -Iinclude -DFACEX_HAVE_COREML -o /tmp/_cm_smoke /tmp/_cm_smoke.c \
               -L. -lfacex -framework CoreML -framework Foundation -lm -lpthread &&
            /tmp/_cm_smoke 2>/dev/null
            rc=$?
            rm -f /tmp/_cm_smoke /tmp/_cm_smoke.c
            exit $rc'
    run "tools/export_coreml.py parses + --help" \
        bash -c 'python3 tools/export_coreml.py --help >/dev/null'

    # Universal binary build (arm64 + x86_64)
    run "make clean"                 bash -c 'make clean >/dev/null 2>&1 || true; true'
    run "make mac-universal"         make mac-universal
    run "libfacex-universal.a is fat" \
        bash -c 'file libfacex-universal.a | grep -q "universal binary"'
    run "universal contains arm64" \
        bash -c 'lipo -info libfacex-universal.a | grep -q arm64'
    run "universal contains x86_64" \
        bash -c 'lipo -info libfacex-universal.a | grep -q x86_64'
    run "arm64 slice has NEON code" \
        bash -c 'lipo -thin arm64 libfacex-universal.a -output /tmp/_a.a && \
                 ar x /tmp/_a.a transformer_ops.o && \
                 [ "$(otool -tv transformer_ops.o | grep -cE "(fmla|fmul)")" -gt 100 ] && \
                 rm -f /tmp/_a.a transformer_ops.o'
    run "x86_64 slice has AVX2 code" \
        bash -c 'lipo -thin x86_64 libfacex-universal.a -output /tmp/_x.a && \
                 ar x /tmp/_x.a transformer_ops.o && \
                 [ "$(otool -tv transformer_ops.o | grep -cE "(vfmadd|vmovups)")" -gt 100 ] && \
                 rm -f /tmp/_x.a transformer_ops.o'

    # Restore default build
    run "make clean"                 bash -c 'make clean >/dev/null 2>&1 || true; true'
    run "make (default restore)"     make
else
    skip "Mac perf variants" "not on Apple Silicon"
fi

# ---------------------------------------------------------------------------
hdr "Unified bench tool"
if [[ -f data/edgeface_xs_fp32.bin ]]; then
    run "make bench"               make bench
    run "facex-bench --help"       bash -c './facex-bench --help 2>&1 | grep -q "Usage"'
    run "facex-bench md output"    bash -c './facex-bench --iters 5 --warmup 2 --format md   | grep -q "FaceX bench"'
    run "facex-bench csv output"   bash -c './facex-bench --iters 5 --warmup 2 --format csv  | grep -q "label,compiled,active"'
    run "facex-bench json output"  bash -c './facex-bench --iters 5 --warmup 2 --format json | grep -q "stages"'
    run "facex-bench embed-only stage" \
        bash -c './facex-bench --iters 5 --warmup 2 --stage embed --format csv | grep -q ",embed,5,"'
    run "scripts/bench_all.sh produces a sweep table" \
        bash -c 'scripts/bench_all.sh --iters 5 --warmup 2 --configs "default" 2>/dev/null | grep -q "default"'
else
    skip "unified bench" "data/edgeface_xs_fp32.bin missing"
fi

# ---------------------------------------------------------------------------
hdr "i.MX NPU compile checks"
# We don't have libtensorflowlite_c locally; use minimal stub headers so
# the syntax check works on any host. Real builds against a vendor SDK
# happen via `make imx93 SDK=...` etc.
STUB_DIR="$(mktemp -d -t facex_tflite_stub.XXXX)"
mkdir -p "$STUB_DIR/tensorflow/lite/c"
cat > "$STUB_DIR/tensorflow/lite/c/c_api.h" <<'EOF'
#ifndef TFL_STUB_H
#define TFL_STUB_H
#include <stddef.h>
#include <stdint.h>
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteTensor TfLiteTensor;
typedef struct TfLiteDelegate TfLiteDelegate;
typedef enum { kTfLiteOk=0, kTfLiteError=1 } TfLiteStatus;
typedef enum { kTfLiteNoType=0, kTfLiteFloat32=1, kTfLiteInt8=9 } TfLiteType;
typedef struct { float scale; int32_t zero_point; } TfLiteQuantizationParams;
TfLiteModel* TfLiteModelCreateFromFile(const char*);
void TfLiteModelDelete(TfLiteModel*);
TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate(void);
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions*);
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions*,int);
void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions*,TfLiteDelegate*);
TfLiteInterpreter* TfLiteInterpreterCreate(const TfLiteModel*,const TfLiteInterpreterOptions*);
void TfLiteInterpreterDelete(TfLiteInterpreter*);
TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter*);
TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter*);
TfLiteTensor* TfLiteInterpreterGetInputTensor(const TfLiteInterpreter*,int32_t);
const TfLiteTensor* TfLiteInterpreterGetOutputTensor(const TfLiteInterpreter*,int32_t);
TfLiteType TfLiteTensorType(const TfLiteTensor*);
void* TfLiteTensorData(const TfLiteTensor*);
TfLiteQuantizationParams TfLiteTensorQuantizationParams(const TfLiteTensor*);
#endif
EOF
cat > "$STUB_DIR/tensorflow/lite/c/c_api_experimental.h" <<'EOF'
#ifndef TFL_STUB_EXP_H
#define TFL_STUB_EXP_H
#include "c_api.h"
#endif
EOF
run "src/backend_tflite.c syntax-check" \
    clang -fsyntax-only -DFACEX_BACKEND_TFLITE -Iinclude -I"$STUB_DIR" src/backend_tflite.c
run "tests/test_imx_npu_compile.c syntax-check" \
    clang -fsyntax-only -Iinclude -I"$STUB_DIR" tests/test_imx_npu_compile.c
rm -rf "$STUB_DIR"

# ---------------------------------------------------------------------------
hdr "ESP32-P4 component (syntax-only, no IDF here)"
# We don't have ESP-IDF locally — synthesize the minimal IDF headers
# (esp_err.h, esp_log.h, esp_timer.h, sdkconfig.h) just enough to syntax-
# check our wrapper. Real builds happen via idf.py on a host with IDF.
IDF_STUB="$(mktemp -d -t facex_idf_stub.XXXX)"
cat > "$IDF_STUB/esp_err.h" <<'EOF'
#pragma once
#include <stdint.h>
typedef int esp_err_t;
#define ESP_OK                  0
#define ESP_FAIL               -1
#define ESP_ERR_INVALID_ARG     0x102
#define ESP_ERR_INVALID_STATE   0x103
#define ESP_ERR_NOT_FOUND       0x105
#define ESP_LOGE(t, ...)        ((void)0)
#define ESP_LOGW(t, ...)        ((void)0)
#define ESP_LOGI(t, ...)        ((void)0)
#define ESP_LOGD(t, ...)        ((void)0)
#define ESP_RETURN_ON_ERROR(x, t, ...) do { esp_err_t _e=(x); if (_e) return _e; } while(0)
#define ESP_ERROR_CHECK(x)      do { (void)(x); } while(0)
EOF
cat > "$IDF_STUB/esp_log.h"   <<'EOF'
#pragma once
#include "esp_err.h"
EOF
cat > "$IDF_STUB/esp_timer.h" <<'EOF'
#pragma once
#include <stdint.h>
static inline int64_t esp_timer_get_time(void){return 0;}
EOF
cat > "$IDF_STUB/sdkconfig.h" <<'EOF'
#pragma once
#define CONFIG_FACEX_BACKEND_STUB 1
EOF
run "components/facex/src/facex_esp.c syntax-check" \
    clang -fsyntax-only \
          -Icomponents/facex/include \
          -I"$IDF_STUB" \
          -include esp_err.h \
          components/facex/src/facex_esp.c
rm -rf "$IDF_STUB"

# ---------------------------------------------------------------------------
hdr "Camera benchmark"
if [[ "$OS" = "Darwin" && "$SKIP_CAMERA" = 0 ]]; then
    run "make bench-camera"          make bench-camera
    run "facex-camera-bench --help"  ./facex-camera-bench --help
    # Single bench run, retry once on flake. macOS holds the camera
    # device briefly after a process exits — a 2 s settle gap between
    # back-to-back runs avoids fighting it.
    bench_once() {
        local logf="$1"; shift
        ( "$@" >"$logf" 2>&1 ) &
        local pid=$! waited=0
        # 20 s window — camera stack can need 3-5 s to deliver first frame
        # cold, especially right after a previous bench run released the
        # device. 30 frames at 30 fps then takes ~1 s.
        while kill -0 "$pid" 2>/dev/null; do
            sleep 1; waited=$((waited+1))
            if [[ $waited -gt 20 ]]; then kill "$pid" 2>/dev/null; sleep 1; break; fi
        done
        wait "$pid" 2>/dev/null
        grep -q "fps" "$logf"
    }
    bench_run() {
        local label="$1"; shift
        local logf="$(mktemp -t facex_bench.XXXX)"
        printf '\n→ %-40s ' "$label"
        if bench_once "$logf" "$@"; then
            green "PASS"; PASS+=("$label")
        else
            sleep 2  # let macOS release the device
            if bench_once "$logf" "$@"; then
                green "PASS (retried)"; PASS+=("$label")
            else
                red "FAIL"; sed 's/^/    | /' "$logf" | head -10; FAIL+=("$label")
            fi
        fi
        rm -f "$logf"
    }
    bench_run "camera-only baseline (--no-detect)" ./facex-camera-bench --frames 30 --no-detect
    sleep 2  # gap between back-to-back camera runs
    if [[ -f data/edgeface_xs_fp32.bin ]]; then
        bench_run "full pipeline 30 frames" ./facex-camera-bench --frames 30
    fi
else
    skip "camera bench" "non-Darwin or --skip-camera"
fi

# ---------------------------------------------------------------------------
hdr "Summary"
TOTAL=$(( ${#PASS[@]} + ${#FAIL[@]} + ${#SKIP[@]} ))
green  "passed:  ${#PASS[@]}/$TOTAL"
yellow "skipped: ${#SKIP[@]}/$TOTAL"
if [[ ${#FAIL[@]} -eq 0 ]]; then
    green  "failed:  0/$TOTAL"
    echo
    green  "ALL OK"
    exit 0
else
    red    "failed:  ${#FAIL[@]}/$TOTAL"
    echo
    for f in "${FAIL[@]}"; do red "  ✗ $f"; done
    exit 1
fi
