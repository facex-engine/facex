#!/usr/bin/env bash
# bench_all.sh — sweep build configurations and print a unified
# comparison table.
#
# Builds the library + facex-bench under each (build-flag) combo
# selected by --configs, runs facex-bench against each, collects the
# CSV rows, and emits one Markdown table to stdout.
#
# Usage:
#   scripts/bench_all.sh                  # default sweep
#   scripts/bench_all.sh --iters 200      # more samples per config
#   scripts/bench_all.sh --configs "default,ACCELERATE=1"
#   scripts/bench_all.sh --format csv     # raw CSV (one row per stage)
#
# The default sweep covers what's runnable on the host:
#   * default   (NEON / scalar, baseline)
#   * ACCELERATE=1
#   * SME=1                  (compile-tested; inert on M1-M3)
#   * SME=1 ACCELERATE=1     (combined)
#
# Cross-platform: skips Apple-only configs on Linux. Caller must
# have already run `bash download_weights.sh` (or set FACEX_EMBED).

set -u
cd "$(dirname "$0")/.."

ITERS=${ITERS:-100}
WARMUP=${WARMUP:-10}
FMT=${FMT:-md}
EMBED_W=${FACEX_EMBED:-data/edgeface_xs_fp32.bin}
DETECT_W=${FACEX_DETECT:-weights/yunet_fp32.bin}

UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

DEFAULT_CONFIGS=("default" "ACCELERATE=1" "SME=1" "SME=1 ACCELERATE=1")
if [[ "$UNAME_S" != "Darwin" ]]; then
    # Apple-only flags don't apply.
    DEFAULT_CONFIGS=("default")
fi
CONFIGS=("${DEFAULT_CONFIGS[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iters)   ITERS=$2;   shift 2 ;;
        --warmup)  WARMUP=$2;  shift 2 ;;
        --format)  FMT=$2;     shift 2 ;;
        --configs) IFS=',' read -ra CONFIGS <<< "$2"; shift 2 ;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$EMBED_W" ]]; then
    echo "error: embed weights not found at $EMBED_W" >&2
    echo "       run \`bash download_weights.sh\` first" >&2
    exit 1
fi

# Per-config CSV rows accumulate here (always CSV regardless of final format,
# converted at the end).
ROWS_FILE=$(mktemp -t facex_bench_rows.XXXX)
trap 'rm -f "$ROWS_FILE"' EXIT
echo "label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face" > "$ROWS_FILE"

run_config() {
    local label="$1"; shift
    local make_flags="$*"

    {
        echo
        echo "=== building: $label  ($make_flags) ==="
        make clean >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        if ! make $make_flags bench >/dev/null 2>&1; then
            echo "build FAILED for $label" >&2
            return 1
        fi
    } >&2

    # Run bench in CSV mode and append data rows (skip the header).
    # shellcheck disable=SC2086
    ./facex-bench \
        --iters "$ITERS" --warmup "$WARMUP" \
        --label "$label" \
        --format csv \
        --embed  "$EMBED_W" \
        --detect "$DETECT_W" 2>/dev/null \
        | tail -n +2 >> "$ROWS_FILE"
}

for cfg in "${CONFIGS[@]}"; do
    if [[ "$cfg" == "default" ]]; then
        run_config "default" || true
    else
        # Sanitize the label by stripping spaces.
        label="${cfg// /+}"
        run_config "$label" $cfg || true
    fi
done

# ---- Output ----
case "$FMT" in
    csv)
        cat "$ROWS_FILE"
        ;;
    md|markdown)
        echo "# FaceX bench sweep"
        echo
        echo "host: \`$UNAME_S / $UNAME_M\`  "
        echo "iters: $ITERS  warmup: $WARMUP  embed: \`$EMBED_W\`  detect: \`$DETECT_W\`"
        echo
        echo "| label | active | stage | min ms | median ms | mean ms | p95 ms | p99 ms |"
        echo "|---|---|---|--:|--:|--:|--:|--:|"
        awk -F',' 'NR>1 {
            gsub(/^"|"$/, "", $1);
            gsub(/^"|"$/, "", $3);
            printf("| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                   $1, $3, $4, $6, $7, $8, $9, $10);
        }' "$ROWS_FILE"
        ;;
    *)
        echo "unknown --format: $FMT" >&2; exit 2 ;;
esac
