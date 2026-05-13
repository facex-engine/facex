/*
 * tensor_column.c — Tensor Column execution engine.
 *
 * Instead of processing the full network layer-by-layer (writing huge
 * intermediate tensors to main memory), process VERTICAL STRIPS of the
 * network that fit entirely in L2 cache.
 *
 * A "tensor column" is a vertical slice of width W_col pixels through
 * all layers of the network. Each layer's activations for that slice
 * fit in L2, so data flows through multiple layers without touching RAM.
 *
 * For conv layers with receptive field expansion, each layer needs a
 * slightly wider input than output (halo). We compute the halo widths
 * bottom-up and process overlapping strips.
 *
 * Example for 3 conv layers (3x3, pad=1, stride=1):
 *   Output strip width: W_col = 32
 *   Layer 3 input: 32 + 2 = 34 (1 pixel halo each side)
 *   Layer 2 input: 34 + 2 = 36
 *   Layer 1 input: 36 + 2 = 38
 *
 * For stride-2 layers, the input width doubles:
 *   Layer with stride 2: output 32 → input 64 + halo
 *
 * Memory per column: sum of (channels × height × strip_width) per layer
 * Target: fit in L2 cache (~256KB per core)
 *
 * Performance gain: activations stay in L2 (200 GB/s) instead of
 * main memory (40 GB/s) → 3-5x bandwidth improvement.
 */

#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============ Halo computation ============ */

typedef struct {
    int w_in;    /* input strip width needed (including halo) */
    int w_out;   /* output strip width produced */
    int halo_l;  /* left halo pixels */
    int halo_r;  /* right halo pixels */
    int stride;
    int channels_in;
    int channels_out;
} ColumnLayerInfo;

/*
 * Compute required input widths for each layer, working backwards from
 * the desired output strip width.
 *
 * Returns array of ColumnLayerInfo for each layer.
 */
ColumnLayerInfo* nn2_tc_compute_halos(
    int n_layers,
    const int* kernel_sizes,  /* [n_layers] */
    const int* strides,       /* [n_layers] */
    const int* paddings,      /* [n_layers] */
    const int* channels_in,   /* [n_layers] */
    const int* channels_out,  /* [n_layers] */
    int output_strip_width)   /* desired output W */
{
    ColumnLayerInfo* info = (ColumnLayerInfo*)calloc(n_layers, sizeof(ColumnLayerInfo));

    /* Work backwards: start from output strip width */
    int w = output_strip_width;
    for (int i = n_layers - 1; i >= 0; i--) {
        info[i].w_out = w;
        info[i].stride = strides[i];
        info[i].channels_in = channels_in[i];
        info[i].channels_out = channels_out[i];

        /* For conv with kernel k, stride s, pad p:
         * w_in = (w_out - 1) * stride + kernel - 2*pad
         * But for tensor columns, we need the actual input pixels: */
        int k = kernel_sizes[i];
        int s = strides[i];
        int p = paddings[i];
        int w_in = (w - 1) * s + k; /* without padding, the raw input width */
        info[i].halo_l = k / 2;     /* halo needed on left */
        info[i].halo_r = k / 2;     /* halo needed on right */
        info[i].w_in = w_in;

        /* Next layer's output is this layer's input */
        w = w_in;
    }

    return info;
}

/*
 * Estimate memory needed for one tensor column.
 * Returns bytes needed for all intermediate activations.
 */
size_t nn2_tc_memory_estimate(const ColumnLayerInfo* info, int n_layers,
                               int max_height)
{
    size_t total = 0;
    for (int i = 0; i < n_layers; i++) {
        /* Height at this layer depends on strides of preceding layers */
        int h = max_height;
        for (int j = 0; j < i; j++)
            h = (h + 1) / info[j].stride; /* approximate */

        size_t layer_size = (size_t)info[i].channels_out * h * info[i].w_out * sizeof(float);
        total += layer_size;
    }
    return total;
}

/*
 * Find optimal strip width that fits in target cache size.
 */
int nn2_tc_optimal_width(
    int n_layers,
    const int* kernel_sizes,
    const int* strides,
    const int* paddings,
    const int* channels_in,
    const int* channels_out,
    int input_height,
    size_t target_cache_bytes)  /* e.g. 256*1024 for L2 */
{
    /* Binary search for the largest strip width that fits */
    int lo = 4, hi = 128;
    int best = lo;

    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        ColumnLayerInfo* info = nn2_tc_compute_halos(
            n_layers, kernel_sizes, strides, paddings,
            channels_in, channels_out, mid);
        size_t mem = nn2_tc_memory_estimate(info, n_layers, input_height);
        free(info);

        if (mem <= target_cache_bytes) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return best;
}

/* ============ Column execution for a sequence of convolutions ============ */

/*
 * Process a vertical strip through a sequence of conv layers.
 * All intermediate data stays in the provided buffer (should be L2-sized).
 *
 * input:  full input tensor [Cin, H, W] (read only the strip region)
 * output: strip of final output [Cout, Hout, strip_w_out]
 * strip_x: starting x position in the output
 */
void nn2_tc_execute_strip(
    const ConvLayer* layers, int n_layers,
    const float* input, int H, int W,
    float* output, int Hout, int Wout,
    int strip_x, int strip_w,
    float* work_buf)  /* L2-sized work buffer */
{
    /* Compute halo requirements */
    int w = strip_w;
    int x_starts[32], x_widths[32]; /* max 32 layers */
    int heights[32];

    /* Backward pass: compute required input widths */
    x_widths[n_layers] = strip_w;
    x_starts[n_layers] = strip_x;
    heights[n_layers] = Hout;

    for (int i = n_layers - 1; i >= 0; i--) {
        int k = layers[i].k;
        int s = layers[i].stride;
        int p = layers[i].pad;

        /* Output width at layer i+1 needs input width at layer i */
        int w_out = x_widths[i + 1];
        int x_out = x_starts[i + 1];

        /* Compute input range */
        int x_in = x_out * s - p;
        int w_in = (w_out - 1) * s + k;

        /* Clamp to valid range */
        if (x_in < 0) x_in = 0;
        int h_in = (i == 0) ? H : heights[i]; /* approximate */

        x_starts[i] = x_in;
        x_widths[i] = w_in;
        heights[i] = (i == 0) ? H : (heights[i + 1] - 1) * s + k;
        if (heights[i] > H) heights[i] = H;
    }

    /* Forward pass: process each layer on the strip */
    const float* cur_input = input;
    float* ping = work_buf;
    float* pong = work_buf + 256 * 256 * 64; /* half of work buffer */
    int cur_x = x_starts[0];
    int cur_w = x_widths[0];
    int cur_h = H;

    for (int i = 0; i < n_layers; i++) {
        const ConvLayer* L = &layers[i];
        int out_h = (cur_h + 2 * L->pad - L->k) / L->stride + 1;
        int out_w = x_widths[i + 1];
        int out_x = x_starts[i + 1];

        /* Extract strip from input and run conv */
        /* For simplicity, use full conv on the strip region */
        /* TODO: implement strip-aware conv that reads only the needed region */

        /* For now, just demonstrate the concept with a simple per-strip conv */
        /* This would need a specialized conv that operates on sub-regions */

        cur_h = out_h;
        cur_w = out_w;
        cur_x = out_x;

        /* Swap buffers */
        float* tmp = ping; ping = pong; pong = tmp;
    }
}

/* ============ Print info ============ */

void nn2_tc_print_info(int input_size)
{
    /* YOLOv8n backbone layer params */
    int ks[] = {3, 3, 3, 3, 3};         /* kernel sizes */
    int ss[] = {2, 2, 2, 2, 2};         /* strides */
    int ps[] = {1, 1, 1, 1, 1};         /* paddings */
    int ci[] = {3, 16, 32, 64, 128};    /* channels in */
    int co[] = {16, 32, 64, 128, 256};  /* channels out */
    int n = 5;

    printf("=== Tensor Column Analysis for YOLOv8n (input=%d) ===\n\n", input_size);

    for (size_t cache_kb = 128; cache_kb <= 512; cache_kb *= 2) {
        int opt_w = nn2_tc_optimal_width(n, ks, ss, ps, ci, co,
                                          input_size, cache_kb * 1024);
        ColumnLayerInfo* info = nn2_tc_compute_halos(n, ks, ss, ps, ci, co, opt_w);
        size_t mem = nn2_tc_memory_estimate(info, n, input_size);

        printf("L2 = %zuKB → strip_width = %d\n", cache_kb, opt_w);
        printf("  Input strip: %d pixels wide (with halos)\n", info[0].w_in);
        printf("  Memory per column: %.1f KB\n", mem / 1024.0);
        printf("  Strips needed: %d\n", (input_size + opt_w - 1) / opt_w);

        int total_strips = (input_size + opt_w - 1) / opt_w;
        /* Estimate: if data stays in L2 (200GB/s) vs DRAM (40GB/s) */
        float bw_ratio = 200.0f / 40.0f;
        printf("  Expected speedup from L2 residency: %.1fx\n", bw_ratio);
        printf("  (assuming memory-bound execution)\n\n");

        free(info);
    }
}
