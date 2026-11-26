# Integer screening with exact finalist scoring

Run from the repository root on Linux with NumPy and GCC:

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python tests/bench_gallery_codes.py

This standalone experiment targets a single nonzero query against finite nonzero 512-dimensional enrollment vectors. The index scans integer codes and then reads selected original float32 rows from a warm memory-mapped file. Original rows are required for exact scoring. Enrollment time, disk latency on a cold cache, total RSS and model inference are outside these measurements.

Let x and q be unit vectors, c the enrollment mean, v=x-c and t=q-c. Then
q·x = 1 - (||v||² + ||t||²)/2 + v·t.
Store the row length, row scale, original norm and reconstruction error r=||v-V|| for a quantized approximation V. Quantize t to T and measure d=||t-T||. The error in the integer dot estimate is bounded by r||t|| + (||v||+r)d, by expanding v·t - V·T. The uncentered variant directly estimates x·q. The experiment adds an absolute floating point allowance of 1e-12. It scores five seed candidates exactly, then includes every remaining row whose upper bound reaches the fifth seed score. The final reduction and stable row ordering match CompactGallery. This is an empirical CPU implementation of a real-arithmetic bound, without a formal floating point proof.

AMD EPYC 9V45, one pinned CPU, one BLAS thread, one warmup and 40 measured queries per case. Rotating execution order, elapsed wall time. Every final ID and score matched the baseline, including exact ties. Median finalist counts include the five seeds.

| Gallery | Method | Median ms | p95 ms | Median finalists |
|---|---|---:|---:|---:|
| 65536 independent random rows | CompactGallery | 4.00 | 4.56 | — |
| same | 8 bit | 1.60 | 1.89 | 9 |
| same | 4 bit | 13.38 | 15.90 | 17975 |
| 4096 near-identical rows with duplicate pairs | CompactGallery | 2.58 | 2.97 | — |
| same | 8 bit | 2.53 | 2.86 | 4096 |
| same | centered 8 bit | 0.174 | 0.246 | 8 |
| same | centered 4 bit | 0.996 | 1.256 | 1430 |

Uncentered 8-bit index payload is 33.5 MiB at 65536 rows (32 MiB codes plus 1.5 MiB row metadata). The backing file adds 128 MiB of vector payload, which can also occupy the OS page cache. Standalone approximate 8-bit seed recall@5 was 95.5% in the random workload. Exact refinement therefore matters. The clustered case deliberately uses one shared center for both enrollment and query generation. Mixed clusters, queries far from a cluster, realistic face gallery distributions, and cold reads require separate measurements before integrating this into the product.
