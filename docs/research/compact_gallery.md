# Single-query CPU gallery search

The float64 Gallery remains available. CompactGallery stores the original float32 enrollment matrix, one float64 norm and a boolean flag per row. These byte counts describe retained array payloads, excluding construction, temporaries, outputs and allocator overhead.

For a finite, nonzero query, both classes normalize in float64. CompactGallery scans with a float32 query and divides each rough dot product by the stored enrollment norm. On an IEEE CPU the absolute dot error divided by that norm is bounded by a multiple of unit roundoff because the sum of the absolute normalized products is at most one (Cauchy-Schwarz). For D=512 we use gamma(2D+4), with gamma(m)=mu/(1-mu) and u the unit roundoff. The shortlist includes scores within twice this bound of the kth approximate score: both the kth score and its competitor can move. Rows with norms outside [1e-20, 1e20] are always refined and do not set a finite cutoff. If fewer than k ordinary rows remain, all rows are refined.

Both gallery classes score finalists with the same contiguous row-wise float64 reduction. This also fixes a tie bug in the previous implementation, where identical enrollment rows could acquire different last bits in different BLAS tiles. Earlier Gallery scores may differ in those last bits. Identical best rows now abstain in identify, including at min_margin=0.

## Measurements

One Linux CPU, one BLAS thread, k=5, one warmup and 40 measured random queries per gallery size. Both versions were warmed and the order alternated. Median CPU milliseconds per search, excluding enrollment.

| Enrollment rows | Gallery ms | CompactGallery ms | Gallery MiB | CompactGallery MiB |
|---:|---:|---:|---:|---:|
| 4096 | 0.806 | 0.568 | 16 | 8.04 |
| 32768 | 6.582 | 5.439 | 128 | 64.28 |
| 65536 | 12.995 | 10.535 | 256 | 128.56 |

IDs and float64 scores matched for all benchmark queries. Fifteen real-model outputs on repository fixture images also matched when queried against each other. Tests separately check a full-score reference with nearly tied candidates, duplicate rows, extreme norms, threshold boundaries and noncontiguous enrollment. One deliberate all-duplicates benchmark on 4096 rows took 2.994 ms for Gallery and 3.578 ms for CompactGallery. This experiment measures gallery lookup in isolation.

```sh
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 python tests/test_compact_gallery.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 python tests/bench_compact_gallery.py
```
