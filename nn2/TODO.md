# nn2 NanoDet-Plus-m TODO

## Status
- Backbone ShuffleNetV2: WORKS (all 3 stages + stride/regular blocks)
- Neck Ghost-PAN: CRASHES after backbone (memory corruption in workspace)
- Head GFL: Code written, not tested
- Export: DONE (110 layers, 4.5MB)

## Root Cause
Workspace-based memory management causes corruption. The ping-pong buffers (A/B)
and scratch buffer (tmp) overlap or are too small for stage3 intermediates.

## Fix Plan
1. Replace ALL workspace pointer arithmetic with per-tensor malloc/free
   - Backbone: malloc for each conv output, free when no longer needed
   - Neck: already partially done with malloc (r1/r2/r3)
   - This sacrifices some speed (malloc overhead) but guarantees correctness
2. Once forward pass works end-to-end, optimize memory:
   - Profile memory usage to determine actual peak
   - Allocate one big arena with verified offsets
   - Pre-compute all tensor sizes at init time

## Performance Projections (from operator benchmarks)
- DW k=3 convs: ~0.04ms each (AVX-512 + threaded) 
- DW k=5 convs: ~0.5ms each (needs AVX-512 optimization)
- PW 1x1 convs: ~0.07ms each (GEMM)
- Estimated NanoDet total: ~5-8ms (need k=5 DW optimization)
- Target: ~1-3ms (with k=5 DW AVX-512 + optimized memory)

## Key Files
- src/nanodet.c: Forward pass + weight loading + detect
- src/ops.c: DW conv (AVX-512 for k=3), channel shuffle
- tools/export_nanodet.py: ONNX → NN2N format
- weights/nanodet_plus_m_320.bin: Exported weights
