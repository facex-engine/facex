"""Direct comparison: nn2 vs ONNX Runtime at 320x320."""
import time
import subprocess
import numpy as np
import onnxruntime as ort

RUNS = 50
WARMUP = 10

print("=" * 55)
print("  nn2 vs ONNX Runtime — YOLOv8n @ 320x320")
print("=" * 55)

# --- ONNX Runtime ---
sess = ort.InferenceSession("yolov8n_320.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)

for _ in range(WARMUP):
    sess.run(None, {inp.name: dummy})

times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    sess.run(None, {inp.name: dummy})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

times.sort()
onnx_avg = sum(times) / len(times)
onnx_med = times[len(times) // 2]
onnx_best = times[0]

print(f"\nONNX Runtime {ort.__version__} (CPU):")
print(f"  Average: {onnx_avg:.1f} ms  ({1000/onnx_avg:.0f} FPS)")
print(f"  Median:  {onnx_med:.1f} ms")
print(f"  Best:    {onnx_best:.1f} ms")

# --- nn2 (from last benchmark) ---
nn2_avg = 8.8  # from nn2.exe benchmark
print(f"\nnn2 (pure C, AVX-512):")
print(f"  Average: {nn2_avg:.1f} ms  ({1000/nn2_avg:.0f} FPS)")

# --- Comparison ---
ratio = onnx_avg / nn2_avg
print(f"\n{'=' * 55}")
if ratio > 1:
    print(f"  nn2 is {ratio:.2f}x FASTER than ONNX Runtime")
else:
    print(f"  ONNX Runtime is {1/ratio:.2f}x faster than nn2")
print(f"  nn2: {nn2_avg:.1f}ms vs ONNX RT: {onnx_avg:.1f}ms")
print(f"{'=' * 55}")
