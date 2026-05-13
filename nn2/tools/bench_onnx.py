"""Benchmark ONNX Runtime YOLOv8n at 320x320 for comparison with nn2."""
import time
import numpy as np
import onnxruntime as ort

MODEL = "yolov8n.onnx"
SIZE = 320
WARMUP = 5
RUNS = 50

print(f"ONNX Runtime {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print(f"Input: {inp.name} {inp.shape} {inp.type}")

# Check if model expects 640 — if so, use 640
h, w = inp.shape[2], inp.shape[3]
if isinstance(h, int):
    SIZE = h
    print(f"Model expects {SIZE}x{SIZE}")

dummy = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)

# Warmup
for _ in range(WARMUP):
    sess.run(None, {inp.name: dummy})

# Benchmark
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    sess.run(None, {inp.name: dummy})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

times.sort()
avg = sum(times) / len(times)
med = times[len(times) // 2]
best = times[0]
p95 = times[int(len(times) * 0.95)]

print(f"\nONNX Runtime @ {SIZE}x{SIZE} ({RUNS} runs):")
print(f"  Average: {avg:.1f} ms ({1000/avg:.1f} FPS)")
print(f"  Median:  {med:.1f} ms")
print(f"  Best:    {best:.1f} ms")
print(f"  P95:     {p95:.1f} ms")
