"""Benchmark ONNX Runtime YOLOv8n at 320x320 — resize input to match nn2."""
import time
import numpy as np
import onnxruntime as ort

MODEL = "yolov8n.onnx"
WARMUP = 5
RUNS = 50

print(f"ONNX Runtime {ort.__version__}")

# Use 320x320 input regardless of model's expected size
# ONNX RT supports dynamic shapes for YOLOv8
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 0  # auto
sess = ort.InferenceSession(MODEL, sess_opts, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print(f"Input: {inp.name} {inp.shape}")

# Test both 320 and 640
for SIZE in [320, 640]:
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

    print(f"\n=== ONNX Runtime @ {SIZE}x{SIZE} ({RUNS} runs) ===")
    print(f"  Average: {avg:.1f} ms ({1000/avg:.1f} FPS)")
    print(f"  Median:  {med:.1f} ms")
    print(f"  Best:    {best:.1f} ms")
