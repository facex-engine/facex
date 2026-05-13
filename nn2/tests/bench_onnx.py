import sys, time, numpy as np, onnxruntime as ort
WEIGHTS_V2 = "tests/minifasnet_v2_27.onnx"
WEIGHTS_V1SE = "tests/minifasnet_v1se_40.onnx"
T_V2 = "tests/test_27.bin"
T_V1SE = "tests/test_40.bin"

def bench(weights_path, test_path, name, iters=100):
    # ONNX Runtime — try AVX-512 BLAS via default CPU EP first.
    s = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
    bgr = np.frombuffer(open(test_path, "rb").read(), dtype=np.uint8).reshape(80, 80, 3)
    x = bgr.astype(np.float32).transpose(2, 0, 1)[None]
    # Warmup
    for _ in range(10): _ = s.run(None, {"input": x})
    # Measure
    t0 = time.perf_counter()
    for _ in range(iters):
        p = s.run(None, {"input": x})[0][0]
    dt = (time.perf_counter() - t0) * 1000 / iters
    print(f"ONNX {name}: probs=[{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]  P(live)={p[1]:.4f}  {dt:.3f} ms/iter ({iters} iters)")
    return dt

dt27 = bench(WEIGHTS_V2, T_V2, "v2 @ 2.7")
dt40 = bench(WEIGHTS_V1SE, T_V1SE, "v1se @ 4.0")
print(f"ENSEMBLE (sum) = {(dt27+dt40):.3f} ms")
