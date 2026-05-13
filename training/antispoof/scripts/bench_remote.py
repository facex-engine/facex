"""Upload weights + test bins, run nn2 + ONNX benchmarks on remote, print table."""

import paramiko, time, re, os
from pathlib import Path

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)
sftp = c.open_sftp()


def remote_mkdir_p(path):
    parts = path.replace("\\", "/").split("/")
    cur = ""
    for p in parts:
        cur = (cur + "/" + p) if cur else p
        try: sftp.stat(cur)
        except FileNotFoundError:
            try: sftp.mkdir(cur)
            except Exception: pass


# === Upload weights and test inputs to remote nn2/weights and nn2/tests ===
TO_UPLOAD = [
    ("D:/apps/facex/nn2/weights/minifasnet_v2_27.bin", "e:/aidos/nn2/weights/minifasnet_v2_27.bin"),
    ("D:/apps/facex/nn2/weights/minifasnet_v1se_40.bin", "e:/aidos/nn2/weights/minifasnet_v1se_40.bin"),
    ("D:/apps/facex/training/data/antispoof/test_27.bin", "e:/aidos/nn2/tests/test_27.bin"),
    ("D:/apps/facex/training/data/antispoof/test_40.bin", "e:/aidos/nn2/tests/test_40.bin"),
    ("D:/apps/facex/wasm/minifasnet_v2_27.onnx", "e:/aidos/nn2/tests/minifasnet_v2_27.onnx"),
    ("D:/apps/facex/wasm/minifasnet_v1se_40.onnx", "e:/aidos/nn2/tests/minifasnet_v1se_40.onnx"),
]

print("=== uploading ===")
for src, dst in TO_UPLOAD:
    if not Path(src).exists():
        print(f"  SKIP missing: {src}"); continue
    remote_mkdir_p(os.path.dirname(dst))
    sftp.put(src, dst)
    print(f"  ✓ {dst}  ({Path(src).stat().st_size} B)")

# Upload the ONNX bench script
ONNX_BENCH = '''import sys, time, numpy as np, onnxruntime as ort
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
'''
sftp.open("e:/aidos/nn2/tests/bench_onnx.py", "w").write(ONNX_BENCH)
print("  ✓ bench_onnx.py")

# === Benchmark nn2 (both models, 200 iters) ===
ITERS = 200
print("\n=== nn2_antispoof.exe ===")

def run(cmd, timeout=120):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")

# Run nn2 — V2
o, e = run(f'cmd /c "cd /d e:\\aidos\\nn2 && nn2_antispoof.exe v2 weights/minifasnet_v2_27.bin tests/test_27.bin {ITERS} 2>&1"')
print(o.strip())
nn2_v2 = o.strip()
if e.strip(): print("ERR:", e.strip())

# Run nn2 — V1SE
o, e = run(f'cmd /c "cd /d e:\\aidos\\nn2 && nn2_antispoof.exe v1se weights/minifasnet_v1se_40.bin tests/test_40.bin {ITERS} 2>&1"')
print(o.strip())
nn2_v1se = o.strip()
if e.strip(): print("ERR:", e.strip())

# === Benchmark ONNX ===
print("\n=== onnxruntime ===")
o, e = run(f'cmd /c "cd /d e:\\aidos\\nn2 && python tests/bench_onnx.py 2>&1"')
print(o.strip())
onnx_block = o.strip()
if e.strip(): print("ERR:", e.strip())

# === Parse + compose final comparison table ===
def parse_nn2(s):
    m = re.search(r"P\\(live\\)=([0-9.]+)\\s+([0-9.]+) ms/iter", s)
    if not m: return None
    return float(m.group(1)), float(m.group(2))

def parse_onnx(s, name):
    m = re.search(rf"ONNX {re.escape(name)}.*?P\\(live\\)=([0-9.]+)\\s+([0-9.]+) ms/iter", s)
    if not m: return None
    return float(m.group(1)), float(m.group(2))

nn2_v2_p, nn2_v2_ms = parse_nn2(nn2_v2) or (None, None)
nn2_v1se_p, nn2_v1se_ms = parse_nn2(nn2_v1se) or (None, None)
onnx_v2_p, onnx_v2_ms = parse_onnx(onnx_block, "v2 @ 2.7") or (None, None)
onnx_v1se_p, onnx_v1se_ms = parse_onnx(onnx_block, "v1se @ 4.0") or (None, None)

print("\\n=== COMPARISON ===")
print(f"{'model':<14} {'engine':<8} {'P(live)':<10} {'ms/iter':<10}")
print("-" * 50)
if nn2_v2_ms is not None:
    print(f"{'MiniFASNetV2':<14} {'nn2':<8} {nn2_v2_p:<10.4f} {nn2_v2_ms:<10.3f}")
if onnx_v2_ms is not None:
    print(f"{'MiniFASNetV2':<14} {'onnx':<8} {onnx_v2_p:<10.4f} {onnx_v2_ms:<10.3f}")
if nn2_v2_ms and onnx_v2_ms:
    print(f"  speedup nn2/onnx = {onnx_v2_ms/nn2_v2_ms:.2f}x")
print()
if nn2_v1se_ms is not None:
    print(f"{'MiniFASNetV1SE':<14} {'nn2':<8} {nn2_v1se_p:<10.4f} {nn2_v1se_ms:<10.3f}")
if onnx_v1se_ms is not None:
    print(f"{'MiniFASNetV1SE':<14} {'onnx':<8} {onnx_v1se_p:<10.4f} {onnx_v1se_ms:<10.3f}")
if nn2_v1se_ms and onnx_v1se_ms:
    print(f"  speedup nn2/onnx = {onnx_v1se_ms/nn2_v1se_ms:.2f}x")

if nn2_v2_ms and nn2_v1se_ms and onnx_v2_ms and onnx_v1se_ms:
    print()
    print(f"Ensemble: nn2 {nn2_v2_ms+nn2_v1se_ms:.2f} ms vs onnx {onnx_v2_ms+onnx_v1se_ms:.2f} ms"
          f"  ({(onnx_v2_ms+onnx_v1se_ms)/(nn2_v2_ms+nn2_v1se_ms):.2f}x speedup)")

sftp.close(); c.close()
