"""Check device-specific code paths in nn (decoder) and nn2."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    'powershell -Command "Get-ChildItem e:/aidos/nn/src -ErrorAction SilentlyContinue | Where-Object {$_.Name -match \\"neon|arm|sse|avx|simd|cpu\\"} | Select Name,Length"',
    'powershell -Command "Get-ChildItem e:/aidos/nn -Filter Makefile* -ErrorAction SilentlyContinue | Select Name"',
    'powershell -Command "Get-Content e:/aidos/nn/Makefile -ErrorAction SilentlyContinue | Select -First 60"',
    'powershell -Command "Select-String -Path e:/aidos/nn/src/*.c,e:/aidos/nn/src/*.h -Pattern \\"__aarch64__|__arm__|__ARM_NEON|__riscv|esp32|raspberry\\" -ErrorAction SilentlyContinue | Select Path,Line -First 20"',
    'powershell -Command "Get-Content e:/aidos/nn/README.md -ErrorAction SilentlyContinue | Select -Skip 100 -First 40"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:120]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    o = out.read().decode("utf-8", errors="replace")
    print(o[:3000] if o.strip() else "(empty)")
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:200])
c.close()
