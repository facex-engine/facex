"""Find existing WASM artifacts in aidos."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    # WASM files anywhere in aidos
    'powershell -Command "Get-ChildItem e:/aidos -Filter *.wasm -Recurse -Depth 6 -ErrorAction SilentlyContinue | Select FullName,Length"',
    # JS wrappers
    'powershell -Command "Get-ChildItem e:/aidos -Filter facex*.js -Recurse -Depth 6 -ErrorAction SilentlyContinue | Select FullName,Length"',
    'powershell -Command "Get-ChildItem e:/aidos -Filter detect*.js -Recurse -Depth 6 -ErrorAction SilentlyContinue | Select FullName,Length"',
    # Anything FaceX/det related
    'powershell -Command "Get-ChildItem e:/aidos/face/lib -Recurse -ErrorAction SilentlyContinue | Select FullName,Length"',
    'powershell -Command "Get-ChildItem e:/aidos/face/static -Recurse -ErrorAction SilentlyContinue | Select FullName,Length | Select -First 40"',
    # Anything that mentions emscripten
    'powershell -Command "Get-ChildItem e:/aidos -Filter *emcc* -Recurse -ErrorAction SilentlyContinue"',
    'powershell -Command "Select-String -Path e:/aidos/nn2/Makefile,e:/aidos/nn2/build*.sh -Pattern \\"emscripten|emcc|wasm\\" -ErrorAction SilentlyContinue | Select-Object Path,Line"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:120]}")
    _, out, err = c.exec_command(cmd, timeout=60)
    o = out.read().decode("utf-8", errors="replace")
    print(o[:4000] if o.strip() else "(empty)")
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:300])
c.close()
