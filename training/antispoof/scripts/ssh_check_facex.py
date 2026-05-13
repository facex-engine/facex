"""Inspect facex_detect.h and facex.h to see if there's an own face detector."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    'powershell -Command "Get-Content e:/aidos/nexus-sdk-v2/include/facex_detect.h"',
    'powershell -Command "Get-Content e:/aidos/nexus-sdk-v2/include/facex.h"',
    'powershell -Command "Get-Content e:/aidos/nexus-sdk-v2/include/facex_v2.h"',
    # Look for face-detection-specific weights
    'powershell -Command "Get-ChildItem e:/aidos/nexus-sdk-v2/weights -ErrorAction SilentlyContinue | Select Name,Length"',
    # Look for FaceX detection source/training
    'powershell -Command "Get-ChildItem e:/aidos/face -Recurse -ErrorAction SilentlyContinue -Filter *.bin -Depth 3 | Select Name,FullName,Length -First 30"',
    'powershell -Command "Get-ChildItem e:/aidos/nexus-sdk-v2/src/facex -ErrorAction SilentlyContinue | Select Name,Length"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:100]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    print(out.read().decode("utf-8", errors="replace")[:5000])
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:500])
c.close()
