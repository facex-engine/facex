"""Inspect aidos subdirs to identify the runtime."""

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    # find any ML model files
    'powershell -Command "Get-ChildItem e:\\aidos -Include *.onnx,*.engine,*.plan,*.trt,*.tflite,*.param,*.mnn,*.dlc,*.bin,*.xml,*.so,*.dll -Recurse -ErrorAction SilentlyContinue | Select FullName,Length | Select -First 100 | Format-Table -AutoSize"',
    # contents of nn subdir
    'powershell -Command "Get-ChildItem e:\\aidos\\nn -Depth 2 -ErrorAction SilentlyContinue | Select Name,Length | Format-Table -AutoSize"',
    'powershell -Command "Get-ChildItem e:\\aidos\\face -Depth 2 -ErrorAction SilentlyContinue | Select Name,Length | Format-Table -AutoSize"',
    'powershell -Command "Get-ChildItem e:\\aidos\\nexus-sdk-v2 -Depth 1 -ErrorAction SilentlyContinue | Select Name,Length | Format-Table -AutoSize"',
    # 1.txt contents
    'powershell -Command "Get-Content e:\\aidos\\1.txt | Select -First 80"',
    # any README/license files
    'powershell -Command "Get-ChildItem e:\\aidos -Include README*,LICENSE*,*.md -Recurse -ErrorAction SilentlyContinue | Select FullName -First 20 | Format-Table -AutoSize"',
]
for cmd in cmds:
    short = cmd.split('"')[1][:90] if '"' in cmd else cmd[:90]
    print(f"\n$ {short}")
    _, out, err = c.exec_command(cmd, timeout=60)
    o = out.read().decode("utf-8", errors="replace")
    e = err.read().decode("utf-8", errors="replace")
    if o.strip(): print(o[:6000])
    if e.strip(): print("STDERR:", e[:500])

c.close()
