"""Inspect nn2 source to determine MiniFASNet compatibility."""

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    # nn2 src directory layout
    'powershell -Command "Get-ChildItem e:\\aidos\\nn2\\src -ErrorAction SilentlyContinue | Where-Object {-not $_.PSIsContainer} | Select Name,Length | Sort Name | Format-Table -AutoSize"',
    # find op implementations
    'powershell -Command "Get-ChildItem e:\\aidos\\nn2\\src -Filter *.c -Recurse -ErrorAction SilentlyContinue | Select Name,Length | Sort Name | Format-Table -AutoSize"',
    # README in nn2
    'powershell -Command "Get-Content e:\\aidos\\nn2\\README.md -ErrorAction SilentlyContinue | Select -First 200"',
    # Look at headers
    'powershell -Command "Get-ChildItem e:\\aidos\\nn2 -Filter *.h -Recurse -ErrorAction SilentlyContinue | Select Name | Format-Table -AutoSize"',
    # Check ops via grep
    'powershell -Command "Select-String -Path e:\\aidos\\nn2\\src\\*.c,e:\\aidos\\nn2\\src\\*.h -Pattern \\"prelu|sigmoid|depthwise|globalavg|squeeze\\" -ErrorAction SilentlyContinue | Select Path,LineNumber,Line -First 30"',
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
