"""SSH to admin@10.0.0.3 and inspect e:/aidos."""

import paramiko, sys

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    'powershell -Command "Get-ChildItem e:\\aidos -Recurse -Depth 2 | Select-Object Mode,Length,Name,FullName | Format-Table -AutoSize"',
    'powershell -Command "Get-ChildItem e:\\aidos | Measure-Object -Property Length -Sum"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:120]}")
    _, out, err = c.exec_command(cmd)
    o = out.read().decode("utf-8", errors="replace")
    e = err.read().decode("utf-8", errors="replace")
    if o.strip(): print(o[:4000])
    if e.strip(): print("STDERR:", e[:1000])

c.close()
