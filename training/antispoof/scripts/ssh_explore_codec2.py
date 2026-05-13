"""More codec/research depth."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    'powershell -Command "Get-Content e:/aidos/nexus-format/README.md -ErrorAction SilentlyContinue"',
    'powershell -Command "Get-ChildItem e:/aidos/nexus-format/docs -ErrorAction SilentlyContinue | Select Name,Length"',
    # nn2 specifics
    'powershell -Command "Get-Content e:/aidos/nn2/README.md -ErrorAction SilentlyContinue | Select -First 80"',
    # H:/bz/bz1
    'powershell -Command "Test-Path h:/bz/bz1; Get-ChildItem h:/bz/bz1 -ErrorAction SilentlyContinue | Select Name,Length -First 30"',
    # H:/bz top research
    'powershell -Command "Get-ChildItem h:/bz -Filter *.md -ErrorAction SilentlyContinue | Select Name -First 20"',
    # NVR-related
    'powershell -Command "Get-ChildItem e:/aidos/nvr-archive -ErrorAction SilentlyContinue | Select Name,Length"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:120]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    o = out.read().decode("utf-8", errors="replace")
    print(o[:6000] if o.strip() else "(empty)")
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:300])
c.close()
