"""Explore e:/aidos codec/decoder/encoder/nvr projects."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    # NexusDecode README
    'powershell -Command "Get-Content e:/aidos/nn/README.md -ErrorAction SilentlyContinue | Select -First 120"',
    # NexusEncode
    'powershell -Command "Get-ChildItem e:/aidos/nn-encoder -ErrorAction SilentlyContinue | Select Name,Length"',
    'powershell -Command "Get-Content e:/aidos/nn-encoder/README.md -ErrorAction SilentlyContinue | Select -First 80"',
    # NXV format
    'powershell -Command "Get-ChildItem e:/aidos/nexus-format -ErrorAction SilentlyContinue | Select Name,Length"',
    'powershell -Command "Get-Content e:/aidos/nexus-format/README.md -ErrorAction SilentlyContinue | Select -First 80"',
    # nexus-format docs
    'powershell -Command "Get-ChildItem e:/aidos/nexus-format/docs -ErrorAction SilentlyContinue | Select Name,Length"',
    # research base
    'powershell -Command "Get-ChildItem e:/aidos/research -ErrorAction SilentlyContinue | Select Name,Length | Select -First 30"',
    # h265 / mp4rescue
    'powershell -Command "Get-ChildItem e:/aidos/h265, e:/aidos/mp4rescue -ErrorAction SilentlyContinue | Select Name,Length"',
    # Sizes
    'powershell -Command "& { $dirs=@(\\"nn\\",\\"nn-encoder\\",\\"nn2\\",\\"nexus-format\\",\\"nexus-sdk\\",\\"nexus-sdk-v2\\",\\"face\\",\\"v3\\",\\"webtortsp\\"); foreach ($d in $dirs) { $p=\\"e:/aidos/$d\\"; if (Test-Path $p) { Write-Output \\"$d\\" } } }"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:120]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    o = out.read().decode("utf-8", errors="replace")
    print(o[:4500] if o.strip() else "(empty)")
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:400])
c.close()
