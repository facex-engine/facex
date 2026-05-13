"""Find gcc on remote."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    'powershell -Command "Get-Command gcc -ErrorAction SilentlyContinue | Select-Object Source"',
    'powershell -Command "where.exe gcc 2>&1"',
    'powershell -Command "Get-ChildItem c:/, d:/, e:/ -Filter gcc.exe -Recurse -ErrorAction SilentlyContinue -Depth 4 2>&1 | Select-Object FullName -First 10"',
    'powershell -Command "bash -c \\"which gcc; gcc --version 2>&1 | head -1\\""',
    'powershell -Command "cd e:/aidos/nn2; cat build.sh | head -5"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:100]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    print(out.read().decode("utf-8", errors="replace")[:1500])
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:300])
c.close()
