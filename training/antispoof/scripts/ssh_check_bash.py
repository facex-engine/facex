import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    'powershell -Command "Get-Command bash | Select-Object Source,Path"',
    'powershell -Command "where.exe bash 2>$null"',
    'powershell -Command "bash -c \\"which gcc 2>&1; ls /c/mingw64/bin/gcc* 2>&1 | head -3\\""',
    # Try direct windows-style
    'powershell -Command "& c:/mingw64/bin/gcc.exe --version 2>&1 | Select -First 1"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:100]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    print(out.read().decode("utf-8", errors="replace")[:1500])
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:500])
c.close()
