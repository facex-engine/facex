"""Find gcc by checking common install paths."""
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)

cmds = [
    # Check common gcc paths
    'powershell -Command "$candidates = @(\'c:/mingw64/bin/gcc.exe\', \'c:/msys64/mingw64/bin/gcc.exe\', \'c:/msys64/ucrt64/bin/gcc.exe\', \'c:/program files/mingw-w64/bin/gcc.exe\', \'d:/mingw64/bin/gcc.exe\', \'c:/tools/mingw64/bin/gcc.exe\', \'c:/programdata/chocolatey/bin/gcc.exe\'); foreach ($p in $candidates) { if (Test-Path $p) { Write-Output \\"FOUND: $p\\" } }"',
    # Maybe in user profile
    'powershell -Command "Get-ChildItem c:/msys64 -ErrorAction SilentlyContinue | Select Name"',
    'powershell -Command "Get-ChildItem c:/mingw64 -ErrorAction SilentlyContinue | Select Name"',
    # What does last successful build use?
    'powershell -Command "Get-Content e:/aidos/nn2/build.sh | Select -First 5"',
    # Check env var
    'powershell -Command "$env:Path -split \';\' | Where-Object { $_ -like \'*gcc*\' -or $_ -like \'*ming*\' -or $_ -like \'*msys*\' }"',
]
for cmd in cmds:
    print(f"\n$ {cmd[:100]}")
    _, out, err = c.exec_command(cmd, timeout=30)
    o = out.read().decode("utf-8", errors="replace")
    print(o[:2000] if o.strip() else "(empty)")
    e = err.read().decode("utf-8", errors="replace")
    if e.strip(): print("ERR:", e[:500])
c.close()
