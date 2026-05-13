"""Push antispoof files to remote nn2 and run build."""

import paramiko
from pathlib import Path

REMOTE_ROOT = "e:/aidos/nn2"
LOCAL_ROOT = Path("D:/apps/facex/nn2")

# Files we touched / added
FILES = [
    "include/nn2_antispoof.h",
    "src/nn2_antispoof_ops.h",
    "src/antispoof_ops.c",
    "src/minifasnet.c",
    "src/main_antispoof.c",
    "tools/export_minifasnet.py",
    "build_antispoof.sh",
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)
sftp = c.open_sftp()


def remote_mkdir_p(path):
    """Make all intermediate directories on remote."""
    parts = path.split("/")
    cur = ""
    for p in parts:
        cur = (cur + "/" + p) if cur else p
        try: sftp.stat(cur)
        except FileNotFoundError:
            try: sftp.mkdir(cur)
            except Exception as e: print(f"mkdir {cur}: {e}")


print("=== uploading ===")
for rel in FILES:
    local = LOCAL_ROOT / rel
    remote = f"{REMOTE_ROOT}/{rel}"
    if not local.exists():
        print(f"  SKIP (missing): {rel}"); continue
    remote_mkdir_p(str(Path(remote).parent).replace("\\", "/"))
    sftp.put(str(local), remote)
    print(f"  ✓ {rel}  ({local.stat().st_size} B)")

print("\n=== building ===")
# Default `bash` on remote is WSL (Linux). Need Git Bash; avoid quoting
# trouble by using the 8.3 short path.
cmd = 'C:/PROGRA~1/Git/bin/bash.exe -c "cd /e/aidos/nn2 && bash build_antispoof.sh 2>&1"'
_, out, err = c.exec_command(cmd, timeout=300)
o = out.read().decode("utf-8", errors="replace")
e = err.read().decode("utf-8", errors="replace")
print(o[-6000:] if len(o) > 6000 else o)
if e.strip(): print("STDERR:", e[-3000:])

sftp.close(); c.close()
