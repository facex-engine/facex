"""Mirror nn2 source tree from remote 10.0.0.3 to local D:/apps/facex/nn2."""

import os, stat, sys, posixpath
from pathlib import Path
import paramiko

LOCAL = Path("D:/apps/facex/nn2")
REMOTE = "/cygdrive/e/aidos/nn2"   # try cygwin-style first; fall back below
ALT_REMOTE = "e:/aidos/nn2"        # SFTP often accepts forward slashes on Windows

# Skip these — they're build artifacts / images / huge YUV
SKIP_DIRS = {".git", ".gitignore", "__pycache__"}
SKIP_EXT = {".yuv", ".gcda", ".o", ".obj", ".lib", ".a", ".exe", ".dll",
            ".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".raw",
            ".h264", ".h265", ".bit", ".bin", ".pth", ".onnx",
            ".md5", ".pyc", ".zip", ".rar"}

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.0.0.3", port=22, username="admin", password="1235", timeout=15)
sftp = c.open_sftp()


def walk(remote_root):
    """Yield (remote_path, local_path) for source-ish files only."""
    stack = [remote_root]
    while stack:
        cur = stack.pop()
        try:
            entries = sftp.listdir_attr(cur)
        except Exception as e:
            print(f"  cannot listdir {cur}: {e}", file=sys.stderr)
            continue
        for ent in entries:
            name = ent.filename
            if name in SKIP_DIRS: continue
            full = posixpath.join(cur, name)
            if stat.S_ISDIR(ent.st_mode):
                stack.append(full)
            else:
                ext = os.path.splitext(name)[1].lower()
                if ext in SKIP_EXT: continue
                if ent.st_size > 2_000_000: continue
                rel = posixpath.relpath(full, remote_root)
                yield full, rel, ent.st_size


def try_root(r):
    try:
        sftp.stat(r); return True
    except FileNotFoundError: return False


root = None
for cand in [ALT_REMOTE, REMOTE]:
    if try_root(cand): root = cand; break
if root is None:
    print("no remote root found"); sys.exit(1)
print(f"remote root: {root}")

LOCAL.mkdir(parents=True, exist_ok=True)
n = 0; total = 0
for full, rel, sz in walk(root):
    local_path = LOCAL / rel.replace("/", os.sep)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get(full, str(local_path))
        n += 1; total += sz
        if n % 50 == 0: print(f"  {n} files, {total/1024:.0f} KB")
    except Exception as e:
        print(f"  failed {rel}: {e}")
print(f"\nmirrored {n} files, {total/1024/1024:.1f} MB to {LOCAL}")

sftp.close(); c.close()
