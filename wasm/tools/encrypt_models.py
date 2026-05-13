"""Encrypt all .onnx models in wasm/ with AES-256-GCM.

Output format per file:
  [12 bytes IV/nonce][N bytes ciphertext + 16 bytes tag]

Random IV per file. Same key for all (rotate as needed).

Run:
  pip install cryptography
  python tools/encrypt_models.py
"""

import os, secrets, struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

WASM = Path(__file__).resolve().parent.parent
TARGETS = [
    "facex_detect.onnx",
    "facex_landmark.onnx",
    "facex_3d_landmarks.onnx",
    "facex_nano.onnx",
    "facex_tiny.onnx",
    "minifasnet_v2_27.onnx",
    "minifasnet_v1se_40.onnx",
]

KEY_FILE = WASM.parent / ".model_key"


def load_or_create_key():
    if KEY_FILE.exists():
        return bytes.fromhex(KEY_FILE.read_text().strip())
    key = secrets.token_bytes(32)
    KEY_FILE.write_text(key.hex())
    print(f"  generated new key → {KEY_FILE}")
    return key


def encrypt(src: Path, dst: Path, key: bytes):
    plain = src.read_bytes()
    iv = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(iv, plain, None)   # ct = ciphertext || tag(16)
    dst.write_bytes(iv + ct)
    print(f"  ✓ {src.name} ({len(plain)/1024:.0f} KB) → {dst.name} ({(len(iv)+len(ct))/1024:.0f} KB)")


def main():
    key = load_or_create_key()
    print(f"key (32 bytes, hex): {key.hex()}")
    print(f"  → first 8: {key.hex()[:16]}")
    print(f"  → middle 8: {key.hex()[16:32]}")
    print(f"  → last 16: {key.hex()[32:]}")
    print()

    for name in TARGETS:
        src = WASM / name
        if not src.exists():
            print(f"  SKIP missing: {name}"); continue
        dst = WASM / (Path(name).stem + ".enc")
        encrypt(src, dst, key)


if __name__ == "__main__":
    main()
