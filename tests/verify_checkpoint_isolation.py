"""Compare alternating native handles to freshly started single-checkpoint CLI processes."""
import argparse
import ctypes
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "cli", "weights"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.facex_init.argtypes = [ctypes.c_char_p] * 3
    lib.facex_init.restype = ctypes.c_void_p
    lib.facex_free.argtypes = [ctypes.c_void_p]
    floats = ctypes.POINTER(ctypes.c_float)
    lib.facex_embed.argtypes = [ctypes.c_void_p, floats, floats]
    source = args.weights.read_bytes()
    assert source[:4] == b"EFXS" and int.from_bytes(source[8:12], "little") == 6144
    images = np.random.default_rng(518).uniform(-1,1,(4,112,112,3)).astype("f")
    with tempfile.TemporaryDirectory() as directory:
        variant = bytearray(source)
        np.frombuffer(variant, "<f4", 1536, 12)[:] *= -1
        alternate = Path(directory) / "alternate.bin"
        alternate.write_bytes(variant)
        paths = (args.weights, alternate)
        refs = [subprocess.check_output([str(args.cli.resolve()), str(p), "--server"],
                                        input=images.tobytes()) for p in paths]
        assert refs[0] != refs[1]
        handles = [lib.facex_init(bytes(p), None, None) for p in paths]
        assert all(handles)
        try:
            for i, image in enumerate(images):
                for j in (0, 1, 1, 0):
                    out = np.empty(512, dtype="f")
                    assert lib.facex_embed(handles[j], image.ctypes.data_as(floats),
                                           out.ctypes.data_as(floats)) == 0
                    assert out.tobytes() == refs[j][i*2048:(i+1)*2048]
        finally:
            for handle in handles:
                lib.facex_free(handle)


if __name__ == "__main__":
    main()
