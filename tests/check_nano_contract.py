"""Compare the published Nano ONNX to the EFM3 loader through its public C API."""
import ctypes
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bench_mfn_release import export_folded_weights


def main():
    model, library, weights = map(Path, sys.argv[1:])
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "input" and session.get_inputs()[0].shape == [1, 3, 112, 112]
    assert session.get_outputs()[0].name == "embedding" and session.get_outputs()[0].shape == [1, 256]
    export_folded_weights(model, weights)
    lib = ctypes.CDLL(str(library.resolve()))
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.mfn_open.argtypes, lib.mfn_open.restype = [ctypes.c_char_p], ctypes.c_void_p
    lib.mfn_embedding_dim.argtypes = [ctypes.c_void_p]
    lib.mfn_engine_forward.argtypes, lib.mfn_engine_forward.restype = [ctypes.c_void_p, ptr, ptr], None
    lib.mfn_close.argtypes, lib.mfn_close.restype = [ctypes.c_void_p], None
    handle = lib.mfn_open(bytes(weights))
    assert handle
    try:
        assert lib.mfn_embedding_dim(handle) == 256
        x = np.linspace(-1, 1, 3 * 112 * 112, dtype=np.float32).reshape(1, 3, 112, 112)
        expected = session.run(["embedding"], {"input": x})[0][0]
        out = np.empty(256, dtype=np.float32)
        lib.mfn_engine_forward(handle, x, out)
        assert abs(np.linalg.norm(expected) - 1) < 1e-5
        np.testing.assert_allclose(out, expected, atol=5e-5, rtol=0)
        print(f"Nano returns {out.size} components; maximum native delta {np.max(np.abs(out - expected)):.8g}")
    finally:
        lib.mfn_close(handle)


if __name__ == "__main__":
    main()
