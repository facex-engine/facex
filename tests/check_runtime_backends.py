"""Compare the C API, Python SDK and CLI on real weights and concurrent calls."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np

from facex import FaceX


def check_native_handles(sdk, embedder):
    lib = sdk._lib
    handles = [lib.facex_init(bytes(embedder), None, None) for _ in range(3)]
    pointer = ctypes.POINTER(ctypes.c_float)
    try:
        assert all(handles)
        frames = np.random.default_rng(42).uniform(-1, 1, (8, 112, 112, 3)).astype(np.float32)

        def embed(handle, i):
            out = np.empty(512, dtype=np.float32)
            rc = lib.facex_embed(handle, frames[i].ctypes.data_as(pointer), out.ctypes.data_as(pointer))
            assert rc == 0
            return out

        expected = [embed(handles[0], i) for i in range(len(frames))]
        gate = Barrier(len(frames))

        def repeat(i):
            gate.wait()
            for handle in (handles[0], handles[i % len(handles)], handles[(i + 1) % len(handles)]):
                np.testing.assert_array_equal(embed(handle, i), expected[i])

        with ThreadPoolExecutor(len(frames)) as pool:
            list(pool.map(repeat, range(len(frames))))
    finally:
        for handle in handles:
            lib.facex_free(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "cli", "detector", "embedder"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    weights = dict(embed_weights=args.embedder.resolve(), detect_weights=args.detector.resolve())
    frame = np.fromfile(Path(__file__).with_name("test_face_160.raw"), dtype=np.uint8).reshape(160, 160, 3)
    face = np.ascontiguousarray(frame[:112, :112], dtype=np.float32) / 127.5 - 1
    with FaceX(lib_path=args.library.resolve(), **weights) as native, FaceX(cli_path=args.cli.resolve(), **weights) as child:
        assert (native.backend, child.backend) == ("ctypes", "cli")
        check_native_handles(native, weights["embed_weights"])
        np.testing.assert_array_equal(native.embed(face), child.embed(face))
        for include in (True, False):
            a, b = (engine.detect(frame, embeddings=include) for engine in (native, child))
            assert len(a) == len(b) == 1 and a[0]["score"] > 0.9
            assert set(a[0]) == set(b[0]) and ("embedding" in b[0]) == include
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
        assert child.detect(frame, max_faces=0) == []
    print("Concurrent C handles and Python library and CLI results agree")


if __name__ == "__main__":
    main()
