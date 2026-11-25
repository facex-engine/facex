"""Compare the release ONNX to two native builds. Run from the repository root.

Requires Linux x86 with AVX2/FMA, GCC, NumPy, Pillow, onnx and onnxruntime.
Pass the local facex_nano.onnx from the facex-nano-1.0 release.
"""
import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/scripts"))
from arch import ARCH_NANO
from binformat import _pack_arch

RELEASE_SHA256 = "bd0094143bb4c7d04a76dcdad4e332f5985847b921ec4d6678c3fd89aaea3889"
BASE = "b21cc3d2f292818b23879f988b32dce34b335895"
BUFFER = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


def export_folded_weights(model, path):
    graph = onnx.load(model).graph
    initializers = {item.name: onnx.numpy_helper.to_array(item) for item in graph.initializer}
    tensors = []
    for node in graph.node:
        if node.op_type == "Conv":
            weights, bias = (initializers[name] for name in node.input[1:])
            # ONNX already folded BN. Make the native loader apply unit scale.
            tensors.extend((weights, np.ones_like(bias), bias,
                            np.zeros_like(bias), np.full_like(bias, 1 - 1e-5)))
        elif node.op_type == "PRelu":
            tensors.append(initializers[node.input[1]])
    with path.open("wb") as out:
        out.write(b"EFM3" + struct.pack("<I", 3) + _pack_arch(ARCH_NANO))
        out.write(struct.pack("<II", 0, len(tensors)))
        for tensor in tensors:
            data = tensor.astype("<f4").tobytes()
            out.write(struct.pack("<I", len(data)))
            out.write(data)


def build_engine(directory, name, source, weights):
    source_file = directory / (name + ".c")
    source_file.write_text(source + """
static MfnEngine benchmark_engine;
int setup(const char *path) { return mfn_engine_init(path, &benchmark_engine); }
void evaluate(const float *x, float *y) { mfn_engine_forward(&benchmark_engine, x, y); }
void release(void) { mfn_engine_free(&benchmark_engine); }
""")
    binary = directory / (name + ".so")
    subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-fPIC", "-shared",
                    "-I", str(ROOT / "src"), str(source_file), "-lm", "-o", str(binary)],
                   check=True)
    lib = ctypes.CDLL(str(binary))
    lib.setup.argtypes = [ctypes.c_char_p]
    lib.evaluate.argtypes = [BUFFER, BUFFER]
    lib.evaluate.restype = None
    if lib.setup(os.fsencode(weights)) != 0:
        raise RuntimeError("Native EFM3 initialization failed")
    return lib


def timed_forward(engine, image, embedding):
    start = time.thread_time_ns()
    for _ in range(12):
        engine.evaluate(image, embedding)
    return (time.thread_time_ns() - start) / 12e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != RELEASE_SHA256:
        parser.error("The file differs from the measured Nano release")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    baseline = subprocess.check_output(["git", "show", BASE + ":src/facex_mfn.c"],
                                       cwd=ROOT, text=True)
    # Both sides need the correct workspace dimensions for a valid comparison.
    baseline = baseline.replace("(H - e->arch.stem_kernel) / e->arch.stem_stride + 1",
                                "(H + 2 * e->stem.padding - e->stem.kernel) / e->stem.stride + 1")
    current = (ROOT / "src/facex_mfn.c").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        weights = directory / "nano.bin"
        export_folded_weights(args.model, weights)
        engines = [build_engine(directory, name, source, weights)
                   for name, source in (("baseline", baseline), ("current", current))]
        outputs = [np.empty(256, "f") for _ in engines]
        files = sorted((ROOT / "tests/golden").glob("*.npz"))
        if not files:
            raise RuntimeError("Missing golden images")
        errors, cosines = [], []
        for file in files:
            with np.load(file) as fixture:
                image = Image.fromarray(fixture["image"]).resize((112, 112))
            x = np.ascontiguousarray((np.asarray(image, dtype="f").transpose(2, 0, 1)
                                      - 127.5) / 128)
            reference = session.run(None, {"input": x[None]})[0].ravel()
            engines[1].evaluate(x, outputs[1])
            current_out = outputs[1].astype("d")
            errors.append(np.max(abs(current_out - reference)))
            cosines.append(np.dot(current_out, reference) /
                           (np.linalg.norm(current_out) * np.linalg.norm(reference.astype("d"))))
        print(f"{len(files)} images, max error {max(errors):.7g}, min cosine {min(cosines):.10f}")
        if not max(errors) < 1e-4 or not min(cosines) > .99999:
            raise AssertionError("ONNX agreement failed")
        rng = np.random.default_rng(20261125)
        for _ in range(3):
            image = rng.uniform(-1, 1, (3, 112, 112)).astype("f")
            times = [[], []]
            for i, engine in enumerate(engines):
                engine.evaluate(image, outputs[i])
            np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-4, rtol=0)
            for pair in range(9):
                for i in ([1, 0] if pair % 2 else [0, 1]):
                    times[i].append(timed_forward(engines[i], image, outputs[i]))
            print("CPU ms per image", [round(statistics.median(t), 4) for t in times])
        for engine in engines:
            engine.release()


if __name__ == "__main__":
    main()
