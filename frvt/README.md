# FaceX adapter for FRTE 1:1 verification

This directory implements the NIST FRVT 1:1 interface for Linux x86-64. The adapter uses ONNX Runtime with three model files from the application configuration directory.

| File | Purpose |
|---|---|
| `facex_detect.onnx` | Face detection on a 320 × 320 input |
| `facex_landmark.onnx` | 98 landmarks used to derive five alignment points |
| `facex_xs.onnx` | 512-component XS embedding of an aligned 112 × 112 RGB face |

The adapter selects the largest detection in each image and averages embeddings for up to five images of one person. Each 2,053-byte template contains the four-byte `FXT1` header, one validity byte and 512 float32 components. Valid unit embeddings are compared with cosine similarity rescaled to [0, 1000]. The public 256-component Nano model has a different output shape. [Implementation](src/facex_frvt.cpp) · [Nano contract](../MFN_README.md#public-nano-release)

## Build on Linux

Set `ORT` to a Linux x86-64 ONNX Runtime SDK containing `include/` and `lib/`. The example prepares version 1.21.0 from Microsoft’s NuGet package using `curl` and `unzip`. From the repository root:

```sh
export ORT="$PWD/frvt/ort-sdk"
sh frvt/prepare_ort.sh "$ORT"
cmake -S frvt -B frvt/build -DONNXRUNTIME_ROOT="$ORT" -DCMAKE_BUILD_TYPE=Release
cmake --build frvt/build --parallel
cmake --install frvt/build --prefix "$PWD/frvt/bundle"
```

The install target produces `frvt/bundle/lib/libfrvt_11_facex_001.so`. To build with the Ubuntu 20.04 container and its pinned ONNX Runtime SDK, use the following commands from the repository root:

```sh
docker build -t facex-frvt-build frvt
mkdir -p frvt/bundle
docker run --rm -v "$PWD/frvt/bundle:/work/build" facex-frvt-build
```

## Prepare and validate a bundle

Place the shared adapter and the matching ONNX Runtime libraries in `lib/`, the three ONNX files in `config/`, and the release notes in `doc/`. Record the model hashes, runtime version and adapter commit so the package can be reproduced.

The alignment regression covers translation, rotation, border pixels and degenerate landmarks. Run it from the repository root:

```sh
c++ -O2 -std=c++11 tests/frvt_alignment.cpp -o /tmp/frvt-alignment
/tmp/frvt-alignment
```

Use the NIST validation harness for the complete package and consult the program for its current submission procedure. An official evaluation result identifies its submission, test population and operating point. The project's historical LFW figures are reported separately. [NIST harness](https://github.com/usnistgov/frvt) · [FRTE 1:1](https://pages.nist.gov/frvt/html/frvt11.html) · [FaceX measurements](../README.md#models-and-measurements)
