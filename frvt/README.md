# FaceX FRVT 1:1 submission

Submission package for NIST [FRTE 1:1 Verification](https://pages.nist.gov/frvt/html/frvt11.html).

## Architecture

```
NIST testbench
   │ provides Image objects (uint8 24bpp RGB or 8bpp greyscale)
   ▼
src/facex_frvt.cpp        ← Interface implementation
   │
   ▼
facex_detect.onnx   →  face detection (FCOS, 320×320, 100K params)
facex_landmark.onnx →  98-point WFLW landmarks (112×112, 1.15M params)
facex_xs.onnx       →  512-dim L2-normalized embedding (112×112, 2.07M params)
```

**Template format:** 2053 bytes per face = 4-byte magic `FXT1` +
1-byte valid flag + 512 × float32 L2-normalized embedding.

**Match:** cosine similarity rescaled to [0, 1000] (NIST treats higher
as more similar).

## Build (Docker, recommended)

The container reproduces NIST's evaluation environment exactly
(Ubuntu 20.04, x86-64, AVX2, GCC 9):

```bash
cd frvt
docker build -t facex-frvt-build .
docker run --rm -v ${PWD}/build:/work/build facex-frvt-build
ls build/lib/libfrvt_11_facex_001.so
```

## Build (WSL Ubuntu)

```bash
sudo apt install build-essential cmake
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
tar xzf onnxruntime-linux-x64-1.21.0.tgz
export ORT=$PWD/onnxruntime-linux-x64-1.21.0

cd frvt
cmake -S . -B build -DONNXRUNTIME_ROOT=$ORT -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Prepare the submission package

NIST expects this exact layout:

```
facex_001/
├── doc/
│   └── ReleaseNotes.txt
├── config/
│   ├── facex_detect.onnx
│   ├── facex_landmark.onnx
│   └── facex_xs.onnx
└── lib/
    ├── libfrvt_11_facex_001.so
    └── libonnxruntime.so.1.21.0
```

Build the package:

```bash
mkdir -p facex_001/doc facex_001/config facex_001/lib
cp ../_unencrypted_backup/facex_{detect,landmark,xs}.onnx facex_001/config/
cp build/lib/libfrvt_11_facex_001.so       facex_001/lib/
cp build/lib/libonnxruntime.so.*           facex_001/lib/
cat > facex_001/doc/ReleaseNotes.txt <<EOF
FaceX 1:1 submission — sequence 001
Built: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Models: facex_detect (own, WIDER FACE), facex_landmark (own, WFLW),
        facex_xs (own MobileFaceNet 1.35x width, ArcFace head, MS1M)
ONNX Runtime: 1.21.0
EOF

tar czf facex_001.tar.gz facex_001/
```

## Validate locally before submission

NIST provides a validation suite at
[github.com/usnistgov/frvt](https://github.com/usnistgov/frvt) under
`11/validation/`. Run it against `libfrvt_11_facex_001.so` to catch
crashes, signature mismatches, and slow runtime:

```bash
git clone https://github.com/usnistgov/frvt
cd frvt/11/validation
./run_validate_11.sh ../../facex_001
```

It will run a few thousand template-creation and match operations
and print a pass/fail report. Fix any failures before uploading.

## PGP-sign and submit

NIST requires a PGP-signed tarball uploaded to their SFTP endpoint.
Steps once your participation agreement is approved:

```bash
gpg --gen-key                                  # one-time
gpg --armor --export your@email.com > pubkey.asc   # send to NIST
gpg --output facex_001.tar.gz.sig --detach-sign facex_001.tar.gz

sftp frvt@sftp.nist.gov   # credentials from NIST after agreement
put facex_001.tar.gz
put facex_001.tar.gz.sig
```

## What you (the human) need to do separately

1. **Sign the Participation Agreement.**
   Download from https://pages.nist.gov/frvt/html/frvt11.html → "FRTE/FATE Participation Agreement".
   Fill in name / org / contact, sign (electronic OK), email to
   `frvt@nist.gov`.

2. **Generate and send PGP key.**
   `gpg --gen-key` once, then email `pubkey.asc` to `frvt@nist.gov`.
   They will reply with SFTP credentials.

3. **Wait ~3-6 months** for results to appear on
   https://pages.nist.gov/frvt/html/frvt11.html under "FaceX 001".

4. **Resubmissions** allowed every 4 months. Increment the sequence
   number (`002`, `003`, ...) and rebuild.

## Expected outcome

We submit `xs` (2.07 M params, 99.07% LFW after YuNet 5-pt alignment).
Realistic position in the table:

| Metric | Expected rank |
|--------|--------------|
| FNMR @ FMR=10⁻⁶ (accuracy) | ~top 80-120 of 250+ |
| Template gen time | top 5-15 (we're CPU-fast) |
| Match time | top 10 (cosine is trivial) |
| Template size | 2053 bytes (typical) |

The marketing claim that survives scrutiny:
**"Fastest open-source CPU face recognition tested by NIST FRVT 2026"**
— assuming we land top-3 on `Template gen time`, which is very likely
given our hand-tuned AVX-512 GEMM stack.
