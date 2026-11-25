# FaceX Python

Python binding for the [FaceX](https://github.com/facex-engine/facex) face detection + recognition library.

## Install

```bash
cd python/
pip install -e .

# With Pillow for image loading:
pip install -e ".[image]"
```

## Quick start

```python
import numpy as np
from facex import FaceX

# Initialize (auto-finds library and weights)
fx = FaceX()

# Load an image as RGB uint8 numpy array (H, W, 3)
from PIL import Image
image = np.array(Image.open("photo.jpg").convert("RGB"))

# Detect faces — returns list of dicts
faces = fx.detect(image)
for face in faces:
    print(f"bbox: {face['bbox']}, score: {face['score']:.3f}")
    print(f"keypoints: {face['keypoints']}")
    # face['embedding'] is a 512-dim numpy float32 vector

# Compare two faces
if len(faces) >= 2:
    sim = FaceX.similarity(faces[0]["embedding"], faces[1]["embedding"])
    print(f"Similarity: {sim:.3f}  ({'same' if sim > 0.3 else 'different'} person)")
```

## Rectangular YuNet input

For camera frames and photos, initialize the native SDK with the rectangle policy.
Install the image extra for Pillow and build the shared library from this version.

```python
fx = FaceX(embed_weights="weights/edgeface_xs_fp32.bin",
           detect_weights="weights/yunet_fp32.bin",
           input_policy="rectangle_bgr")
faces = fx.detect(image)
```

Pass RGB images to the SDK. The detector receives BGR internally and uses up to
409600 input pixels with a calibrated score threshold of 0.70. The previous
RGB rectangle policy remains available as `input_policy="rectangle"` at 0.58.
For a 1920 × 1080 source it resizes to 832 × 468 and centers that image on an
832 × 480 canvas. Boxes and five keypoints use original image coordinates.
Alignment and embedding sample the original RGB pixels. Set `score_threshold`
explicitly to override the calibrated threshold. Existing SDK callers use native
image dimensions by default.

For small faces use `input_policy="rectangle_bgr_800"`. This raises the pixel
budget to 800² and uses a calibrated threshold of 0.75. The REST equivalent is
`FACEX_DETECT_SIZE=800`; the default is 640.

For small faces in the 640 BGR mode, use `min_box_side=8` with
`input_policy="rectangle_bgr"`, or `FACEX_MIN_BOX_SIDE=8` in REST. This admits
predicted boxes with both sides at least eight detector pixels and selects a
score threshold of 0.75. The intermediate setting 12 selects 0.74, and the
default 16 selects 0.70. The 800 BGR mode uses 0.75 for all three sizes.
See [the minimum side comparison](../docs/research/yunet_minimum_side.md).

The Docker REST service uses the BGR rectangle policy for `POST /detect` and returns
up to 32 faces. Mount both weights at the paths above under `/app`, or set
`EMBED_WEIGHTS` and `DETECT_WEIGHTS` to their paths inside the container.

```sh
curl -F file=@photo.jpg http://localhost:8080/detect
```

## Configuration

### Auto-detection

FaceX looks for the native library and weights relative to the package location
(expects `../../weights/` and `../../libfacex.so` or `../../facex.dll`).

### Environment variables

| Variable | Description |
|---|---|
| `FACEX_LIB` | Path to shared library (`.dll` / `.so` / `.dylib`) |
| `FACEX_CLI` | Path to `facex-cli` binary (subprocess fallback) |
| `FACEX_ROOT` | Root of FaceX project (alternative search path) |
| `FACEX_EMBED_WEIGHTS` | Path to embedding model weights |
| `FACEX_DETECT_WEIGHTS` | Path to detector weights |

### Explicit paths

```python
fx = FaceX(
    lib_path="/path/to/libfacex.so",
    embed_weights="/path/to/embed.bin",
    detect_weights="/path/to/detect.bin",
    score_threshold=0.5,
    nms_threshold=0.4,
    max_faces=20,
)
```

## API reference

### `FaceX()`

Main class. Uses ctypes (shared library) if available, falls back to subprocess (CLI).

- **`detect(image, max_faces=None)`** -- Detect faces in an RGB uint8 image. Returns list of dicts with `bbox`, `score`, `keypoints`, `embedding`.
- **`embed(face_112x112)`** -- Compute embedding for a pre-aligned 112x112 face (ctypes only). Input: float32 HWC in [-1, 1]. Returns 512-dim float32 array.
- **`similarity(emb1, emb2)`** -- Static method. Cosine similarity between two 512-dim embeddings. Returns float in [-1, 1].
- **`close()`** -- Free native resources. Also works as context manager (`with FaceX() as fx: ...`).
- **`backend`** -- Property: `"ctypes"`, `"cli"`, or `"none"`.

### `similarity(emb1, emb2)`

Module-level function, same as `FaceX.similarity()`. Works without initializing FaceX.

```python
from facex import similarity
sim = similarity(emb1, emb2)
```

## Backends

| Backend | Requires | Detection | Embedding (standalone) |
|---|---|---|---|
| ctypes | `libfacex.so` / `facex.dll` | Yes | Yes |
| cli | `facex-cli` binary | Yes | No (use detect) |
| none | nothing | No | No |

Similarity always works (pure numpy).

### Geometry only and guided refinement

Use `fx.detect(image, embeddings=False)` or `POST /detect?embeddings=false` to
get original-coordinate boxes, scores and five landmarks. The embedding field
is omitted. The default response includes embeddings.

For a crowded still image, initialize the native SDK with
`input_policy="rectangle_bgr", min_box_side=8` and call:

```python
faces = fx.detect_guided(image, extra_cpu_ms=30, embeddings=False, max_faces=32)
```

The first pass selects a small-face neighborhood. The SDK runs at most one
448-budget crop, estimates its CPU cost from the first pass and skips it when
the estimate exceeds the extra budget. This is an admission estimate, not a
hard deadline. A zero budget selects the one-pass result. The guided operating
point uses confidence 0.78, and crops use original-frame coordinates. If
requested, embeddings are computed only for the retained faces on the source
RGB image.

### Video and concurrent REST requests

For a camera with upstream tracking, call detection with `embeddings=False`,
associate the resulting geometry with tracker IDs and pass it to a per-camera
cache. This optional SDK policy refreshes on first sight, loss, overlap, a
large change in appearance or position, improved crop quality, low confidence,
small boxes and at most every 12 frames. Supply every frame, including empty
frames, and call `reset()` when seeking or reconnecting. Mark uncertain
associations with `ambiguous=True`. ID switches that the tracker does not
report can still carry an outdated embedding until the next refresh.

```python
from facex.video import VideoEmbeddings

cache = VideoEmbeddings(fx, max_age=12)
# Each entry contains track_id, bbox, score and the five source keypoints.
results = cache.update(rgb, frame_number, tracked_faces)
```

On Linux with Gunicorn, start synchronous workers from the preloaded entry
point to share packed model pages by fork. The model files must be present
when the master starts. Each process allocates its own inference buffers;
keep one active inference call per process. To replace weights restart the
master. Example inside the REST container:

```sh
gunicorn --preload --worker-class sync --workers 4 --bind 0.0.0.0:8080 preloaded:app
```

### Exact gallery search and enrollment

Use a row per identity and keep the row to identity mapping with the matrix.
The gallery owns a normalized float64 copy. Rebuild it when enrollment changes.

```python
from facex import Gallery, enrollment_template

template, agreement = enrollment_template(views, weights=qualities)
gallery = Gallery(enrollment_rows)
indices, scores = gallery.search(query_rows, k=5)
decisions = gallery.identify(query_rows, min_score=calibrated_score,
                            min_margin=calibrated_margin)
```

A decision of -1 means abstention. A margin requires two enrollment rows.
Equal best scores always abstain. The optional medoid mode selects an actual
enrolled view. Agreement ranges from zero to one and measures directional
consistency. Calibrate decision thresholds on separate data for each checkpoint.

The REST compare endpoint accepts a numeric array or the little-endian
float32 Base64 string returned by the compact detect endpoint for either input.
