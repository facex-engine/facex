"""
FaceX REST API server.

Endpoints:
  POST /detect   — detect faces in image, returns bbox + kps + embeddings
  POST /embed    — embed pre-aligned 112x112 face
  POST /compare  — compare two embeddings (cosine similarity)
  GET  /health   — health check

Usage:
  gunicorn -w 2 -b 0.0.0.0:8080 server:app
"""

from contextlib import contextmanager, nullcontext
import base64
import binascii
import os
import subprocess
import threading
import time

import numpy as np
from flask import Flask, g, request, jsonify
from PIL import Image
from facex import FaceX, similarity

app = Flask(__name__)

# FaceX subprocess pool
FACEX_CLI = os.environ.get('FACEX_CLI', './facex-server')
EMBED_WEIGHTS = os.environ.get('EMBED_WEIGHTS', 'weights/edgeface_xs_fp32.bin')
DETECT_WEIGHTS = os.environ.get('DETECT_WEIGHTS', 'weights/yunet_fp32.bin')


DETECT_SIZE = os.environ.get("FACEX_DETECT_SIZE", "640")
if DETECT_SIZE not in ("640", "800"):
    raise ValueError("FACEX_DETECT_SIZE must be 640 or 800")
MIN_BOX_SIDE = int(os.environ.get("FACEX_MIN_BOX_SIDE", "16"))
if MIN_BOX_SIDE not in (8, 12, 16):
    raise ValueError("FACEX_MIN_BOX_SIDE must be 8, 12 or 16")
DETECT_POLICY = "rectangle_bgr_800" if DETECT_SIZE == "800" else "rectangle_bgr"


class FaceXProcess:
    """Persistent subprocess for face embedding."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [FACEX_CLI, EMBED_WEIGHTS, '--server'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def embed(self, face_112x112_hwc):
        """Compute 512-dim embedding from 112x112 float32 HWC face."""
        data = face_112x112_hwc.astype(np.float32).tobytes()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
        out = self.proc.stdout.read(512 * 4)
        if len(out) != 512 * 4:
            raise OSError("Incomplete embedding from worker")
        return np.frombuffer(out, dtype=np.float32).copy()

    def close(self):
        self.proc.terminate()
        self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()


# Global process (per worker)
_fx = None
_embed_lock = threading.Lock()
_detector = None
_detector_lock = threading.Lock()
# One image can be prepared while the preceding image is inside each model.
_detect_capacity = threading.BoundedSemaphore(2)
_embed_capacity = threading.BoundedSemaphore(2)


QUEUE_WAIT_MS = int(os.environ.get("FACEX_QUEUE_WAIT_MS", "250"))
if not 0 <= QUEUE_WAIT_MS <= 86_400_000:
    raise ValueError("FACEX_QUEUE_WAIT_MS must be between 0 and 86400000")


class InferenceBusy(Exception):
    pass


@app.errorhandler(InferenceBusy)
def inference_busy(_error):
    return jsonify(error="Inference deadline expired", code="inference_busy"), 503


@app.before_request
def set_inference_deadline():
    if request.endpoint not in ("detect", "embed") or request.method != "POST":
        return
    start = time.monotonic_ns()
    g.inference_deadline = start + QUEUE_WAIT_MS * 1_000_000 if QUEUE_WAIT_MS else None
    value = request.headers.get("X-FaceX-Deadline-Ms")
    if value is not None:
        if not (1 <= len(value) <= 15 and value.isascii() and value.isdigit()):
            return jsonify(error="X-FaceX-Deadline-Ms must be a Unix timestamp in milliseconds"), 400
        remaining = int(value) * 1_000_000 - time.time_ns()
        if remaining <= 0:
            raise InferenceBusy()
        deadline = start + remaining
        if g.inference_deadline is None or deadline < g.inference_deadline:
            g.inference_deadline = deadline


def check_inference_deadline():
    if g.inference_deadline is not None and time.monotonic_ns() >= g.inference_deadline:
        raise InferenceBusy()


@contextmanager
def inference_slot(lock):
    deadline = g.inference_deadline
    if deadline is None:
        lock.acquire()
    else:
        remaining = deadline - time.monotonic_ns()
        if remaining <= 0 or not lock.acquire(timeout=min(remaining / 1e9, threading.TIMEOUT_MAX)):
            raise InferenceBusy()
    try:
        check_inference_deadline()
        yield
    finally:
        lock.release()


def preparation_slot(capacity):
    if g.inference_deadline is None:
        return nullcontext()
    return inference_slot(capacity)


def initialize_detector():
    """Load weights before forking or under the detector lock in a worker."""
    global _detector
    if _detector is None:
        engine = FaceX(embed_weights=EMBED_WEIGHTS, detect_weights=DETECT_WEIGHTS,
                       input_policy=DETECT_POLICY, min_box_side=MIN_BOX_SIDE,
                       license_key=os.environ.get("LICENSE_KEY"))
        if engine.backend != "ctypes":
            raise RuntimeError("Missing native FaceX library")
        _detector = engine


def embed_in_worker(arr):
    global _fx
    # The binary protocol has no request IDs. Keep each write and read together.
    with inference_slot(_embed_lock):
        if _fx is None:
            _fx = FaceXProcess()
        check_inference_deadline()
        try:
            return _fx.embed(arr)
        except OSError:
            failed, _fx = _fx, None
            failed.close()
            raise


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '2.0.0'})


@app.route('/embed', methods=['POST'])
def embed():
    """Embed a pre-aligned 112x112 face image.

    Input: multipart file upload (JPEG/PNG, 112x112)
    Output: {"embedding": [512 floats], "ms": float}
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    with preparation_slot(_embed_capacity):
        f = request.files['file']
        try:
            with Image.open(f.stream) as uploaded:
                img = uploaded.convert('RGB').resize((112, 112))
                arr = np.asarray(img, dtype=np.float32)
            arr /= 127.5
            arr -= 1
        except (OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error='Invalid image'), 400

        t0 = time.perf_counter()
        try:
            emb = embed_in_worker(arr)
        except OSError:
            app.logger.exception("Embedding worker failed")
            return jsonify({"error": "Face embedding unavailable"}), 503
        ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        'embedding': emb.tolist(),
        'ms': round(ms, 2),
    })


@app.route('/compare', methods=['POST'])
def compare():
    """Compare two embeddings.

    Input: JSON {"emb1": [512], "emb2": [512]}
    Output: {"similarity": float, "match": bool}
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'emb1' not in data or 'emb2' not in data:
        return jsonify({'error': 'Need emb1 and emb2'}), 400

    def decode(value):
        if isinstance(value, str):
            if len(value) != 2732:
                raise ValueError("Expected 512 base64 encoded float32 components")
            raw = base64.b64decode(value, validate=True)
            if len(raw) != 2048:
                raise ValueError("Expected 2048 embedding bytes")
            return np.frombuffer(raw, dtype="<f4")
        if not isinstance(value, list) or len(value) != 512 or any(
                type(v) not in (int, float) for v in value):
            raise ValueError("Expected an array of 512 real components")
        return value

    try:
        sim = similarity(decode(data["emb1"]), decode(data["emb2"]))
    except (ValueError, OverflowError, binascii.Error):
        return jsonify(error="Invalid embedding"), 400

    return jsonify({
        'similarity': round(sim, 6),
        'match': sim > 0.3,
    })


@app.route('/detect', methods=['POST'])
def detect():
    """Detect faces in image, return bbox + embeddings.

    Input: multipart file upload (any size JPEG/PNG)
    Output: {"faces": [{"bbox": [x1,y1,x2,y2], "keypoints": [[x,y], ...], "score": float, "embedding": [512]}], "ms": float}
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    flag = request.args.get("embeddings", "true").lower()
    if flag not in ("true", "false"):
        return jsonify({"error": "embeddings must be true or false"}), 400
    embeddings = flag == "true"
    encoding = request.args.get("embedding_format", "array")
    if encoding not in ("array", "base64_f32le"):
        return jsonify({"error": "embedding_format must be array or base64_f32le"}), 400

    with preparation_slot(_detect_capacity):
        try:
            with Image.open(request.files['file'].stream) as uploaded:
                rgb = np.asarray(uploaded if uploaded.mode == 'RGB' else uploaded.convert('RGB'))
        except (OSError, ValueError, Image.DecompressionBombError):
            return jsonify({'error': 'Invalid image'}), 400

        start = time.perf_counter()
        with inference_slot(_detector_lock):
            try:
                initialize_detector()
                check_inference_deadline()
                faces = _detector.detect(rgb, max_faces=32, embeddings=embeddings)
            except (RuntimeError, OSError):
                app.logger.exception("Face detection failed")
                return jsonify({'error': 'Face detection unavailable'}), 503
    if embeddings:
        for face in faces:
            vector = face["embedding"]
            if encoding == "base64_f32le":
                face["embedding"] = base64.b64encode(vector.astype("<f4", copy=False)).decode("ascii")
            else:
                face["embedding"] = vector.tolist()
    return jsonify({'faces': faces, 'ms': round((time.perf_counter() - start) * 1000, 2)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
