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

import os
import subprocess
import threading
import time

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from facex import FaceX

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
            stderr=subprocess.PIPE,
        )

    def embed(self, face_112x112_hwc):
        """Compute 512-dim embedding from 112x112 float32 HWC face."""
        data = face_112x112_hwc.astype(np.float32).tobytes()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
        out = self.proc.stdout.read(512 * 4)
        return np.frombuffer(out, dtype=np.float32).copy()

    def close(self):
        self.proc.terminate()


# Global process (per worker)
_fx = None
_detector = None
_detector_lock = threading.Lock()


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


def get_fx():
    global _fx
    if _fx is None:
        _fx = FaceXProcess()
    return _fx


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

    f = request.files['file']
    img = Image.open(f.stream).convert('RGB').resize((112, 112))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # normalize to [-1, 1]

    t0 = time.time()
    fx = get_fx()
    emb = fx.embed(arr)
    ms = (time.time() - t0) * 1000

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
    data = request.get_json()
    if not data or 'emb1' not in data or 'emb2' not in data:
        return jsonify({'error': 'Need emb1 and emb2'}), 400

    emb1 = np.array(data['emb1'], dtype=np.float32)
    emb2 = np.array(data['emb2'], dtype=np.float32)

    sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

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

    try:
        with Image.open(request.files['file'].stream) as uploaded:
            rgb = np.asarray(uploaded.convert('RGB'))
    except (OSError, ValueError, Image.DecompressionBombError):
        return jsonify({'error': 'Invalid image'}), 400

    start = time.perf_counter()
    with _detector_lock:
        try:
            initialize_detector()
            faces = _detector.detect(rgb, max_faces=32, embeddings=embeddings)
        except (RuntimeError, OSError):
            app.logger.exception("Face detection failed")
            return jsonify({'error': 'Face detection unavailable'}), 503
    if embeddings:
        for face in faces:
            face['embedding'] = face['embedding'].tolist()
    return jsonify({'faces': faces, 'ms': round((time.perf_counter() - start) * 1000, 2)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
