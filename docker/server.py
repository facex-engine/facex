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

import json
import os
import shlex
import struct
import sys
import time

_subprocess = __import__('subprocess')

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# FaceX subprocess pool
FACEX_CLI = os.environ.get('FACEX_CLI', './facex-server')
EMBED_WEIGHTS = os.environ.get('EMBED_WEIGHTS', 'weights/edgeface_xs_fp32.bin')
DETECT_WEIGHTS = os.environ.get('DETECT_WEIGHTS', 'weights/yunet_fp32.bin')

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_MAGIC = [
    b'\xff\xd8\xff',        # JPEG
    b'\x89PNG\r\n\x1a\n',  # PNG
]


def _validate_image_upload(stream):
    """Check magic bytes and file size. Returns error string or None."""
    header = stream.read(8)
    stream.seek(0)
    if not any(header.startswith(m) for m in ALLOWED_MAGIC):
        return 'Invalid file type: only JPEG and PNG are accepted'
    stream.seek(0, 2)
    if stream.tell() > MAX_UPLOAD_BYTES:
        return 'File too large (max 10 MB)'
    stream.seek(0)
    return None


class FaceXProcess:
    """Persistent subprocess for face embedding."""

    def __init__(self):
        self.proc = _subprocess.Popen(
            [shlex.quote(FACEX_CLI), shlex.quote(EMBED_WEIGHTS), '--server'],
            stdin=_subprocess.PIPE,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
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
    err = _validate_image_upload(f.stream)
    if err:
        return jsonify({'error': err}), 400
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
    Output: {"faces": [{"bbox": [x1,y1,x2,y2], "score": float, "embedding": [512]}], "ms": float}
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    err = _validate_image_upload(f.stream)
    if err:
        return jsonify({'error': err}), 400
    img = Image.open(f.stream).convert('RGB')

    t0 = time.time()

    # Resize to 160x160 for detection (letterbox)
    w, h = img.size
    scale = min(160 / w, 160 / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    padded = Image.new('RGB', (160, 160), (0, 0, 0))
    dx, dy = (160 - nw) // 2, (160 - nh) // 2
    padded.paste(img_resized, (dx, dy))

    # TODO: run C detector via subprocess or ctypes
    # For now, return placeholder
    ms = (time.time() - t0) * 1000

    return jsonify({
        'faces': [],
        'ms': round(ms, 2),
        'note': 'Detection via REST API coming soon. Use /embed for pre-aligned faces.',
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)
