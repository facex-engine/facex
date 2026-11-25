"""
FaceX core module — ctypes binding with subprocess fallback.
"""

import ctypes
import ctypes.util
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 512
MAX_FACES_DEFAULT = 20
_POLICY_THRESHOLDS = {"native": 0.5, "rectangle": 0.58, "rectangle_bgr": 0.70, "rectangle_bgr_800": 0.75}

# FaceXResult struct layout:
#   float x1, y1, x2, y2, score   (5 floats = 20 bytes)
#   float kps[10]                  (10 floats = 40 bytes)
#   float embedding[512]           (512 floats = 2048 bytes)
# Total: 527 floats = 2108 bytes
_RESULT_FLOATS = 5 + 10 + EMBEDDING_DIM  # 527
_RESULT_BYTES = _RESULT_FLOATS * 4  # 2108


class FaceXResult(ctypes.Structure):
    """Mirrors the C FaceXResult struct."""
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("kps", ctypes.c_float * 10),
        ("embedding", ctypes.c_float * EMBEDDING_DIM),
    ]


def _result_to_dict(r: FaceXResult, embeddings=True) -> dict:
    """Convert a FaceXResult struct to a Python dict."""
    kps_raw = list(r.kps)
    keypoints = [(kps_raw[i], kps_raw[i + 1]) for i in range(0, 10, 2)]
    face = {
        "bbox": (r.x1, r.y1, r.x2, r.y2),
        "score": r.score,
        "keypoints": keypoints,
    }
    if embeddings:
        face["embedding"] = np.ctypeslib.as_array(r.embedding).copy()
    return face


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------
def _find_lib() -> Optional[str]:
    """Try to locate the FaceX shared library."""
    # Check env var first
    env = os.environ.get("FACEX_LIB")
    if env and os.path.isfile(env):
        return env

    system = platform.system()
    if system == "Windows":
        names = ["facex.dll", "libfacex.dll"]
    elif system == "Darwin":
        names = ["libfacex.dylib"]
    else:
        names = ["libfacex.so"]

    # Search relative to this file, then facex root, then PATH
    search_dirs = []

    # facex project root (two levels up from python/facex/)
    pkg_dir = Path(__file__).resolve().parent
    facex_root = pkg_dir.parent.parent
    search_dirs.append(facex_root)
    search_dirs.append(facex_root / "lib")
    search_dirs.append(facex_root / "build")
    search_dirs.append(facex_root / "build" / "Release")

    # Also check FACEX_ROOT env
    root_env = os.environ.get("FACEX_ROOT")
    if root_env:
        p = Path(root_env)
        search_dirs.extend([p, p / "lib", p / "build"])

    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)

    # Last resort: ctypes.util
    found = ctypes.util.find_library("facex")
    return found


def _find_cli() -> Optional[str]:
    """Try to locate the facex-cli executable."""
    env = os.environ.get("FACEX_CLI")
    if env and os.path.isfile(env):
        return env

    system = platform.system()
    ext = ".exe" if system == "Windows" else ""
    names = [f"facex-cli{ext}", f"facex-example{ext}"]

    pkg_dir = Path(__file__).resolve().parent
    facex_root = pkg_dir.parent.parent
    search_dirs = [facex_root, facex_root / "build", facex_root / "build" / "Release"]

    root_env = os.environ.get("FACEX_ROOT")
    if root_env:
        p = Path(root_env)
        search_dirs.extend([p, p / "build"])

    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
    return None


def _find_weights() -> Tuple[Optional[str], Optional[str]]:
    """Locate default weight files (embed, detect)."""
    env_embed = os.environ.get("FACEX_EMBED_WEIGHTS")
    env_detect = os.environ.get("FACEX_DETECT_WEIGHTS")
    if env_embed and env_detect:
        return env_embed, env_detect

    pkg_dir = Path(__file__).resolve().parent
    facex_root = pkg_dir.parent.parent
    weights_dir = facex_root / "weights"

    embed_names = ["embed.bin", "edgeface_fp32.bin", "edgeface.bin"]
    detect_names = ["detect.bin", "yunet_fp32.bin", "yunet.bin"]

    embed_path = env_embed
    detect_path = env_detect

    if not embed_path:
        for name in embed_names:
            p = weights_dir / name
            if p.is_file():
                embed_path = str(p)
                break

    if not detect_path:
        for name in detect_names:
            p = weights_dir / name
            if p.is_file():
                detect_path = str(p)
                break

    return embed_path, detect_path


# ---------------------------------------------------------------------------
# Pure-Python similarity (always available)
# ---------------------------------------------------------------------------
def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Cosine similarity between two 512-dim embeddings.

    Args:
        emb1: numpy float32 array of shape (512,)
        emb2: numpy float32 array of shape (512,)

    Returns:
        Cosine similarity in [-1, 1]. Values > 0.3 indicate same person.
    """
    emb1 = np.asarray(emb1, dtype=np.float32).ravel()
    emb2 = np.asarray(emb2, dtype=np.float32).ravel()
    if emb1.shape[0] != EMBEDDING_DIM or emb2.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Embeddings must be {EMBEDDING_DIM}-dim, "
                         f"got {emb1.shape[0]} and {emb2.shape[0]}")
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return float(dot / (norm1 * norm2))


# ---------------------------------------------------------------------------
# FaceX class
# ---------------------------------------------------------------------------
class FaceX:
    """
    Python interface to the FaceX face detection + recognition engine.

    Tries ctypes (shared library) first, falls back to subprocess (CLI).
    Pure-Python similarity is always available even without native code.

    Args:
        lib_path:        Path to shared library (.dll/.so/.dylib). Auto-detected if None.
        cli_path:        Path to facex-cli binary. Auto-detected if None.
        embed_weights:   Path to embedding weights. Auto-detected if None.
        detect_weights:  Path to detector weights. Auto-detected if None.
        license_key:     License key for encrypted weights (None for plain).
        score_threshold: Override the score threshold calibrated for the selected input policy.
        input_policy: Use "native", "rectangle" (RGB), or a YuNet BGR policy ("rectangle_bgr" or "rectangle_bgr_800").
        min_box_side: Minimum detector box side in pixels (8, 12 or 16).
        nms_threshold:   NMS IoU threshold (default 0.4).
        max_faces:       Maximum faces to detect per image (default 20).
    """

    def __init__(
        self,
        lib_path: Optional[str] = None,
        cli_path: Optional[str] = None,
        embed_weights: Optional[str] = None,
        detect_weights: Optional[str] = None,
        license_key: Optional[str] = None,
        score_threshold: Optional[float] = None,
        nms_threshold: float = 0.4,
        max_faces: int = MAX_FACES_DEFAULT,
        input_policy: str = "native",
        min_box_side: int = 16,
    ):
        self._lib = None
        self._handle = None
        self._cli_path = cli_path
        self._max_faces = max_faces
        if input_policy not in _POLICY_THRESHOLDS:
            raise ValueError("Unknown detector input policy")
        if min_box_side not in (8, 12, 16):
            raise ValueError("min_box_side must be 8, 12 or 16")
        self._min_box_side = min_box_side
        self._input_policy = input_policy
        if score_threshold is None:
            if input_policy == "rectangle_bgr":
                score_threshold = {8: 0.75, 12: 0.74, 16: 0.70}[min_box_side]
            else:
                score_threshold = _POLICY_THRESHOLDS[input_policy]
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold

        # Resolve weights
        auto_embed, auto_detect = _find_weights()
        self._embed_weights = embed_weights or auto_embed
        self._detect_weights = detect_weights or auto_detect

        # Try ctypes first
        lib_file = lib_path or _find_lib()
        if lib_file:
            try:
                self._init_ctypes(lib_file, license_key)
                return
            except OSError as e:
                print(f"[facex] ctypes load failed ({e}), trying subprocess fallback",
                      file=sys.stderr)

        # Fallback: locate CLI
        if not self._cli_path:
            self._cli_path = _find_cli()

        if not self._lib and not self._cli_path:
            print("[facex] WARNING: No shared library or CLI found. "
                  "Only similarity() will work. Set FACEX_LIB or FACEX_CLI env var.",
                  file=sys.stderr)

    def _init_ctypes(self, lib_path: str, license_key: Optional[str]):
        """Load the shared library and initialize the engine."""
        self._lib = ctypes.CDLL(lib_path)

        # facex_init
        self._lib.facex_init.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self._lib.facex_init.restype = ctypes.c_void_p

        # facex_detect
        self._lib.facex_detect.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(FaceXResult),
            ctypes.c_int,
        ]
        self._lib.facex_detect.restype = ctypes.c_int

        # facex_embed
        self._lib.facex_embed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.facex_embed.restype = ctypes.c_int

        # facex_similarity
        self._lib.facex_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.facex_similarity.restype = ctypes.c_float

        # facex_free
        self._lib.facex_free.argtypes = [ctypes.c_void_p]
        self._lib.facex_free.restype = None

        # facex_set_score_threshold
        self._lib.facex_set_score_threshold.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        self._lib.facex_set_score_threshold.restype = None

        # facex_set_nms_threshold
        self._lib.facex_set_nms_threshold.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        self._lib.facex_set_nms_threshold.restype = None

        if self._input_policy != "native":
            if not hasattr(self._lib, "facex_detect_prepared"):
                raise RuntimeError("Rectangle input requires a library with facex_detect_prepared")
            image_ptr = ctypes.POINTER(ctypes.c_uint8)
            self._lib.facex_detect_prepared.argtypes = [
                ctypes.c_void_p, image_ptr, ctypes.c_int, ctypes.c_int,
                image_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(FaceXResult), ctypes.c_int,
            ]
            self._lib.facex_detect_prepared.restype = ctypes.c_int

        if self._min_box_side != 16:
            if not hasattr(self._lib, "facex_set_min_box_side"):
                raise RuntimeError("Custom min_box_side requires an updated FaceX library")
            self._lib.facex_set_min_box_side.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._lib.facex_set_min_box_side.restype = None

        if hasattr(self._lib, "facex_detect_prepared_ex"):
            pixel = ctypes.POINTER(ctypes.c_uint8)
            scalar = ctypes.POINTER(ctypes.c_float)
            self._lib.facex_detect_prepared_ex.argtypes = [
                ctypes.c_void_p, pixel, ctypes.c_int, ctypes.c_int, pixel,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(FaceXResult), ctypes.c_int, ctypes.c_int, ctypes.c_float]
            self._lib.facex_embed_landmarks.argtypes = [
                ctypes.c_void_p, pixel, ctypes.c_int, ctypes.c_int, scalar, scalar]

        # Init engine
        embed_b = self._embed_weights.encode() if self._embed_weights else None
        detect_b = self._detect_weights.encode() if self._detect_weights else None
        key_b = license_key.encode() if license_key else None

        self._handle = self._lib.facex_init(embed_b, detect_b, key_b)
        if not self._handle:
            self._lib = None
            raise RuntimeError("facex_init returned NULL — check weight paths")

        if self._min_box_side != 16:
            self._lib.facex_set_min_box_side(self._handle, self._min_box_side)
        self._lib.facex_set_score_threshold(self._handle, self._score_threshold)
        self._lib.facex_set_nms_threshold(self._handle, self._nms_threshold)

    @property
    def backend(self) -> str:
        """Return which backend is active: 'ctypes', 'cli', or 'none'."""
        if self._lib and self._handle:
            return "ctypes"
        if self._cli_path:
            return "cli"
        return "none"

    def detect(
        self,
        image: np.ndarray,
        max_faces: Optional[int] = None,
        *, embeddings: bool = True,
    ) -> List[dict]:
        """
        Detect faces and compute embeddings.

        Args:
            image: RGB uint8 numpy array, shape (H, W, 3).
            max_faces: Override max faces for this call.
            embeddings: Pass False to omit alignment, embedding inference and vectors.

        Returns:
            List of dicts, each with keys:
                bbox: (x1, y1, x2, y2) in pixel coords
                score: detection confidence [0, 1]
                keypoints: list of 5 (x, y) tuples
                embedding: numpy float32 array of shape (512,)
        """
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB uint8 image, got shape {image.shape}")

        h, w = image.shape[:2]
        mf = max_faces or self._max_faces

        if self._input_policy != "native" and self.backend != "ctypes":
            raise RuntimeError("Rectangle input requires the native FaceX shared library")
        if self._lib and self._handle:
            return self._detect_ctypes(image, w, h, mf, embeddings)
        elif self._cli_path:
            if not embeddings or self._min_box_side != 16:
                raise RuntimeError("This detection option requires the native shared library")
            return self._detect_cli(image, w, h, mf)
        else:
            raise RuntimeError(
                "No backend available. Install the shared library or set FACEX_CLI."
            )

    def _detect_ctypes(self, image, w, h, max_faces, embeddings=True, side=None, threshold=-1):
        results = (FaceXResult * max_faces)()
        pixels = ctypes.POINTER(ctypes.c_uint8)
        ptr = image.ctypes.data_as(pixels)
        canvas, rw, rh = image, w, h
        if self._input_policy != "native":
            from .yunet import prepare_rectangle
            side = side or (800 if self._input_policy == "rectangle_bgr_800" else 640)
            canvas, rw, rh = prepare_rectangle(image, bgr=self._input_policy != "rectangle", side=side)
        ch, cw = canvas.shape[:2]
        args = (self._handle, ptr, w, h, canvas.ctypes.data_as(pixels), cw, ch,
                rw, rh, results, max_faces)
        if hasattr(self._lib, "facex_detect_prepared_ex"):
            n = self._lib.facex_detect_prepared_ex(*args, int(embeddings), threshold)
        elif not embeddings or threshold >= 0:
            raise RuntimeError("This option requires an updated native library")
        elif self._input_policy == "native":
            n = self._lib.facex_detect(self._handle, ptr, w, h, results, max_faces)
        else:
            n = self._lib.facex_detect_prepared(*args)
        if n < 0:
            raise RuntimeError(f"facex_detect returned error code {n}")
        return [_result_to_dict(results[i], embeddings) for i in range(n)]

    def detect_guided(self, image, *, extra_cpu_ms, embeddings=True, max_faces=None):
        """Refine at most one crop when its estimated CPU cost fits the extra budget."""
        from .yunet import guided_crop, merge_faces
        if self._input_policy != "rectangle_bgr" or self._min_box_side != 8 or self.backend != "ctypes":
            raise ValueError("Guided detection requires native rectangle_bgr with min_box_side=8")
        if not np.isfinite(extra_cpu_ms) or extra_cpu_ms < 0:
            raise ValueError("extra_cpu_ms must be finite and nonnegative")
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
            raise ValueError("Expected a nonempty RGB image")
        h, w = image.shape[:2]
        start = time.thread_time_ns()
        faces = self._detect_ctypes(image, w, h, 32, False, 640, .35)
        estimate = (time.thread_time_ns() - start) / 1e6 * (448/640)**2
        crop = guided_crop(faces, w, h) if estimate <= extra_cpu_ms else None
        if crop:
            x, y, cw, ch = crop
            patch = np.ascontiguousarray(image[y:y+ch, x:x+cw])
            extra = self._detect_ctypes(patch, cw, ch, 32, False, 448, .35)
            faces = merge_faces(faces, extra, x, y)
        threshold = .78 if estimate <= extra_cpu_ms else self._score_threshold
        limit = self._max_faces if max_faces is None else max_faces
        if limit < 0:
            raise ValueError("max_faces must be nonnegative")
        faces = [f for f in faces if f["score"] >= threshold][:min(32, limit)]
        if embeddings:
            for face in faces:
                face["embedding"] = self.embed_landmarks(image, face["keypoints"])
        return faces

    def embed_landmarks(self, image, keypoints):
        """Align five source landmarks and embed the original RGB crop."""
        if self.backend != "ctypes" or not hasattr(self._lib, "facex_embed_landmarks"):
            raise RuntimeError("Landmark embedding requires an updated native library")
        image = np.ascontiguousarray(image, dtype=np.uint8)
        points = np.ascontiguousarray(keypoints, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError("Expected an RGB image and five finite landmark pairs")
        h, w = image.shape[:2]
        scalar = ctypes.POINTER(ctypes.c_float)
        vec = np.empty(512, dtype=np.float32)
        rc = self._lib.facex_embed_landmarks(
            self._handle, image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), w, h,
            points.ctypes.data_as(scalar), vec.ctypes.data_as(scalar))
        if rc < 0:
            raise RuntimeError("Landmark embedding failed")
        return vec

    def _detect_cli(self, image: np.ndarray, w: int, h: int, max_faces: int) -> List[dict]:
        """Subprocess fallback: pipe image data to CLI and parse output."""
        # Build command — the CLI protocol:
        # stdin: 4 bytes width (le) + 4 bytes height (le) + w*h*3 bytes RGB uint8
        # stdout: 4 bytes num_faces (le) + num_faces * FaceXResult as raw floats
        args = [self._cli_path, "--detect", "--pipe"]
        if self._embed_weights:
            args.extend(["--embed-weights", self._embed_weights])
        if self._detect_weights:
            args.extend(["--detect-weights", self._detect_weights])
        args.extend(["--max-faces", str(max_faces)])
        args.extend(["--score-threshold", str(self._score_threshold)])

        header = struct.pack("<ii", w, h)
        input_data = header + image.tobytes()

        try:
            proc = subprocess.run(
                args,
                input=input_data,
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(f"CLI not found: {self._cli_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("facex-cli timed out")

        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"facex-cli failed (rc={proc.returncode}): {stderr}")

        out = proc.stdout
        if len(out) < 4:
            raise RuntimeError("facex-cli produced no output")

        n = struct.unpack("<i", out[:4])[0]
        expected = 4 + n * _RESULT_BYTES
        if len(out) < expected:
            raise RuntimeError(
                f"Incomplete output: expected {expected} bytes, got {len(out)}"
            )

        faces = []
        offset = 4
        for i in range(n):
            floats = struct.unpack_from(f"<{_RESULT_FLOATS}f", out, offset)
            offset += _RESULT_BYTES
            x1, y1, x2, y2, score = floats[:5]
            kps_raw = floats[5:15]
            emb = np.array(floats[15:], dtype=np.float32)
            keypoints = [(kps_raw[j], kps_raw[j + 1]) for j in range(0, 10, 2)]
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "keypoints": keypoints,
                "embedding": emb,
            })
        return faces

    def embed(self, face_112x112: np.ndarray) -> np.ndarray:
        """
        Compute embedding for a pre-aligned 112x112 face crop.

        Args:
            face_112x112: float32 numpy array shape (112, 112, 3), values in [-1, 1].

        Returns:
            numpy float32 array of shape (512,), L2-normalized.
        """
        face = np.ascontiguousarray(face_112x112, dtype=np.float32)
        if face.shape != (112, 112, 3):
            raise ValueError(f"Expected (112, 112, 3), got {face.shape}")

        if self._lib and self._handle:
            return self._embed_ctypes(face)
        else:
            raise RuntimeError(
                "embed() requires the shared library (ctypes backend). "
                "Use detect() which includes embedding, or install libfacex."
            )

    def _embed_ctypes(self, face: np.ndarray) -> np.ndarray:
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        ptr_in = face.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_out = embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._lib.facex_embed(self._handle, ptr_in, ptr_out)
        if rc != 0:
            raise RuntimeError(f"facex_embed returned error code {rc}")
        return embedding

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two 512-dim embeddings."""
        return similarity(emb1, emb2)

    def close(self):
        """Release native resources."""
        if self._lib and self._handle:
            self._lib.facex_free(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"FaceX(backend={self.backend!r})"
