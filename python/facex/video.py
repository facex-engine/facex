"""Embedding refresh for one camera with externally assigned track IDs."""
from bisect import bisect_left, bisect_right
import math

import numpy as np
from PIL import Image


def _iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w*h
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def _ambiguous_tracks(boxes, tracks):
    flags = [bool(face.get("ambiguous")) for face in tracks]
    if len(boxes) < 2 or all(flags):
        return flags

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    left = [boxes[i][0] for i in order]
    max_right = []
    furthest = -math.inf
    for i in order:
        furthest = max(furthest, boxes[i][2])
        max_right.append(furthest)

    for i, box in enumerate(boxes):
        if flags[i]:
            continue
        # Outside this range the horizontal intersection has zero width.
        begin = bisect_right(max_right, box[0])
        end = bisect_left(left, box[2])
        for k in range(begin, end):
            j = order[k]
            other = boxes[j]
            if i == j or other[2] <= box[0] or other[3] <= box[1] or box[3] <= other[1]:
                continue
            if _iou(box, other) > .15:
                flags[i] = flags[j] = True
                break
    return flags


def _appearance(rgb, box):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    patch = rgb[max(0, math.floor(y1)):min(h, math.ceil(y2)),
                max(0, math.floor(x1)):min(w, math.ceil(x2))]
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h or not patch.size:
        raise ValueError("Track box lies outside the image")
    thumb = np.asarray(Image.fromarray(patch).convert("L").resize((16, 16)), dtype=np.float32)
    sharpness = np.abs(np.diff(thumb, axis=0)).mean() + np.abs(np.diff(thumb, axis=1)).mean()
    centered = thumb - thumb.mean()
    signature = centered / max(1., np.linalg.norm(centered))
    quality = math.sqrt((x2-x1)*(y2-y1)) * (1 + min(64., sharpness)/64)
    return signature, quality


class VideoEmbeddings:
    """One instance per stream. Omitted or ambiguous tracks must reacquire."""
    def __init__(self, engine, max_age=12):
        if int(max_age) != max_age or max_age < 1:
            raise ValueError("max_age must be a positive frame count")
        self.engine = engine
        self.max_age = max_age
        self.reset()

    def reset(self):
        self._tracks = {}
        self._last_frame = None

    def update(self, rgb, frame_number, tracks):
        """Return geometry, embedding and refresh flag for each supplied track."""
        if self._last_frame is not None and frame_number <= self._last_frame:
            raise ValueError("Frame numbers must increase. Call reset() when seeking.")
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image")
        tracks = list(tracks)
        ids = [f["track_id"] for f in tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate track ID in one frame")
        boxes = [tuple(face["bbox"]) for face in tracks]
        for box in boxes:
            if len(box) != 4 or not all(map(math.isfinite, box)) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid track box")
        ambiguity = _ambiguous_tracks(boxes, tracks)
        next_tracks, results = {}, []
        for key, face, box, ambiguous in zip(ids, tracks, boxes, ambiguity):
            signature, quality = _appearance(rgb, box)
            prev = self._tracks.get(key)
            uncertain = face["score"] < .90 or min(box[2]-box[0], box[3]-box[1]) < 40
            refresh = (uncertain or prev is None or self._last_frame != frame_number-1 or ambiguous or
                       prev["ambiguous"] or prev["uncertain"] or frame_number-prev["at"] >= self.max_age or
                       _iou(box, prev["box"]) < .35 or quality > 1.25*prev["quality"] or
                       float(np.linalg.norm(signature-prev["signature"])) > .55)
            if refresh:
                vec = self.engine.embed_landmarks(rgb, face["keypoints"])
                state = dict(at=frame_number, quality=quality, signature=signature, vector=vec)
            else:
                state = prev.copy()
            state.update(box=box, ambiguous=ambiguous, uncertain=uncertain)
            next_tracks[key] = state
            results.append(dict(face, embedding=state["vector"].copy(), refreshed=refresh))
        self._tracks = next_tracks
        self._last_frame = frame_number
        return results
