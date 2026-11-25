"""Exact cosine search against a fixed enrollment matrix."""
import operator

import numpy as np

from .core import EMBEDDING_DIM


def _unit_rows(values):
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != EMBEDDING_DIM or raw.dtype.kind not in "iuf":
        raise ValueError("Expected a matrix of 512 dimensional real vectors")
    with np.errstate(over="ignore"):
        rows = raw.astype(np.float32, copy=False).astype(np.float64)
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite embedding")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, norms, out=rows, where=norms != 0)
    return rows


class Gallery:
    """Enrollment row number is the stable ID. Equal scores prefer the earlier row."""
    def __init__(self, embeddings):
        self._rows = _unit_rows(embeddings)
        if not len(self._rows):
            raise ValueError("Empty enrollment matrix")
        if not np.all(np.any(self._rows, axis=1)):
            raise ValueError("Enrollment vectors must have a direction")

    def search(self, queries, k=1):
        """Return ID and cosine matrices of shape (query count, k)."""
        k = operator.index(k)
        if not 1 <= k <= len(self._rows):
            raise ValueError("k must be within the enrollment size")
        queries = _unit_rows(queries)
        ids = np.empty((len(queries), k), dtype=np.intp)
        scores = np.empty((len(queries), k), dtype=np.float64)
        # Bound temporary score rows independently of the submitted batch size.
        block = max(1, (1 << 20) // (8 * len(self._rows)))
        for start in range(0, len(queries), block):
            values = queries[start:start + block] @ self._rows.T
            np.clip(values, -1., 1., out=values)
            for i, row in enumerate(values, start):
                if k == 1:
                    selected = [row.argmax()]
                else:
                    cutoff = np.partition(row, len(row) - k)[-k]
                    above = np.flatnonzero(row > cutoff)
                    ties = np.flatnonzero(row == cutoff)[:k - len(above)]
                    candidates = np.concatenate((above, ties))
                    selected = candidates[np.lexsort((candidates, -row[candidates]))]
                ids[i] = selected
                scores[i] = row[selected]
        return ids, scores

    def identify(self, queries, *, min_score, min_margin):
        """Return enrollment row IDs, or -1 when score or separation is insufficient.

        Thresholds here are inclusive. Equal best scores always require abstention.
        Use one row per identity when interpreting margins as identity separation.
        """
        if not -1 <= min_score <= 1 or not 0 <= min_margin <= 2:
            raise ValueError("Invalid cosine score or margin threshold")
        ids, values = self.search(queries, min(2, len(self._rows)))
        accepted = (values[:, 0] >= min_score) & np.any(np.asarray(queries, dtype=np.float32), axis=1)
        if values.shape[1] == 2:
            gap = values[:, 0] - values[:, 1]
            accepted &= (gap > 0) & (gap >= min_margin)
        elif min_margin > 0:
            accepted[:] = False
        return np.where(accepted, ids[:, 0], -1)


def enrollment_template(embeddings, weights=None, *, mode="mean"):
    """Combine views of one identity. Return a unit vector and directional agreement.

    The medoid maximizes weighted cosine similarity to all included views.
    Agreement is the resultant length on the unit sphere and is not a probability.
    """
    rows = _unit_rows(embeddings)
    if not len(rows) or mode not in ("mean", "medoid"):
        raise ValueError("Expected at least one view and mean or medoid mode")
    weight = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight.shape != (len(rows),) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Expected one finite nonnegative weight per view")
    usable = np.any(rows, axis=1) & (weight > 0)
    if not usable.any():
        raise ValueError("No usable enrollment view")
    rows, weight = rows[usable], weight[usable]
    weight = weight / weight.max()
    resultant = (weight / weight.sum()) @ rows
    length = np.linalg.norm(resultant)
    if length <= 8 * np.finfo(np.float64).eps:
        raise ValueError("Views have no consistent direction")
    vector = resultant / length if mode == "mean" else rows[np.argmax(rows @ resultant)]
    return vector.astype(np.float32), float(min(length, 1.))
