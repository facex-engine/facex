"""Exercise the SDK and Flask endpoint with FACEX_LIB, EMBED_WEIGHTS and DETECT_WEIGHTS set."""
import ctypes
import io
import os
import sys

import numpy as np
from PIL import Image

from verify_yunet import ROOT, native_run

sys.path[:0] = [str(ROOT / "python"), str(ROOT / "docker")]
from facex import FaceX
from facex.yunet import prepare_rectangle
from server import app


def main():
    lib = ctypes.CDLL(os.environ["FACEX_LIB"])
    lib.detect_init.restype = ctypes.c_void_p
    det = ctypes.c_void_p(lib.detect_init(os.fsencode(os.environ["DETECT_WEIGHTS"])))
    assert det
    lib.detect_set_score_threshold(det, ctypes.c_float(.58))
    lib.align_face.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_void_p]
    sdk = FaceX(embed_weights=os.environ["EMBED_WEIGHTS"], detect_weights=os.environ["DETECT_WEIGHTS"],
                input_policy="rectangle")
    client = app.test_client()
    patch = Image.open(ROOT / "data/test_face.jpg").convert("RGB").resize((241, 241))
    try:
        for w, h in ((797, 451), (451, 797), (640, 640)):
            image = Image.new("RGB", (w, h))
            image.paste(patch, ((w - 241) // 2, (h - 241) // 2))
            rgb = np.asarray(image)
            canvas, rw, rh = prepare_rectangle(rgb)
            ch, cw = canvas.shape[:2]
            expected = native_run(lib, det, canvas)
            faces = sdk.detect(rgb)
            assert len(expected) == len(faces) > 0
            for row, face in zip(expected, faces):
                coords = np.r_[row[:4], row[5:]].reshape(-1, 2)
                coords -= [(cw-rw)//2, (ch-rh)//2]
                coords *= [w/rw, h/rh]
                np.clip(coords, [0, 0], [w, h], out=coords)
                actual = np.r_[face["bbox"], np.ravel(face["keypoints"])].reshape(-1, 2)
                np.testing.assert_allclose(actual, coords, atol=.0002)
                pts = np.ascontiguousarray(face["keypoints"], dtype="f")
                aligned = np.empty((112, 112, 3), dtype="f")
                assert lib.align_face(rgb.ctypes.data, w, h, pts.ctypes.data, aligned.ctypes.data)
                np.testing.assert_allclose(face["embedding"], sdk.embed(aligned), atol=1e-5)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            response = client.post("/detect", data={"file": (buf, "image.png")})
            assert response.status_code == 200, response.data
            returned = response.get_json()["faces"]
            assert len(returned) == len(faces)
            for api, face in zip(returned, faces):
                for key in ("bbox", "keypoints", "score", "embedding"):
                    np.testing.assert_allclose(api[key], face[key], atol=1e-5)
            print(w, h, "canvas", cw, ch, "faces", len(faces))
        assert client.post("/detect").status_code == 400
        assert client.post("/detect", data={"file": (io.BytesIO(b"x"), "bad.png")}).status_code == 400
        blank = io.BytesIO()
        Image.new("RGB", (400, 225)).save(blank, format="PNG")
        blank.seek(0)
        assert client.post("/detect", data={"file": (blank, "blank.png")}).get_json()["faces"] == []
    finally:
        lib.detect_free(det)
        sdk.close()


if __name__ == "__main__":
    main()
