"""Exercise admission and recovery without model files."""
import io
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import server


class AdmissionTest(unittest.TestCase):
    image = (Path(__file__).resolve().parents[1] / "data/test_face.jpg").read_bytes()

    def post(self, endpoint, value=None):
        headers = {} if value is None else {"X-FaceX-Deadline-Ms": value}
        return server.app.test_client().post(endpoint, headers=headers,
            data={"file": (io.BytesIO(self.image), "face.jpg")})

    def test_both_routes(self):
        for endpoint, name, lock in (("/detect", "_detector", server._detector_lock),
                                     ("/embed", "_fx", server._embed_lock)):
            engine = Mock()
            engine.detect.return_value = []
            engine.embed.return_value = np.zeros(512, dtype="f")
            with patch.object(server, name, engine), patch.object(server, "QUEUE_WAIT_MS", 20):
                self.assertEqual(self.post(endpoint, "invalid").status_code, 400)
                self.assertEqual(self.post(endpoint, "1").json["code"], "inference_busy")
                with lock:
                    future = str(time.time_ns() // 1_000_000 + 10000)
                    self.assertEqual(self.post(endpoint, future).json["code"], "inference_busy")
                self.assertFalse(engine.method_calls)
                self.assertEqual(self.post(endpoint, future).status_code, 200)
                self.assertEqual(len(engine.method_calls), 1)

    def test_full_preparation_capacity_skips_image_decode(self):
        for endpoint, slots in (("/detect", server._detect_capacity),
                                ("/embed", server._embed_capacity)):
            with slots, slots, patch.object(server, "QUEUE_WAIT_MS", 20):
                with patch.object(server.Image, "open") as open_image:
                    response = self.post(endpoint)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json["code"], "inference_busy")
                open_image.assert_not_called()

    def test_client_deadline_with_no_server_limit(self):
        with patch.object(server, "QUEUE_WAIT_MS", 0), server._detector_lock:
            deadline = str(time.time_ns() // 1_000_000 + 30)
            self.assertEqual(self.post("/detect", deadline).status_code, 503)
            self.assertEqual(server.app.test_client().get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
