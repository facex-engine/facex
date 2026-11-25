import base64
import unittest

import numpy as np
import server


class CompareTest(unittest.TestCase):
    def test_round_trip(self):
        client = server.app.test_client()
        rng = np.random.default_rng(91)
        a,b = rng.normal(size=(2,512)).astype("<f4")
        def enc(v):
            return base64.b64encode(v).decode("ascii")
        expected = client.post("/compare",json=dict(emb1=a.tolist(),emb2=b.tolist()))
        self.assertEqual(expected.status_code,200)
        for x in (enc(a), a.tolist()):
            for y in (enc(b), b.tolist()):
                self.assertEqual(client.post("/compare",json=dict(emb1=x,emb2=y)).json,expected.json)
        for value in ("!" * 2732, "A" * 2732, [0]*511, [False]*512, [float("nan")]*512):
            self.assertEqual(client.post("/compare",json=dict(emb1=value,emb2=enc(a))).status_code,400)


if __name__ == "__main__":
    unittest.main()
