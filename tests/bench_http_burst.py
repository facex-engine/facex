"""Send overlapping requests to a local Gunicorn Unix socket and verify response ownership."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hashlib import sha256
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import socket
import threading
from time import perf_counter

import numpy as np
from PIL import Image


class LocalHTTP(HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=30)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


_clients = threading.local()


def wsgi_digest(endpoint, field, jpeg):
    from server import app
    if not hasattr(_clients, "client"):
        _clients.client = app.test_client()
    response = _clients.client.post(endpoint, data={"file": (io.BytesIO(jpeg), "face.jpg")})
    assert response.status_code == 200, response.data
    return sha256(json.dumps(response.json[field], sort_keys=True).encode()).digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--socket")
    mode.add_argument("--wsgi-processes", type=int)
    mode.add_argument("--wsgi-threads", action="store_true")
    args = parser.parse_args()
    jpegs = []
    payloads = []
    for path in sorted((Path(__file__).parent / "golden").glob("*.npz"))[:8]:
        with np.load(path) as fixture:
            buf = io.BytesIO()
            Image.fromarray(fixture["image"]).save(buf, "JPEG")
        jpegs.append(buf.getvalue())
        payloads.append(b'--face\r\nContent-Disposition: form-data; name="file"; filename="f.jpg"\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n--face--\r\n')
    assert len(payloads) == 8
    workers = ProcessPoolExecutor(args.wsgi_processes) if args.wsgi_processes else None
    for endpoint, field in (("/detect", "faces"), ("/embed", "embedding")):
        def send(i):
            start = perf_counter()
            if not args.socket:
                params = (endpoint, field, jpegs[i % 8])
                digest = workers.submit(wsgi_digest, *params).result() if workers else wsgi_digest(*params)
                return perf_counter() - start, digest
            conn = LocalHTTP(args.socket)
            try:
                conn.request("POST", endpoint, payloads[i % 8],
                             {"Content-Type": "multipart/form-data; boundary=face"})
                response = conn.getresponse()
                raw = response.read()
                elapsed = perf_counter() - start
                assert response.status == 200, raw
                result = json.loads(raw)[field]
                return elapsed, sha256(json.dumps(result, sort_keys=True).encode()).digest()
            finally:
                conn.close()
        expected = [send(i)[1] for i in range(8)]
        assert len(set(expected)) == 8
        for concurrency in (1, 8, 32):
            with ThreadPoolExecutor(concurrency) as pool:
                start = perf_counter()
                samples = list(pool.map(send, range(64)))
                elapsed = perf_counter() - start
            assert all(digest == expected[i % 8] for i, (_, digest) in enumerate(samples))
            lat = [s * 1000 for s, _ in samples]
            print(endpoint, concurrency, "clients", round(64/elapsed, 1), "req/s",
                  "p50/p95 ms", np.round(np.percentile(lat, [50, 95]), 1), flush=True)

    if workers:
        workers.shutdown()


if __name__ == "__main__":
    main()
