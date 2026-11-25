"""Profile 32 fresh face embeddings through the Flask endpoint and native API."""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import ctypes
import io
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
import server


def embed_group(image, landmarks):
    server.initialize_detector()
    return [server._detector.embed_landmarks(image, points) for points in landmarks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-workers", type=int, choices=(2, 4))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with Image.open(root / "data/test_face.jpg") as source:
        tile = source.convert("RGB").resize((112, 112))
    image = Image.new("RGB", (896, 448))
    for i in range(32):
        image.paste(tile, (i % 8 * 112, i // 8 * 112))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    payload = buf.getvalue()
    rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    client = server.app.test_client()

    def request(query):
        return client.post("/detect" + query, data={"file": (io.BytesIO(payload), "faces.jpg")})

    queries = ("", "?embedding_format=base64_f32le", "?embeddings=false")
    samples = [[] for _ in queries]
    for rep in range(13):
        results = {}
        for j in (np.arange(3) + rep) % 3:
            start = time.perf_counter_ns()
            response = request(queries[j])
            elapsed = time.perf_counter_ns() - start
            assert response.status_code == 200
            faces = response.json["faces"]
            assert len(faces) == 32
            if rep:
                samples[j].append(elapsed / 1e6)
            results[j] = faces
            if rep == 0:
                print(queries[j] or "array", len(response.data), "bytes")
        for a, b, geom in zip(results[0], results[1], results[2]):
            expected = np.asarray(a.pop("embedding"), dtype="<f4").tobytes()
            assert base64.b64decode(b.pop("embedding"), validate=True) == expected
            assert a == b == geom
    assert request("?embedding_format=bad").status_code == 400
    print("Full request median ms (array, compact, geometry)", list(map(statistics.median, samples)))

    engine = server._detector
    float_ptr = ctypes.POINTER(ctypes.c_float)
    byte_ptr = ctypes.POINTER(ctypes.c_uint8)
    align = engine._lib.align_face
    align.argtypes = [byte_ptr, ctypes.c_int, ctypes.c_int, float_ptr, float_ptr]
    pts = [np.array(face["keypoints"], dtype="f").ctypes.data_as(float_ptr) for face in results[2]]
    aligned = [np.empty((112, 112, 3), dtype="f") for _ in pts]
    src = rgb.ctypes.data_as(byte_ptr)
    stages = [[], [], []]
    for rep in range(9):
        start = time.perf_counter_ns()
        for point, dest in zip(pts, aligned):
            assert align(src, rgb.shape[1], rgb.shape[0], point, dest.ctypes.data_as(float_ptr))
        mid = time.perf_counter_ns()
        vectors = [engine.embed(face) for face in aligned]
        end = time.perf_counter_ns()
        with server.app.app_context():
            server.app.json.dumps([v.tolist() for v in vectors])
        done = time.perf_counter_ns()
        if rep:
            for arr, dt in zip(stages, (mid-start, end-mid, done-end)):
                arr.append(dt / 1e6)
    print("Stage median ms (align, embed, array JSON)", list(map(statistics.median, stages)))

    if args.batch_workers:
        landmarks = [face["keypoints"] for face in results[2]]
        groups = np.array_split(landmarks, args.batch_workers)
        timings = [[], []]
        with ProcessPoolExecutor(args.batch_workers) as pool:
            for rep in range(13):
                for parallel in (rep % 2, 1 - rep % 2):
                    start = time.perf_counter_ns()
                    if parallel:
                        jobs = [pool.submit(embed_group, rgb, group) for group in groups]
                        batch = [v for job in jobs for v in job.result()]
                    else:
                        batch = embed_group(rgb, landmarks)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    assert np.stack(batch).tobytes() == np.stack(vectors).tobytes()
                    if rep:
                        timings[parallel].append(elapsed)
        print("Fresh landmark embeddings median ms (serial, process groups)",
              list(map(statistics.median, timings)))


if __name__ == "__main__":
    main()
