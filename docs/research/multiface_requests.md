# Fresh embeddings and simultaneous requests

The Flask endpoint accepts `POST /detect?embedding_format=base64_f32le`. The usual numeric array is the default. In compact mode each face's embedding is one standard Base64 string containing 512 little-endian float32 values. Decode with `np.frombuffer(base64.b64decode(face["embedding"]), dtype="<f4")`.

On Linux x86-64-v3, a tiled 32-face JPEG took a median 157.29 ms per complete default request and 152.75 ms compact, versus 39.39 ms for geometry only. Response size fell from about 362,085 to 97,882 bytes. All coordinates and all 16,384 float bit patterns matched. The warmed Flask client ran 12 timed requests per mode on one pinned CPU, rotating the order. Separate replay timings for 32 faces were 5.16 ms for alignment, 116.25 ms for the embedding model and SDK calls, and 7.31 ms for conversion and JSON serialization of the vectors. Independent medians do not add up to endpoint latency.

A separate process-group experiment on the same JPEG and unconstrained CPU affinity split the 32 faces over four persistent processes. Alignment and fresh embedding took 61.83 ms versus 132.10 ms sequentially, including dispatch, copying the RGB image to each process and returning the results. All vectors matched bitwise. This experiment has no competing clients and does not enable a process pool inside the service.

## Burst measurements

Each cell below used 64 requests to 8 different JPEGs. The client checked every response payload against the same image's sequential reference, excluding timing. Two isolated processes ran the Flask endpoints. Client timing includes queueing, process dispatch, response parsing and checksum computation. HTTP sockets, proxies, authentication and a downstream identity database are outside the WSGI measurement.

| Concurrent clients | Detect requests/s | Detect p95 ms | Embed requests/s | Embed p95 ms |
|---:|---:|---:|---:|---:|
| 1 | 19.4 | 59 | 92.8 | 13 |
| 8 | 36.1 | 268 | 207.1 | 39 |
| 32 | 39.1 | 816 | 206.9 | 155 |

A separate single-process threaded run also matched all sequential references, including 32 concurrent clients. The embedding CLI's binary request and response now share one lock per parent process. A terminated child yields a 503 and a later request creates a fresh worker.

With the native library, CLI and mounted weights configured, run from the repository root:

~~~sh
PYTHONPATH=python:docker python tests/bench_multiface_response.py --batch-workers 4
FACEX_QUEUE_WAIT_MS=0 PYTHONPATH=python:docker python tests/bench_http_burst.py --wsgi-processes 2
FACEX_QUEUE_WAIT_MS=0 PYTHONPATH=python:docker python tests/bench_http_burst.py --wsgi-threads
~~~

For a running Gunicorn instance listening on a Unix socket, pass `--socket /path/to/socket` instead.

## Inference admission deadline

The Flask worker allows 250 ms by default from handler entry to starting inference on `/detect` and `/embed`. Set `FACEX_QUEUE_WAIT_MS` to a different budget in milliseconds, or zero for unlimited local waiting. An admitted inference runs to completion. A busy response is HTTP 503 with `code: inference_busy`. Live-video clients should discard the expired frame and submit a recent frame when ready.

To include time already spent uploading or waiting at a proxy or a synchronous Gunicorn worker, clients can send `X-FaceX-Deadline-Ms` with the absolute Unix time in milliseconds after which processing the frame is no longer useful. This requires synchronized clocks. The earlier deadline applies. The worker checks expiration before accessing the multipart body and again before model execution. Queues outside Flask can still delay delivery of the 503 response.

In a local 64-request burst with 32 clients, one threaded Flask process and eight real JPEGs, unlimited admission returned 64 successful responses with p95 1767 ms. A 250 ms budget admitted 10 (p95 313 ms) and rejected 54 (p95 253 ms). All admitted responses matched their image's sequential result. A subsequent request completed normally. These are WSGI timings for a single burst.

## Limit decoded images during overload

For requests with a deadline, after multipart parsing each process allows up to two requests per inference endpoint to prepare an image or use the engine. This leaves room to decode the next image while the model is busy. Waiting for preparation and waiting for the model share the original deadline. Expiration in the first queue skips image decoding. Requests with unlimited admission keep the original preprocessing path.

We compared against abc177f in six alternating pairs of local bursts on four pinned logical CPUs. Each burst used one warmed threaded Flask process, 32 clients and 64 detection requests drawn from eight fixture JPEGs resized to 1920 by 1080. The local admission budget was 250 ms. CPU includes the in-process test client. Latency includes successful and rejected responses.

| Median per burst | Before | After |
|---|---:|---:|
| JPEGs opened for decoding | 64 | 10.5 |
| Process CPU seconds | 2.135 | 0.764 |
| p95 response time in ms | 333 | 267 |

Across 384 requests per build, 39 succeeded before and 37 after. All accepted face results matched their sequential references. The measured benefit is less wasted work under overload. A synchronous single-request worker has no competing in-process decodes.
