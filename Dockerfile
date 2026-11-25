# FaceX — Face recognition REST API
# Build: docker build -t facex .
# Run:   docker run -p 8080:8080 facex
#
# API:
#   POST /detect  — detect faces + embeddings (multipart image)
#   POST /embed   — embed pre-aligned 112x112 face
#   POST /compare — compare two embeddings
#   GET  /health  — health check

FROM gcc:13-bookworm AS builder

WORKDIR /build
COPY src/ src/
COPY include/ include/
COPY Makefile .

# Build the embedding CLI and the shared detector against the runtime libc.
RUN make cli CFLAGS="-O3 -march=x86-64-v3" && \
    gcc -O3 -march=x86-64-v3 -shared -fPIC -Iinclude -o libfacex.so \
    src/facex.c src/detect.c src/align.c src/transformer_ops.c \
    src/gemm_int8_4x8c8.c src/threadpool.c src/weight_crypto.c -lm -lpthread && \
    strip facex-cli libfacex.so

FROM python:3.11-slim-bookworm
RUN pip install --no-cache-dir flask gunicorn pillow numpy
WORKDIR /app
COPY --from=builder /build/facex-cli /app/facex-server
COPY --from=builder /build/libfacex.so /app/libfacex.so
COPY python/facex /app/facex
COPY docker/server.py /app/
ENV FACEX_LIB=/app/libfacex.so

# Weights must be mounted or baked in
# docker run -v /path/to/weights:/app/weights facex
EXPOSE 8080
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "server:app"]
