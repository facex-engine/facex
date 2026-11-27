# Build the FRVT 1:1 adapter with Ubuntu 20.04 and the pinned runtime SDK.
# Run:
#   docker build -t facex-frvt-build .
#   docker run --rm -v ${PWD}/build:/work/build facex-frvt-build
# Output: build/lib/libfrvt_11_facex_001.so

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl unzip ca-certificates pkg-config \
        gpg gpg-agent \
    && rm -rf /var/lib/apt/lists/*

COPY prepare_ort.sh /tmp/prepare_ort.sh
RUN sh /tmp/prepare_ort.sh /opt/onnxruntime

WORKDIR /work
COPY . /work

CMD bash -ec '\
  cmake -S /work -B /tmp/facex-build -DONNXRUNTIME_ROOT=/opt/onnxruntime -DCMAKE_BUILD_TYPE=Release && \
  cmake --build /tmp/facex-build --parallel && \
  cmake --install /tmp/facex-build --prefix /work/build && \
  cp -a /opt/onnxruntime/lib/libonnxruntime*.so* /work/build/lib/ \
'
