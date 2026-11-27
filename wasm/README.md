# FaceX browser SDK

Run YuNet detection, five-point alignment and EdgeFace XS recognition in a browser or Web Worker. The SDK returns boxes and landmarks in source-image pixels and one 512-component embedding per aligned face. The [hosted mesh and anti-spoofing demo](https://facex-engine.github.io/facex/demo/) uses a separate ONNX pipeline.

## Run the SDK demo from a checkout

Follow the [native guide](../README.md#build-the-native-runtime) to download EdgeFace and export YuNet. From the repository root:

```sh
cp data/edgeface_xs_fp32.bin weights/yunet_fp32.bin wasm/
python -m http.server 8000 --bind 127.0.0.1 --directory wasm
```

Open `http://127.0.0.1:8000/demo.html`. Camera access requires localhost or HTTPS. A canvas or preloaded image also works. For offline deployment, include both weights, both WASM files and the JavaScript assets in the application and serve them locally.

## Use the package

Build or install the package from this revision with `npm install ./wasm` from the repository root. For a bundled application, copy `detect.wasm` and `facex.wasm` to the asset directory and configure the two weight URLs. The model files are distributed separately.

```js
import {FaceXSDK, checkWasmSimd} from 'facex-wasm';

export async function recognize(canvas) {
  if (!checkWasmSimd()) throw new Error('WebAssembly SIMD is required');
  const sdk = new FaceXSDK({
    wasmBaseUrl: '/assets/facex/',
    detWeightsUrl: '/models/yunet_fp32.bin',
    embWeightsUrl: '/models/edgeface_xs_fp32.bin',
  });
  await sdk.load();
  try {
    return sdk.process(canvas);
  } finally {
    await sdk.close();
  }
}
```

For a video stream, load the SDK once, reuse it across frames and close it when the stream ends. Default weight URLs resolve relative to the page. Cache entries include the resolved URL and model kind. Use versioned asset URLs when replacing a model.

Classic script tags load in this order:

```html
<script src="detect.js"></script>
<script src="facex.js"></script>
<script src="align.js"></script>
<script src="cache.js"></script>
<script src="simd_check.js"></script>
<script src="facex-sdk.js"></script>
```

Append `liveness.js` to use the optional motion heuristic. Five detector landmarks provide a coarse motion signal. Evaluate presentation-attack detection separately for an authentication deployment.

## API and data contracts

| Method or setting | Contract |
|---|---|
| `load()`, `ready`, `close()` | Concurrent loads share initialization. Close releases both native handles. A closed instance can load again |
| `detSize` | Square detection budget in pixels, default 160. The resize preserves aspect ratio and applies letterboxing |
| `detect(source)` | Returns boxes, confidence and five pairs of keypoint coordinates. Accepts an image, canvas or video |
| `embed(imageData)` | Aligned 112 × 112 RGBA ImageData → 512 floats |
| `process(source)` | Returns `faces`, `embeddings` and elapsed `ms` from one captured frame |
| `captureReference(source)` | Returns the first face, its embedding and an independently owned aligned canvas, or `null` |
| `verify(source, reference)` | Scores the first detected face. `noFace` identifies empty frames. `threshold` defaults to 0.3 and is configurable |
| `cosSim(a,b)` | Cosine similarity in `[-1,1]` for 512-component inputs, zero for a zero vector |

The worker accepts `init` with optional absolute weight URLs, `detect` with interleaved RGB bytes or normalized HWC float32, and `embed` with 112 × 112 normalized HWC float32. Wait for `ready`. Error replies contain a `message`. [Worker protocol](worker.js) · [TypeScript declarations](index.d.ts)

## Rebuild the detector

From the repository root, run `make wasm-detector` with Emscripten installed. The distributed detector is compiled from the native YuNet source. The SDK uses the module factory `DetectModule`, copies byte RGB into the detector and copies float RGB into the embedder. [Detector source](../src/detect.c) · [Model hashes and licensing](../README.md#licensing-and-provenance)
