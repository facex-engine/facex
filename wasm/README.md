# facex-wasm

Face recognition in the browser. 75 KB of WebAssembly. No server, no dependencies.

Detect faces, align them, compute 512-dimensional embeddings, and verify identity -- all client-side, all private.

## Requirements

- Chrome 91+, Firefox 89+, Safari 16.4+, Edge 91+ (WebAssembly SIMD required)
- HTTPS (required for camera access)

## Install

```bash
npm install facex-wasm
```

## Quick start

```js
import { FaceXSDK } from 'facex-wasm';

const fx = new FaceXSDK();
await fx.load();

// Full pipeline: detect + align + embed
const result = fx.process(videoElement);
console.log(result.faces);       // [{x1, y1, x2, y2, score, kps}, ...]
console.log(result.embeddings);  // [Float32Array(512), ...]
console.log(result.ms);          // processing time in ms
```

## Script tag usage

```html
<script src="detect.js"></script>
<script src="facex.js"></script>
<script src="align.js"></script>
<script src="facex-sdk.js"></script>
<script>
  const fx = new FaceXSDK();
  await fx.load();
  const result = fx.process(videoElement);
</script>
```

## API

### `new FaceXSDK(options?)`

Create an SDK instance.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `detSize` | `number` | `160` | Detection input size (pixels) |
| `threshold` | `number` | `0.3` | Cosine similarity threshold for match |
| `detWeightsUrl` | `string` | `'yunet_fp32.bin'` | URL to detector weights |
| `embWeightsUrl` | `string` | `'edgeface_xs_fp32.bin'` | URL to embedder weights |
| `onProgress` | `function` | `null` | Progress callback during loading |

### `fx.load(): Promise<FaceXSDK>`

Load WASM engines and model weights. Call once before using other methods.

### `fx.detect(source): Face[]`

Detect faces in a video, image, or canvas element. Returns array of faces with bounding boxes and 5 keypoints.

### `fx.embed(imageData): Float32Array`

Compute a 512-dim embedding from an aligned 112x112 face ImageData.

### `fx.process(source): ProcessResult`

Full pipeline: detect all faces, align each one, compute embeddings.

```js
const { faces, embeddings, ms } = fx.process(videoElement);
```

### `fx.verify(source, refEmbedding): VerifyResult`

Compare a live frame against a reference embedding.

```js
const ref = fx.captureReference(videoElement);
// Later...
const { match, similarity } = fx.verify(videoElement, ref.embedding);
if (match) console.log(`Verified! Similarity: ${similarity.toFixed(2)}`);
```

### `fx.captureReference(source): CaptureResult | null`

Capture a reference embedding from the current frame. Returns `null` if no face detected.

### `fx.cosSim(a, b): number`

Cosine similarity between two 512-dim embeddings. Returns value between -1 and 1.

## Liveness detection

Basic liveness check using motion, blink, and size variation analysis.

```js
import { LivenessDetector } from 'facex-wasm';

const liveness = new LivenessDetector();

function onFrame() {
  const faces = fx.detect(videoElement);
  const { alive, confidence, reason } = liveness.update(faces[0] || null);
  console.log(alive, confidence, reason);
}
```

## Utilities

```js
import { checkWasmSimd, checkBrowserSupport } from 'facex-wasm';

if (!checkWasmSimd()) {
  alert('Your browser does not support WebAssembly SIMD');
}

const issues = checkBrowserSupport();
if (issues.length > 0) {
  console.warn('Browser issues:', issues);
}
```

## Weight files

The SDK needs two weight files served alongside your app:

| File | Size | Description |
|------|------|-------------|
| `yunet_fp32.bin` | 208 KB | Face detector weights |
| `edgeface_xs_fp32.bin` | 7 MB | Face embedder weights |

Weights are cached in IndexedDB after first load.

## Architecture

| File | Size | Purpose |
|------|------|---------|
| `detect.wasm` | 30 KB | Face detector (YuNet, WASM+SIMD) |
| `facex.wasm` | 45 KB | Face embedder (EdgeFace-XS, WASM+SIMD) |
| `facex-sdk.js` | 10 KB | Unified SDK class |
| `align.js` | 4 KB | 5-point affine face alignment |
| `liveness.js` | 5 KB | Motion/blink liveness checks |

Total engine size: ~75 KB WASM + 19 KB JS (before gzip).

## License

Apache-2.0
