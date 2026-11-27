/* Shared transports for the browser SDK and Web Worker. */
const facexCommonJS = typeof module !== 'undefined' && !!module.exports;

async function loadFaceXEngines(options = {}) {
  const detectFactory = facexCommonJS ? require('./detect.js') : DetectModule;
  const embedFactory = facexCommonJS ? require('./facex.js') : FaceXModule;
  const loadWeights = facexCommonJS ? require('./cache.js').loadWeightsCached :
    (typeof loadWeightsCached === 'function' ? loadWeightsCached : async url => {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Weight request failed with status ' + response.status);
      return new Uint8Array(await response.arrayBuffer());
    });
  const configuration = {};
  if (options.wasmBaseUrl) {
    const base = new URL(options.wasmBaseUrl.replace(/\/?$/, '/'), globalThis.location.href);
    configuration.locateFile = file => new URL(file, base).href;
  }
  const [det, fx, detBytes, embBytes] = await Promise.all([
    detectFactory({...configuration}), embedFactory({...configuration}),
    loadWeights(options.detWeightsUrl || 'yunet_fp32.bin', 'detector'),
    loadWeights(options.embWeightsUrl || 'edgeface_xs_fp32.bin', 'embedder'),
  ]);
  det.FS.writeFile('/det.bin', detBytes);
  fx.FS.writeFile('/emb.bin', embBytes);
  let detector = 0, embedder = 0;
  try {
    detector = det.cwrap('detect_init', 'number', ['string'])('/det.bin');
    if (!detector) throw new Error('Detector initialization failed');
    embedder = fx.cwrap('facex_init', 'number', ['string', 'string', 'string'])('/emb.bin', null, null);
    if (!embedder) throw new Error('Embedder initialization failed');
  } catch (error) {
    if (detector) det._detect_free(detector);
    if (embedder) fx._facex_free(embedder);
    throw error;
  } finally {
    det.FS.unlink('/det.bin');
    fx.FS.unlink('/emb.bin');
  }
  return {det, fx, detector, embedder};
}

function facexDetectPixels(engines, rgb, width, height) {
  if (!(rgb instanceof Uint8Array) || rgb.length !== width * height * 3) {
    throw new TypeError('Expected packed RGB bytes for the detector');
  }
  const {det, detector} = engines;
  const limit = 10;
  const input = det._malloc(rgb.length), output = det._malloc(limit * 15 * 4);
  try {
    if (!input || !output) throw new Error('Detector allocation failed');
    det.HEAPU8.set(rgb, input);
    const count = det._detect_run(detector, input, width, height, output, limit);
    if (count < 0 || count > limit) throw new Error('Face detection failed');
    const values = new Float32Array(det.HEAPU8.buffer, output, count * 15);
    const faces = [];
    for (let i = 0; i < values.length; i += 15) {
      faces.push({x1: values[i], y1: values[i + 1], x2: values[i + 2], y2: values[i + 3],
        score: values[i + 4], kps: Array.from(values.subarray(i + 5, i + 15))});
    }
    return faces;
  } finally {
    det._free(output);
    det._free(input);
  }
}

function facexEmbedPixels(engines, pixels) {
  if (!(pixels instanceof Float32Array) || pixels.length !== 112 * 112 * 3) {
    throw new TypeError('Expected a normalized 112 by 112 RGB face');
  }
  const {fx, embedder} = engines;
  const input = fx._malloc(pixels.byteLength), output = fx._malloc(512 * 4);
  try {
    if (!input || !output) throw new Error('Embedder allocation failed');
    fx.HEAPF32.set(pixels, input / 4);
    if (fx._facex_embed(embedder, input, output) !== 0) throw new Error('Embedding failed');
    return fx.HEAPF32.slice(output / 4, output / 4 + 512);
  } finally {
    fx._free(output);
    fx._free(input);
  }
}

class FaceXSDK {
  constructor(options = {}) {
    this.detSize = options.detSize ?? 160;
    if (!Number.isSafeInteger(this.detSize) || this.detSize < 32 || this.detSize % 32) {
      throw new RangeError('detSize must be a positive multiple of 32');
    }
    this.threshold = options.threshold ?? 0.3;
    this.detWeightsUrl = options.detWeightsUrl;
    this.embWeightsUrl = options.embWeightsUrl;
    this.wasmBaseUrl = options.wasmBaseUrl;
    this.onProgress = options.onProgress;
    this._engines = null;
    this._loading = null;
    this._source = document.createElement('canvas');
    this._sourceCtx = this._source.getContext('2d', {willReadFrequently: true});
    this._detCanvas = document.createElement('canvas');
    this._detCanvas.width = this._detCanvas.height = this.detSize;
    this._detCtx = this._detCanvas.getContext('2d', {willReadFrequently: true});
    this._crop = document.createElement('canvas');
    this._crop.width = this._crop.height = 112;
    this._cropCtx = this._crop.getContext('2d');
  }

  get ready() { return !!this._engines; }

  load() {
    if (this.ready) return Promise.resolve(this);
    if (!this._loading) {
      this.onProgress?.('Loading engines and weights...');
      this._loading = loadFaceXEngines(this).then(engines => {
        this._engines = engines;
        this.onProgress?.('Ready');
        return this;
      }).finally(() => { this._loading = null; });
    }
    return this._loading;
  }

  async close() {
    if (this._loading) await this._loading.catch(() => {});
    if (this._engines) {
      const {det, detector, fx, embedder} = this._engines;
      det._detect_free(detector);
      fx._facex_free(embedder);
      this._engines = null;
    }
  }

  detect(source) {
    if (!this.ready) throw new Error('Load the SDK before detection');
    const w = source.videoWidth ?? source.naturalWidth ?? source.width;
    const h = source.videoHeight ?? source.naturalHeight ?? source.height;
    if (!(w > 0 && h > 0)) throw new RangeError('Expected a nonempty frame');
    if (this._source.width !== w) this._source.width = w;
    if (this._source.height !== h) this._source.height = h;
    this._sourceCtx.drawImage(source, 0, 0);
    const size = this.detSize, scale = Math.min(size / w, size / h);
    const nw = Math.max(1, Math.round(w * scale)), nh = Math.max(1, Math.round(h * scale));
    const dx = Math.floor((size - nw) / 2), dy = Math.floor((size - nh) / 2);
    this._detCtx.fillStyle = '#000';
    this._detCtx.fillRect(0, 0, size, size);
    this._detCtx.drawImage(this._source, dx, dy, nw, nh);
    const rgba = this._detCtx.getImageData(0, 0, size, size).data;
    const rgb = new Uint8Array(size * size * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4) {
      rgb[j++] = rgba[i]; rgb[j++] = rgba[i + 1]; rgb[j++] = rgba[i + 2];
    }
    const x = v => Math.max(0, Math.min(w, (v - dx) * w / nw));
    const y = v => Math.max(0, Math.min(h, (v - dy) * h / nh));
    return facexDetectPixels(this._engines, rgb, size, size).map(f => ({
      x1: x(f.x1), y1: y(f.y1), x2: x(f.x2), y2: y(f.y2), score: f.score,
      kps: f.kps.map((v, i) => i % 2 ? y(v) : x(v)),
    }));
  }

  embed(image) {
    if (!this.ready) throw new Error('Load the SDK before embedding');
    if (image.width !== 112 || image.height !== 112 || image.data.length !== 112 * 112 * 4)
      throw new RangeError('Expected aligned 112 by 112 RGBA ImageData');
    const pixels = new Float32Array(112 * 112 * 3);
    for (let i = 0, j = 0; i < image.data.length; i += 4) {
      pixels[j++] = image.data[i] / 127.5 - 1;
      pixels[j++] = image.data[i + 1] / 127.5 - 1;
      pixels[j++] = image.data[i + 2] / 127.5 - 1;
    }
    return facexEmbedPixels(this._engines, pixels);
  }

  _frame() { return this._sourceCtx.getImageData(0, 0, this._source.width, this._source.height); }

  _align(frame, face, ctx) {
    const align = facexCommonJS ? require('./align.js').alignFacePixels : alignFacePixels;
    const points = [];
    for (let i = 0; i < face.kps.length; i += 2) points.push(face.kps.slice(i, i + 2));
    return align(frame, points, ctx);
  }

  process(source) {
    const start = performance.now();
    const faces = this.detect(source);
    const frame = faces.length && this._frame();
    const embeddings = faces.map(f => this.embed(this._align(frame, f, this._cropCtx)));
    return {faces, embeddings, ms: performance.now() - start};
  }

  verify(source, reference) {
    const start = performance.now();
    const faces = this.detect(source);
    if (!faces.length) return {faces, noFace: true, similarity: 0, match: false, ms: performance.now() - start};
    const embedding = this.embed(this._align(this._frame(), faces[0], this._cropCtx));
    const similarity = this.cosSim(embedding, reference);
    return {faces, embedding, similarity, match: similarity > this.threshold, ms: performance.now() - start};
  }

  captureReference(source) {
    const [face] = this.detect(source);
    if (!face) return null;
    const alignedCanvas = document.createElement('canvas');
    alignedCanvas.width = alignedCanvas.height = 112;
    const image = this._align(this._frame(), face, alignedCanvas.getContext('2d'));
    return {face, alignedCanvas, embedding: this.embed(image)};
  }

  cosSim(a, b) {
    if (a.length !== 512 || b.length !== 512) throw new RangeError('Expected two 512 component embeddings');
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < 512; i++) {
      dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    if (!Number.isFinite(na) || !Number.isFinite(nb)) throw new RangeError('Nonfinite embedding');
    return na && nb ? Math.max(-1, Math.min(1, dot / Math.sqrt(na * nb))) : 0;
  }
}

if (facexCommonJS) module.exports = {FaceXSDK};
