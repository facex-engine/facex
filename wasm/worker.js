/*
 * worker.js — FaceX inference in a Web Worker.
 * Keeps main thread responsive while running detection + embedding.
 *
 * Messages:
 *   {type: 'init'} → loads engines and weights
 *   {type: 'detect', pixels: Float32Array, W, H} → returns faces
 *   {type: 'embed', pixels: Float32Array} → returns embedding Float32Array
 */

importScripts('detect.js', 'facex.js', 'cache.js', 'facex-sdk.js');
let engines, loading;

function detect(pixels, w, h) {
  const rgb = pixels instanceof Float32Array
    ? Uint8Array.from(pixels, x => Math.max(0, Math.min(255, Math.round((x + 1) * 127.5))))
    : pixels;
  return facexDetectPixels(engines, rgb, w, h);
}

function embed(pixels) { return facexEmbedPixels(engines, pixels); }

onmessage = async function(e) {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      loading ??= loadFaceXEngines(msg).then(result => { engines = result; })
        .catch(error => { loading = null; throw error; });
      await loading;
      postMessage({type: 'ready'});
      return;
    }
    if (!loading) throw new Error('Initialize the worker first');
    await loading;
    switch (msg.type) {
    case 'detect':
      const faces = detect(msg.pixels, msg.W, msg.H);
      postMessage({ type: 'detect', faces });
      break;
    case 'embed':
      const emb = embed(msg.pixels);
      postMessage({ type: 'embed', emb }, [emb.buffer]);
      break;
    case 'pipeline': {
      const t0 = performance.now();
      const f = detect(msg.detPixels, msg.detW, msg.detH);
      const detMs = performance.now() - t0;
      let emb = null, embMs = 0;
      if (msg.embPixels) {
        const t1 = performance.now();
        emb = embed(msg.embPixels);
        embMs = performance.now() - t1;
      }
      postMessage({ type: 'pipeline', faces: f, emb, detMs, embMs }, emb ? [emb.buffer] : []);
      break;
    }
    }
  } catch (error) {
    postMessage({type: 'error', message: error.message});
  }
};
