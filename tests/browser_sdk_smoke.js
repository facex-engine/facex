async () => {
  function check(condition, message) {
    if (!condition) throw new Error(message);
  }
  check(checkWasmSimd(), 'SIMD support');
  const urls = {detWeightsUrl: '/det.bin', embWeightsUrl: '/emb.bin', wasmBaseUrl: '/wasm'};
  const sdk = new FaceXSDK(urls);
  const loading = sdk.load();
  check(loading === sdk.load(), 'Concurrent load');
  await loading;
  const raw = new Uint8Array(await (await fetch('/frame')).arrayBuffer());
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 160;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(160, 160);
  for (let i = 0, j = 0; j < raw.length; i += 4, j += 3) {
    image.data.set(raw.subarray(j, j + 3), i);
    image.data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const {faces, embeddings} = sdk.process(canvas);
  check(faces.length === 1 && Math.abs(faces[0].score - 0.913699) < 0.0001, 'Face geometry from native reference');
  check(embeddings[0].length === 512 && embeddings[0].every(Number.isFinite), 'Embedding shape');
  const ref = sdk.captureReference(canvas);
  check(sdk.verify(canvas, ref.embedding).similarity > 0.99999, 'Self verification');
  check(ref.alignedCanvas !== sdk.captureReference(canvas).alignedCanvas, 'Independent reference captures');
  for (let y = 0; y < 160; y++) for (let x = 0; x < 160; x++) {
    const p = (y * 160 + x) * 4;
    image.data[p] = x;
    image.data[p + 1] = y;
  }
  ctx.putImageData(image, 0, 0);
  const rotated = REF_POINTS.map(([x, y]) => [140 - y, x + 3]);
  const aligned = alignFace(ctx, 160, 160, rotated, ref.alignedCanvas.getContext('2d'));
  const k = (70 * 112 + 40) * 4;
  check(aligned.data[k] === 70 && aligned.data[k + 1] === 43, 'Inverse rotation and translation');
  await sdk.close();
  check(!sdk.ready, 'Close releases native handles');
  await sdk.load();
  await sdk.close();
  const worker = new Worker('/wasm/worker.js');
  function send(message) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Worker timeout for ' + message.type)), 20000);
      worker.onerror = e => { clearTimeout(timer); reject(new Error(e.message)); };
      worker.onmessage = e => {
        clearTimeout(timer);
        e.data.type === 'error' ? reject(new Error(e.data.message)) : resolve(e.data);
      };
      worker.postMessage(message);
    });
  }
  try {
    check((await send({type: 'init', ...urls})).type === 'ready', 'Worker load');
    const reply = await send({type: 'detect', pixels: raw, W: 160, H: 160});
    check(reply.faces.length === 1 && reply.faces[0].score === faces[0].score, 'Worker detection');
    const {emb} = await send({type: 'embed', pixels: new Float32Array(112 * 112 * 3)});
    check(emb.length === 512 && emb.some(x => x !== 0), 'Worker embedding');
  } finally {
    worker.terminate();
  }
  return {faces: faces.length, confidence: faces[0].score, dimensions: embeddings[0].length};
}
