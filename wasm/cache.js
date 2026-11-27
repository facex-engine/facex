/*
 * cache.js — IndexedDB weight caching for FaceX WASM.
 * First load: fetch from network, store in IndexedDB.
 * Subsequent loads: read from IndexedDB (instant).
 */

const CACHE_DB = 'facex-cache';
const CACHE_STORE = 'weights';
const CACHE_VERSION = 1;

function openCacheDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(CACHE_DB, CACHE_VERSION);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(CACHE_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getCached(key) {
  const db = await openCacheDB();
  try { return await new Promise((resolve) => {
    const tx = db.transaction(CACHE_STORE, 'readonly');
    const req = tx.objectStore(CACHE_STORE).get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => resolve(null);
  }); } finally { db.close(); }
}

async function putCached(key, data) {
  const db = await openCacheDB();
  return new Promise((resolve) => {
    const tx = db.transaction(CACHE_STORE, 'readwrite');
    tx.objectStore(CACHE_STORE).put(data, key);
    tx.oncomplete = () => resolve();
    tx.onabort = tx.onerror = () => resolve();
  }); } finally { db.close(); }
}

/**
 * Load weights with IndexedDB caching.
 * @param {string} url - URL to fetch weights from
 * @param {string} key - Cache key (e.g. 'det_weights' or 'emb_weights')
 * @param {function} onProgress - callback(loaded, total) for progress
 * @returns {Promise<Uint8Array>}
 */
async function loadWeightsCached(url, key, onProgress) {
  // Try cache first
  const absolute = new URL(url, globalThis.document?.baseURI || globalThis.location.href).href;
  key = [key, absolute];
  const cached = await getCached(key).catch(() => null);
  if (cached) {
    if (onProgress) onProgress(cached.byteLength, cached.byteLength);
    return new Uint8Array(cached);
  }

  // Fetch from network with progress
  const response = await fetch(absolute);
  if (!response.ok) throw new Error('Weight request failed with status ' + response.status);
  if (!onProgress || !response.body) {
    const data = await response.arrayBuffer();
    await putCached(key, data).catch(() => {});
    onProgress?.(data.byteLength, data.byteLength);
    return new Uint8Array(data);
  }
  const total = parseInt(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (onProgress) onProgress(loaded, total);
  }

  // Combine chunks
  const data = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.length;
  }

  // Store in cache
  await putCached(key, data.buffer).catch(() => {});

  return data;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {loadWeightsCached};
