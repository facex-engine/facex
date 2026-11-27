/*
 * simd_check.js — Check if browser supports WebAssembly SIMD.
 * Returns true if WASM SIMD is available, false otherwise.
 */

function checkWasmSimd() {
  try {
    // Minimal WASM module that uses v128 SIMD type
    const bytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, // magic
      0x01, 0x00, 0x00, 0x00, // version
      0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type section: () -> v128
      0x03, 0x02, 0x01, 0x00, // function section
      0x0a, 0x16, 0x01, 0x14, 0x00, 0xfd, 0x0c, // code section: v128.const
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x0b // end
    ]);
    return WebAssembly.validate(bytes);
  } catch (e) {
    return false;
  }
}

function checkBrowserSupport() {
  const issues = [];

  if (typeof WebAssembly === 'undefined')
    issues.push('WebAssembly not supported');
  else if (!checkWasmSimd())
    issues.push('WebAssembly SIMD not supported (Chrome 91+, Firefox 89+, Safari 16.4+ required)');

  if (!globalThis.navigator?.mediaDevices?.getUserMedia)
    issues.push('Camera API not available (HTTPS required)');

  return issues;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {checkWasmSimd, checkBrowserSupport};
