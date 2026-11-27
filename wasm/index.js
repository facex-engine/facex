/* Named exports work with CommonJS and ESM-aware bundlers. */
exports.FaceXSDK = require('./facex-sdk.js').FaceXSDK;
exports.LivenessDetector = require('./liveness.js').LivenessDetector;
exports.checkWasmSimd = require('./simd_check.js').checkWasmSimd;
exports.checkBrowserSupport = require('./simd_check.js').checkBrowserSupport;
exports.alignFace = require('./align.js').alignFace;
exports.loadWeightsCached = require('./cache.js').loadWeightsCached;
