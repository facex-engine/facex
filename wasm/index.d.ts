/**
 * facex-wasm — Face recognition in the browser.
 * Detect, align, embed, verify.
 * No server. No dependencies.
 */

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/** Configuration options for FaceXSDK */
export interface FaceXOptions {
  /** Detection input size in pixels (default: 160) */
  detSize?: number;
  /** Directory containing detect.wasm and facex.wasm. */
  wasmBaseUrl?: string;
  /** Cosine similarity threshold for a positive match (default: 0.3) */
  threshold?: number;
  /** URL to detector weights file (default: 'yunet_fp32.bin') */
  detWeightsUrl?: string;
  /** URL to embedder weights file (default: 'edgeface_xs_fp32.bin') */
  embWeightsUrl?: string;
  /** Progress callback invoked during loading */
  onProgress?: (message: string) => void;
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/** A detected face with bounding box, confidence, and 5 keypoints */
export interface Face {
  /** Left edge of bounding box */
  x1: number;
  /** Top edge of bounding box */
  y1: number;
  /** Right edge of bounding box */
  x2: number;
  /** Bottom edge of bounding box */
  y2: number;
  /** Detection confidence (0 to 1) */
  score: number;
  /**
   * 5 keypoints as a flat array of 10 values:
   * [left_eye_x, left_eye_y, right_eye_x, right_eye_y,
   *  nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y]
   */
  kps: number[];
}

/** Result from the full detect-align-embed pipeline */
export interface ProcessResult {
  /** Detected faces with bounding boxes and keypoints */
  faces: Face[];
  /** 512-dimensional embedding for each detected face */
  embeddings: Float32Array[];
  /** Total processing time in milliseconds */
  ms: number;
}

/** Result from a verification (1:1 match) operation */
export interface VerifyResult {
  /** Whether the face matches the reference embedding */
  match: boolean;
  /** Cosine similarity between detected and reference embeddings (0 to 1) */
  similarity: number;
  /** Detected faces */
  faces: Face[];
  /** Embedding of the first detected face, if any */
  embedding?: Float32Array;
  /** Total time in milliseconds */
  ms: number;
  /** True when no face was found in the source */
  noFace?: boolean;
}

/** Result from capturing a reference embedding */
export interface CaptureResult {
  /** 512-dimensional reference embedding */
  embedding: Float32Array;
  /** Detected face used for the embedding */
  face: Face;
  /** Canvas containing the aligned 112x112 face crop */
  alignedCanvas: HTMLCanvasElement;
}

// ---------------------------------------------------------------------------
// Liveness
// ---------------------------------------------------------------------------

/** Options for LivenessDetector */
export interface LivenessOptions {
  /** Number of frames to keep in history (default: 30) */
  historySize?: number;
  /** Minimum motion in pixels to consider alive (default: 2.0) */
  motionThreshold?: number;
  /** Eye aspect ratio threshold for blink detection (default: 0.22) */
  blinkThreshold?: number;
}

/** Result from a liveness check */
export interface LivenessResult {
  /** Whether the subject appears to be a live person */
  alive: boolean;
  /** Confidence score (0 to 1) */
  confidence: number;
  /** Human-readable status message */
  reason: string;
  /** Detailed metrics (present when enough frames collected) */
  details?: {
    motion: number;
    blinks: number;
    sizeVar: number;
  };
}

// ---------------------------------------------------------------------------
// FaceXSDK
// ---------------------------------------------------------------------------

/** Input source type accepted by SDK methods */
export type FaceSource = HTMLVideoElement | HTMLCanvasElement | HTMLImageElement;

/**
 * Unified FaceX browser SDK.
 *
 * Provides face detection, alignment, 512-dim embedding, verification,
 * and reference capture from video, image, or canvas elements.
 */
export declare class FaceXSDK {
  constructor(options?: FaceXOptions);

  /** Whether the SDK is loaded and ready to use */
  readonly ready: boolean;

  /**
   * Load WASM engines and model weights. Call once before any other method.
   * @returns The SDK instance for chaining.
   */
  load(): Promise<FaceXSDK>;
  close(): Promise<void>;

  /**
   * Detect faces in a video, image, or canvas element.
   * @param source - Media element to detect faces in.
   * @returns Array of detected faces with bounding boxes and keypoints.
   */
  detect(source: FaceSource): Face[];

  /**
   * Compute a 512-dimensional embedding from an aligned 112x112 face.
   * @param faceImageData - 112x112 RGBA ImageData of an aligned face.
   * @returns 512-dimensional Float32Array embedding.
   */
  embed(faceImageData: ImageData): Float32Array;

  /**
   * Full pipeline: detect faces, align each, compute embeddings.
   * @param source - Media element to process.
   */
  process(source: FaceSource): ProcessResult;

  /**
   * Verify a live source against a previously captured reference embedding.
   * @param source - Media element containing the live face.
   * @param refEmbedding - 512-dim reference embedding to compare against.
   */
  verify(source: FaceSource, refEmbedding: Float32Array): VerifyResult;

  /**
   * Capture a reference embedding from the current frame.
   * @param source - Media element to capture from.
   * @returns Capture result, or null if no face was detected.
   */
  captureReference(source: FaceSource): CaptureResult | null;

  /**
   * Compute cosine similarity between two 512-dimensional embeddings.
   * @returns Similarity score between -1 and 1.
   */
  cosSim(a: Float32Array, b: Float32Array): number;
}

// ---------------------------------------------------------------------------
// LivenessDetector
// ---------------------------------------------------------------------------

/**
 * Basic liveness detection using face keypoint analysis.
 *
 * Detects motion, blinks, and size variation across frames to distinguish
 * a live person from a static photo or screen replay.
 */
export declare class LivenessDetector {
  constructor(options?: LivenessOptions);

  /**
   * Feed a new face detection frame.
   * @param face - Detected face from FaceXSDK.detect(), or null if no face.
   * @returns Liveness assessment.
   */
  update(face: Face | null): LivenessResult;

  /** Reset all accumulated state */
  reset(): void;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/**
 * Check if the browser supports WebAssembly SIMD.
 * FaceX requires SIMD: Chrome 91+, Firefox 89+, Safari 16.4+.
 */
export declare function checkWasmSimd(): boolean;

/**
 * Check browser support for all FaceX requirements.
 * @returns Array of human-readable issue strings. Empty array = all good.
 */
export declare function checkBrowserSupport(): string[];

/**
 * Align a face to a canonical 112x112 crop using 5 keypoints.
 * @param srcCtx - Source canvas rendering context (full video frame).
 * @param srcW - Source width.
 * @param srcH - Source height.
 * @param kps - 5 keypoints as [[x,y], ...] in source coordinates.
 * @param dstCtx - Destination 112x112 canvas rendering context.
 * @returns 112x112 ImageData of the aligned face.
 */
export declare function alignFace(
  srcCtx: CanvasRenderingContext2D,
  srcW: number,
  srcH: number,
  kps: [number, number][],
  dstCtx: CanvasRenderingContext2D
): ImageData;

/**
 * Load model weights with IndexedDB caching.
 * First call fetches from network and caches; subsequent calls load from cache.
 * @param url - URL to fetch weights from.
 * @param key - Cache key (e.g. 'det_weights').
 * @param onProgress - Optional progress callback(loaded, total).
 */
export declare function loadWeightsCached(
  url: string,
  key: string,
  onProgress?: (loaded: number, total: number) => void
): Promise<Uint8Array>;
