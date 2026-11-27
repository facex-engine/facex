/**
 * FaceX Browser SDK — Face recognition in WebAssembly.
 * No server. No dependencies.
 */

export interface FaceXOptions {
  /** Detection input size (default: 160) */
  detSize?: number;
  /** Directory containing detect.wasm and facex.wasm. */
  wasmBaseUrl?: string;
  /** Cosine similarity threshold for match (default: 0.3) */
  threshold?: number;
  /** URL to detector weights (default: 'det_500m_int8.bin') */
  detWeightsUrl?: string;
  /** URL to embedder weights (default: 'edgeface_xs_fp32.bin') */
  embWeightsUrl?: string;
  /** Progress callback */
  onProgress?: (message: string) => void;
}

export interface Face {
  /** Bounding box */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Detection confidence 0-1 */
  score: number;
  /** 5 keypoints as flat array [lx,ly, rx,ry, nx,ny, lmx,lmy, rmx,rmy] */
  kps: number[];
}

export interface ProcessResult {
  /** Detected faces */
  faces: Face[];
  /** 512-dim embedding for each face */
  embeddings: Float32Array[];
  /** Total processing time in ms */
  ms: number;
}

export interface VerifyResult {
  /** Whether faces match */
  match: boolean;
  /** Cosine similarity from -1 to 1 */
  similarity: number;
  /** Detected faces */
  faces: Face[];
  /** Embedding of detected face (if any) */
  embedding?: Float32Array;
  /** Total time in ms */
  ms: number;
  /** True if no face detected */
  noFace?: boolean;
}

export interface CaptureResult {
  /** Reference embedding */
  embedding: Float32Array;
  /** Detected face */
  face: Face;
  /** Canvas with aligned 112x112 face */
  alignedCanvas: HTMLCanvasElement;
}

export declare class FaceXSDK {
  constructor(options?: FaceXOptions);

  /** Whether the SDK is loaded and ready */
  readonly ready: boolean;

  /** Load engines and weights. Call once before using. */
  load(): Promise<FaceXSDK>;
  close(): Promise<void>;

  /** Detect faces in video/image/canvas */
  detect(source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement): Face[];

  /** Compute 512-dim embedding from 112x112 face ImageData */
  embed(faceImageData: ImageData): Float32Array;

  /** Full pipeline: detect → align → embed */
  process(source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement): ProcessResult;

  /** Verify live source against reference embedding */
  verify(
    source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
    refEmbedding: Float32Array
  ): VerifyResult;

  /** Capture reference from current frame */
  captureReference(
    source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement
  ): CaptureResult | null;

  /** Cosine similarity between two 512-dim embeddings */
  cosSim(a: Float32Array, b: Float32Array): number;
}
