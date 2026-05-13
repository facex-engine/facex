// bench_camera_mac.swift
//
// macOS camera benchmark for FaceX. Pulls frames from the default camera
// via AVFoundation, downscales to 160×160 RGB, calls facex_detect, and
// prints per-second FPS / median latency / face count to stdout.
//
// Build:  see tools/build_bench_camera_mac.sh
// Usage:  ./facex-camera-bench [--frames N] [--width W] [--height H] [--no-detect]
//
// Permission: macOS will prompt the parent terminal for Camera access on
// first run. Grant it in System Settings ▸ Privacy & Security ▸ Camera.

import Foundation
import AVFoundation
import CoreVideo
import CoreImage

// MARK: - CLI args

struct Args {
    var maxFrames: Int = 0          // 0 = forever
    var width:  Int = 160
    var height: Int = 160
    var detectScoreThreshold: Float = 0.5
    var embedderWeights: String = "data/edgeface_xs_fp32.bin"
    var detectorWeights: String = "weights/yunet_fp32.bin"
    var skipDetect: Bool = false
    var summary: Bool = false
    var summaryLabel: String = "camera"
}

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let tok = it.next() {
        switch tok {
        case "--frames":  if let v = it.next(), let n = Int(v)   { a.maxFrames = n }
        case "--width":   if let v = it.next(), let n = Int(v)   { a.width  = n }
        case "--height":  if let v = it.next(), let n = Int(v)   { a.height = n }
        case "--score":   if let v = it.next(), let f = Float(v) { a.detectScoreThreshold = f }
        case "--embed":   if let v = it.next() { a.embedderWeights = v }
        case "--detect":  if let v = it.next() { a.detectorWeights = v }
        case "--no-detect": a.skipDetect = true
        case "--summary":   a.summary = true
        case "--summary-label": if let v = it.next() { a.summaryLabel = v; a.summary = true }
        case "--help", "-h":
            print("""
                facex-camera-bench [options]
                  --frames N           stop after N frames (default: run forever, Ctrl-C to stop)
                  --width  W           downscale width  (default 160)
                  --height H           downscale height (default 160)
                  --score  F           detector score threshold (default 0.5)
                  --embed  PATH        embedder weights .bin (default data/edgeface_xs_fp32.bin)
                  --detect PATH        detector weights .bin (default weights/yunet_fp32.bin)
                  --no-detect          skip the engine call (camera-only baseline)
                  --summary            on exit, print a one-line CSV summary suitable for
                                       merging into the unified bench table (see
                                       scripts/bench_all.sh / docs/benchmarking.md).
                  --summary-label STR  same as --summary, but tag the row with this label
                                       instead of the default "camera".
                """)
            exit(0)
        default: break
        }
    }
    return a
}

let args = parseArgs()

// MARK: - FaceX engine init

guard FileManager.default.fileExists(atPath: args.embedderWeights) else {
    fputs("error: embedder weights not found at \(args.embedderWeights)\n", stderr)
    fputs("       run `bash download_weights.sh` first\n", stderr)
    exit(1)
}
let detectorAvailable = FileManager.default.fileExists(atPath: args.detectorWeights)
if !detectorAvailable {
    fputs("warn: detector weights not at \(args.detectorWeights) — running embed-only\n", stderr)
}

let engine: OpaquePointer? = args.embedderWeights.withCString { ePtr in
    detectorAvailable
        ? args.detectorWeights.withCString { dPtr in
            facex_init(ePtr, dPtr, nil)
          }
        : facex_init(ePtr, nil, nil)
}
guard let fx = engine else {
    fputs("error: facex_init failed\n", stderr); exit(1)
}
facex_set_score_threshold(fx, args.detectScoreThreshold)

print("FaceX \(String(cString: facex_version())) — Mac camera benchmark")
print("input: \(args.width)x\(args.height)  detector: \(detectorAvailable ? "on" : "off")")

// MARK: - Capture session

setbuf(stdout, nil) // unbuffered so updates show in non-TTY runs

// Camera permission: terminal apps need TCC consent. Request it and block
// until the OS dialog is answered (or fail explicitly if denied).
let status = AVCaptureDevice.authorizationStatus(for: .video)
switch status {
case .authorized:
    break
case .notDetermined:
    let sema = DispatchSemaphore(value: 0)
    var granted = false
    AVCaptureDevice.requestAccess(for: .video) { ok in granted = ok; sema.signal() }
    sema.wait()
    if !granted {
        fputs("error: camera access denied — grant in System Settings ▸ Privacy ▸ Camera\n", stderr)
        exit(2)
    }
case .denied, .restricted:
    fputs("error: camera access denied — grant in System Settings ▸ Privacy ▸ Camera\n", stderr)
    exit(2)
@unknown default:
    fputs("error: unknown camera authorization status\n", stderr); exit(2)
}

let session = AVCaptureSession()
session.sessionPreset = .vga640x480

guard let cam = AVCaptureDevice.default(for: .video) else {
    fputs("error: no camera found\n", stderr); exit(1)
}
do {
    let input = try AVCaptureDeviceInput(device: cam)
    if session.canAddInput(input) { session.addInput(input) }
} catch {
    fputs("error: \(error)\n", stderr); exit(1)
}

let output = AVCaptureVideoDataOutput()
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                        kCVPixelFormatType_32BGRA]
output.alwaysDiscardsLateVideoFrames = true

let queue = DispatchQueue(label: "facex.cam.queue")
let processed = ProcessedCounter()

class ProcessedCounter {
    var frameIndex: Int = 0
    var lastReportTime: TimeInterval = 0
    var startTime: TimeInterval = 0
    var samples: [Double] = []
    var allSamples: [Double] = []   // never cleared — for the final summary
    var firstFaceFrame: Int = -1
    var lastBoxes: [(Float, Float, Float, Float, Float)] = []
}

class FrameSink: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let args: Args
    let fx: OpaquePointer
    let counter: ProcessedCounter
    let context = CIContext(options: nil)

    /* Reusable scratch buffers — written once per frame, never freed. */
    var rgbScratch: [UInt8]

    init(args: Args, fx: OpaquePointer, counter: ProcessedCounter) {
        self.args = args
        self.fx = fx
        self.counter = counter
        self.rgbScratch = [UInt8](repeating: 0, count: args.width * args.height * 3)
        super.init()
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pix = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Downscale CVPixelBuffer (BGRA) → args.width×args.height RGB uint8.
        let ci = CIImage(cvPixelBuffer: pix)
        let srcW = CGFloat(CVPixelBufferGetWidth(pix))
        let srcH = CGFloat(CVPixelBufferGetHeight(pix))
        let scale = min(CGFloat(args.width) / srcW, CGFloat(args.height) / srcH)
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let cropRect = CGRect(x: 0, y: 0, width: args.width, height: args.height)

        let bytesPerRow = args.width * 4
        var bgra = [UInt8](repeating: 0, count: args.width * args.height * 4)
        bgra.withUnsafeMutableBytes { ptr in
            let cs = CGColorSpaceCreateDeviceRGB()
            context.render(scaled,
                           toBitmap: ptr.baseAddress!,
                           rowBytes: bytesPerRow,
                           bounds: cropRect,
                           format: .BGRA8,
                           colorSpace: cs)
        }
        // Pack RGB
        for i in 0..<(args.width * args.height) {
            let b = bgra[i * 4 + 0]
            let g = bgra[i * 4 + 1]
            let r = bgra[i * 4 + 2]
            rgbScratch[i * 3 + 0] = r
            rgbScratch[i * 3 + 1] = g
            rgbScratch[i * 3 + 2] = b
        }

        var nfaces: Int32 = 0
        let t0 = now_ms()
        if !args.skipDetect {
            var results = [FaceXResult](repeating: FaceXResult(), count: 8)
            let n = results.withUnsafeMutableBufferPointer { rp -> Int32 in
                rgbScratch.withUnsafeBufferPointer { rgb -> Int32 in
                    facex_detect(fx,
                                 rgb.baseAddress,
                                 Int32(args.width), Int32(args.height),
                                 rp.baseAddress, 8)
                }
            }
            nfaces = max(n, 0)
            counter.lastBoxes.removeAll(keepingCapacity: true)
            for i in 0..<Int(nfaces) {
                let r = results[i]
                counter.lastBoxes.append((r.x1, r.y1, r.x2, r.y2, r.score))
            }
        }
        let dt = now_ms() - t0

        DispatchQueue.main.async {
            self.counter.frameIndex += 1
            self.counter.samples.append(dt)
            self.counter.allSamples.append(dt)
            if self.counter.firstFaceFrame < 0 && nfaces > 0 {
                self.counter.firstFaceFrame = self.counter.frameIndex
            }
            let now = ProcessInfo.processInfo.systemUptime
            if now - self.counter.lastReportTime >= 1.0 {
                let s = self.counter.samples
                let med = median(s)
                let p99 = percentile(s, 0.99)
                let fps = Double(s.count) / (now - self.counter.lastReportTime)
                let label = self.args.skipDetect ? "camera" : "detect+embed"
                print(String(format: "[t=%.1fs] frame %d  %.1f fps  %@ med=%.1f ms  p99=%.1f ms  faces=%d",
                             now, self.counter.frameIndex, fps, label, med, p99, Int(nfaces)))
                if let first = self.counter.lastBoxes.first {
                    print(String(format: "    bbox: [%.0f,%.0f → %.0f,%.0f]  score=%.2f",
                                 first.0, first.1, first.2, first.3, first.4))
                }
                self.counter.samples.removeAll(keepingCapacity: true)
                self.counter.lastReportTime = now
            }
            if self.args.maxFrames > 0 && self.counter.frameIndex >= self.args.maxFrames {
                print("done.")
                if self.args.summary {
                    self.emitSummary()
                }
                facex_free(self.fx)
                exit(0)
            }
        }
    }
}

func now_ms() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) * 1000.0 + Double(ts.tv_nsec) / 1e6
}

extension FrameSink {
    /// Emit a one-line CSV row that joins the unified bench table.
    /// Schema (must match scripts/bench_all.sh expectations):
    ///   label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face
    func emitSummary() {
        let s = counter.allSamples.sorted()
        guard !s.isEmpty else { return }
        let n = s.count
        let minv = s.first!
        let med  = s[n / 2]
        let p95  = s[min(n - 1, Int(Double(n) * 0.95))]
        let p99  = s[min(n - 1, Int(Double(n) * 0.99))]
        let mean = s.reduce(0, +) / Double(n)
        let stage = args.skipDetect ? "camera" : "e2e"
        let face  = counter.firstFaceFrame >= 0 ? 1 : 0
        let label = args.summaryLabel
        // Backend reporting from the camera tool side is "camera" — the
        // engine-side flags live with facex-bench. We document the column
        // in docs/benchmarking.md.
        let compiled = "camera"
        let active   = "camera"
        // Header to stderr so a downstream CSV concat can drop one line.
        FileHandle.standardError.write(
            "label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face\n".data(using: .utf8)!
        )
        let row = String(format: "\"%@\",\"%@\",\"%@\",%@,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                         label, compiled, active, stage, n,
                         minv, med, mean, p95, p99, face)
        FileHandle.standardOutput.write(row.data(using: .utf8)!)
    }
}

func median(_ xs: [Double]) -> Double {
    if xs.isEmpty { return 0 }
    let s = xs.sorted()
    return s[s.count / 2]
}

func percentile(_ xs: [Double], _ p: Double) -> Double {
    if xs.isEmpty { return 0 }
    let s = xs.sorted()
    let idx = min(s.count - 1, Int(Double(s.count) * p))
    return s[idx]
}

let sink = FrameSink(args: args, fx: fx, counter: processed)
output.setSampleBufferDelegate(sink, queue: queue)
if session.canAddOutput(output) { session.addOutput(output) }

// MARK: - Run

session.startRunning()
processed.lastReportTime = ProcessInfo.processInfo.systemUptime
print("capturing… (Ctrl-C to stop)")
RunLoop.main.run()
