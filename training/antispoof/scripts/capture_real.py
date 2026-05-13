"""Capture real webcam frames of the user to use as REAL anti-spoof samples.

Crops face with side = face_side * 1.9 (matching browser antispoof crop),
resizes to 112×112, saves as raw bin + count.

Run, look at camera ~30 sec, move slightly, change angles/expressions.
Press Q to stop early.
"""

import time
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("D:/apps/facex/training/data/antispoof/user_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "frames.bin"
META = OUT_DIR / "count.txt"

SIZE = 112
CROP_MULT = 1.9      # MUST match demo_mesh.html predictSpoof crop ratio
MAX_FRAMES = 400
TARGET_DURATION = 30.0  # seconds

YUNET = "D:/apps/facex/wasm/yunet_2023mar.onnx"


def main():
    print(f"loading YuNet detector from {YUNET}")
    det = cv2.FaceDetectorYN.create(YUNET, "", (640, 640),
                                      score_threshold=0.6, nms_threshold=0.3, top_k=5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open camera 0. Close other apps using webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("camera open. look at the camera. press Q to stop. recording 30 seconds...")

    frames = []
    t0 = time.time(); last_save = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        live = frame.copy()
        if faces is not None and len(faces) > 0:
            # pick largest
            best = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = best[:4]
            cx, cy = x + fw / 2, y + fh / 2
            side = max(fw, fh) * CROP_MULT
            sx = int(cx - side / 2); sy = int(cy - side / 2)
            ex = int(sx + side); ey = int(sy + side)
            # clip and pad if needed
            crop = frame[max(0, sy):min(h, ey), max(0, sx):min(w, ex)]
            # pad to square
            pad_t = max(0, -sy); pad_l = max(0, -sx)
            pad_b = max(0, ey - h); pad_r = max(0, ex - w)
            if pad_t or pad_l or pad_b or pad_r:
                crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                           cv2.BORDER_REPLICATE)
            # OpenCV is BGR — convert to RGB for storage consistency
            crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # throttle: 1 save per 75 ms
            now = time.time()
            if now - last_save >= 0.075:
                frames.append(crop_rgb)
                last_save = now
            # draw rect on preview
            cv2.rectangle(live, (max(0, sx), max(0, sy)),
                          (min(w, ex), min(h, ey)), (0, 255, 0), 2)
        elapsed = time.time() - t0
        cv2.putText(live, f"saved {len(frames)}  t {elapsed:.1f}s   Q=stop",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("capture (look here)", live)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        if len(frames) >= MAX_FRAMES: break
        if elapsed >= TARGET_DURATION and len(frames) >= 200: break

    cap.release(); cv2.destroyAllWindows()
    if not frames:
        print("no frames captured. exiting.")
        return
    arr = np.stack(frames, axis=0).astype(np.uint8)
    arr.tofile(RAW)
    META.write_text(str(len(frames)))
    print(f"saved {len(frames)} frames -> {RAW}  ({arr.nbytes/1024:.0f} KB)")


if __name__ == "__main__":
    main()
