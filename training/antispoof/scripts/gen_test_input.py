"""Generate 80×80 BGR test input + ONNX reference prediction."""

import sys, struct
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "D:/apps/facex/training/Silent-Face-Anti-Spoofing")
from src.generate_patches import CropImage

IMG = "D:/apps/facex/training/Silent-Face-Anti-Spoofing/images/sample/image_T1.jpg"

img = cv2.imread(IMG)
print(f"image: {img.shape}")
# bbox of T1 (from YuNet earlier)
bbox = [119, 113, 169, 239]

cropper = CropImage()
for scale, out in [(2.7, "test_27.bin"), (4.0, "test_40.bin")]:
    crop = cropper.crop(org_img=img, bbox=bbox, scale=scale, out_w=80, out_h=80)
    Path("D:/apps/facex/training/data/antispoof").mkdir(parents=True, exist_ok=True)
    open(f"D:/apps/facex/training/data/antispoof/{out}", "wb").write(crop.tobytes())
    print(f"  wrote {out}: {crop.shape} dtype {crop.dtype}")

# ONNX baseline predictions
s27 = ort.InferenceSession("D:/apps/facex/wasm/minifasnet_v2_27.onnx", providers=["CPUExecutionProvider"])
s40 = ort.InferenceSession("D:/apps/facex/wasm/minifasnet_v1se_40.onnx", providers=["CPUExecutionProvider"])

c27 = cv2.imread("D:/apps/facex/training/data/antispoof/test_27.bin", cv2.IMREAD_COLOR) if False else None
# Re-make for ONNX (same crop)
for scale, name in [(2.7, "v2"), (4.0, "v1se")]:
    c = cropper.crop(org_img=img, bbox=bbox, scale=scale, out_w=80, out_h=80)
    # ToTensor with /255 commented out → raw [0,255], BGR, CHW
    x = c.astype(np.float32).transpose(2, 0, 1)[None]
    sess = s27 if name == "v2" else s40
    p = sess.run(None, {"input": x})[0][0]
    print(f"ONNX {name}: probs={p}  P(live)={p[1]:.4f}")
