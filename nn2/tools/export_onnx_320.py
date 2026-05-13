"""Export YOLOv8n ONNX at 320x320 for fair comparison."""
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=320, simplify=True)
    import shutil
    shutil.move("yolov8n.onnx", "yolov8n_320.onnx")
    print("Exported yolov8n_320.onnx")
except ImportError:
    # Try onnx shape inference to resize
    import onnx
    from onnx import shape_inference
    print("ultralytics not available, trying onnx resize...")
    model = onnx.load("yolov8n.onnx")
    # Change input shape
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value == 640:
                dim.dim_value = 320
    onnx.save(model, "yolov8n_320.onnx")
    print("Saved yolov8n_320.onnx (input resized, intermediate shapes may be wrong)")
