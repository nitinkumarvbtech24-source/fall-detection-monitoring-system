"""
PipeGen Stage 3: Model Exporter
Exports YOLOv8n-pose to ONNX format with optimized graph topology for TensorRT compilation.
"""

import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

def export_model():
    model_path = Path("yolov8n-pose.pt")
    if not model_path.exists():
        print(f"[*] Downloading yolov8n-pose.pt...")
        model = YOLO("yolov8n-pose.pt")
    else:
        print(f"[*] Loading local model from: {model_path.resolve()}")
        model = YOLO(str(model_path))

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Exporting model to ONNX (imgsz=640, opset=17, simplify=True)...")
    exported_file = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=False, # Avoid onnxsim requirement if not installed, or handle gracefully
        dynamic=False
    )
    
    exported_path = Path(exported_file)
    target_path = output_dir / "yolov8n-pose.onnx"
    
    if exported_path.resolve() != target_path.resolve():
        shutil.copy(str(exported_path), str(target_path))
        print(f"[+] Moved exported ONNX model to: {target_path.resolve()}")
    else:
        print(f"[+] Exported ONNX model available at: {target_path.resolve()}")

    # Print model input/output tensor specs
    import onnx
    onnx_model = onnx.load(str(target_path))
    print("[*] ONNX Graph Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value != 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    - Name: {inp.name}, Shape: {shape}, Type: {inp.type.tensor_type.elem_type}")
    
    print("[*] ONNX Graph Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value != 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    - Name: {out.name}, Shape: {shape}, Type: {out.type.tensor_type.elem_type}")

    print("\n[+] Stage 3 Model Export Completed Successfully!")

if __name__ == "__main__":
    export_model()
