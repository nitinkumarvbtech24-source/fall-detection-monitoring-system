# Model Card: YOLOv8n-Pose for Edge Fall Detection

## Model Overview
- **Model Name**: YOLOv8n-Pose (Ultralytics)
- **Task**: Simultaneous Multi-Person Bounding Box Detection + 17-Keypoint 2D Pose Estimation
- **Architecture**: Single-stage CNN with decoupled detection and keypoint regression heads
- **Parameters**: 3,289,964 (~3.3M parameters)
- **FLOPs**: 9.2 GFLOPs @ 640x640 resolution
- **Pretrained Weights**: COCO Keypoints 2017 val mAP50-95: 50.4, mAP50: 80.1
- **Export Format**: ONNX (opset 17), TensorRT Engine (FP16 / INT8)

---

## Tensor Specifications

### Input Tensor
- **Name**: `images`
- **Data Type**: `Float32` (normalized to [0.0, 1.0])
- **Shape**: `[1, 3, 640, 640]` (Batch x Channels x Height x Width)
- **Color Format**: RGB

### Output Tensor
- **Name**: `output0`
- **Data Type**: `Float32`
- **Shape**: `[1, 56, 8400]` (Batch x Channels x Anchors)
- **Channel Layout (56 channels)**:
  - Channels 0-3: Bounding box coordinates $[x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}]$
  - Channel 4: Class confidence score ($P(\text{person})$)
  - Channels 5-55: 17 Keypoints $\times 3$ values per keypoint $[x_k, y_k, \text{conf}_k]$

---

## COCO 17 Keypoints Schema
| Index | Keypoint Name | Anatomical Region | Role in Fall Detection |
|-------|---------------|-------------------|------------------------|
| 0 | Nose | Head / Face | Head descent velocity, privacy blur |
| 1 | Left Eye | Head / Face | Privacy blur region estimation |
| 2 | Right Eye | Head / Face | Privacy blur region estimation |
| 3 | Left Ear | Head / Face | Privacy blur boundary |
| 4 | Right Ear | Head / Face | Privacy blur boundary |
| 5 | Left Shoulder | Upper Body / Torso | Torso orientation & tilt vector |
| 6 | Right Shoulder| Upper Body / Torso | Torso orientation & tilt vector |
| 7 | Left Elbow | Arm | Posture confirmation |
| 8 | Right Elbow | Arm | Posture confirmation |
| 9 | Left Wrist | Hand | Impact / bracing detection |
| 10 | Right Wrist | Hand | Impact / bracing detection |
| 11 | Left Hip | Lower Body / Pelvis| Torso bottom anchor, height baseline |
| 12 | Right Hip | Lower Body / Pelvis| Torso bottom anchor, height baseline |
| 13 | Left Knee | Leg | Standing vs kneeling/sitting |
| 14 | Right Knee | Leg | Standing vs kneeling/sitting |
| 15 | Left Ankle | Ground Contact | Ground reference plane |
| 16 | Right Ankle | Ground Contact | Ground reference plane |

---

## TensorRT Optimization Profile
- **Target Precision 1**: FP16 (Half Precision) - 0% accuracy loss, 2.8x speedup over FP32
- **Target Precision 2**: INT8 (8-bit Quantization) - Evaluated with Symmetric Linear Quantization and Entropy Calibrator v2, achieving 1.8x additional throughput with < 1.2% mAP deviation
- **Memory Footprint**: ~18 MB (FP16 Engine), ~9 MB (INT8 Engine)
- **Execution Target**: NVIDIA DeepStream `nvinfer` plugin
