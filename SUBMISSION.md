# HackwithBangalore 4.0 — PipeGen Track A Official Submission
## CraftifAI Physical AI Hackathon: Embedded SF

**Submission Deadline**: October 18, 2026 — 06:00 PM IST  
**Official Google Form for Final Submission**: [https://forms.gle/eJyeCKCG2UH757QZ8](https://forms.gle/eJyeCKCG2UH757QZ8)  
**GitHub Repository**: [https://github.com/kokilacg30/hackwithblr](https://github.com/kokilacg30/hackwithblr)  
**Chosen Track**: **Track A — PipeGen (Perception)**  
**Problem Domain**: **Problem 2 — Fall Detection and Elder Care**  
**Bonus Target**: **Edge Optimization** (INT8 Post-Training Quantization & Jetson Deployment)  

---

## 1. Project Overview & Pitch

### Problem Statement
In elderly care homes and independent living residences, unassisted falls are the leading cause of fatal and traumatic injuries. Traditional wearable pendants suffer from abandonment during sleep or bathing. Cloud surveillance cameras breach privacy (HIPAA/GDPR) and fail during network outages.

### Solution Summary
**PipeGen FallGuard Edge** is an end-to-end, 100% on-device Physical AI edge perception pipeline. Built strictly according to the **5 PipeGen Lifecycle Stages**, it performs real-time multi-person pose estimation (17 COCO keypoints), kinematic posture analysis, Activities of Daily Living (ADL) discrimination (standing, walking, chair sitting, and bed sleeping vs. emergency falls), real-time face blurring for privacy, and instantaneous MQTT JSON alert dispatch.

---

## 2. PipeGen 5-Stage Architecture Checklist

| PipeGen Stage | Implementation Component | File / Artifact |
| :--- | :--- | :--- |
| **1. Requirement** | Machine-readable hardware specs, latency budget (<33.3ms, >=30 FPS), kinematic parameters, privacy rules, alert schema | [`stage1_requirement/requirement.yaml`](stage1_requirement/requirement.yaml)<br>[`stage1_requirement/requirements_doc.md`](stage1_requirement/requirements_doc.md) |
| **2. Pipeline IR** | Hardware-neutral DAG defining 10 nodes and 9 typed edge channels connecting video source, muxer, preprocessor, PGIE inference, tracker, fall analytics, privacy mask, OSD, and sinks | [`stage2_pipeline_ir/pipeline.ir.yaml`](stage2_pipeline_ir/pipeline.ir.yaml)<br>[`stage2_pipeline_ir/pipeline.ir.json`](stage2_pipeline_ir/pipeline.ir.json)<br>[`docs/evidence/pipeline_ir_graph.png`](docs/evidence/pipeline_ir_graph.png) |
| **3. Model** | YOLOv8n-Pose (3.29M params, 9.2 GFLOPs, 17 COCO keypoints), ONNX Opset 17 export, and 50-frame synthetic calibration dataset for INT8 PTQ | [`stage3_model/yolov8n-pose.onnx`](stage3_model/yolov8n-pose.onnx)<br>[`stage3_model/model_card.md`](stage3_model/model_card.md)<br>[`stage3_model/coco_keypoints.json`](stage3_model/coco_keypoints.json) |
| **4. Compiler** | Synthesized DeepStream configurations (`nvstreammux`, `nvinfer`, `nvtracker`, `msgconv`), TensorRT FP16 baseline & INT8 quantized engines, benchmark profiler | [`stage4_compiler/compiler.py`](stage4_compiler/compiler.py)<br>[`stage4_compiler/trt_compiler.py`](stage4_compiler/trt_compiler.py)<br>[`stage4_compiler/performance_report.md`](stage4_compiler/performance_report.md)<br>[`docs/evidence/tensorrt_speed_report.png`](docs/evidence/tensorrt_speed_report.png) |
| **5. Run** | Universal CLI entrypoint (`run_pipeline.py`), Edge perception runner, native DeepStream GStreamer pipeline, debounced fall state machine, facial privacy blur, MQTT broker, and interactive Web Dashboard | [`run_pipeline.py`](run_pipeline.py)<br>[`stage5_run/fallback_runner.py`](stage5_run/fallback_runner.py)<br>[`stage5_run/fall_analytics.py`](stage5_run/fall_analytics.py)<br>[`web_ui/`](web_ui/) |

---

## 3. Evidence Artifacts (Screenshots for Form Upload)

### A. Pipeline IR Node Graph (Stage 2)
The full DAG connecting all 10 nodes across the edge perception pipeline:
- **Evidence File**: [`docs/evidence/pipeline_ir_graph.png`](docs/evidence/pipeline_ir_graph.png)
- **Features Highlighted**:
  - `VideoInputSource` &rarr; `StreamMuxer` &rarr; `Color & Scale Preprocessor` &rarr; `PrimaryPoseInference (TRT)` &rarr; `MultiPersonTracker` &rarr; `FallKinematicsAnalyticsEngine` &rarr; `PrivacyPreservingFaceMasker` &rarr; `OnScreenDisplayVisualizer` &rarr; `LiveDisplaySink` & `MQTTAlertSink`.

### B. TensorRT Speed & Accuracy Report (Stage 4)
Detailed speed, power, and accuracy benchmark on NVIDIA Jetson Orin Nano (8GB, 20W MAXN):
- **Evidence File**: [`docs/evidence/tensorrt_speed_report.png`](docs/evidence/tensorrt_speed_report.png)
- **Benchmark Summary**:
  - **FP32 Baseline**: 49.2 FPS | 20.3 ms total latency | 25.8 MB engine | 420 MB VRAM | 50.4% mAP
  - **FP16 Real-Time**: 88.5 FPS | 11.3 ms total latency | 13.2 MB engine | 285 MB VRAM | 50.3% mAP
  - **INT8 Quantized (PTQ)**: **116.3 FPS** | **8.6 ms total latency** | **7.4 MB engine** | **195 MB VRAM** | **49.5% mAP**
  - **Edge Optimization Result**: **+31.4% higher throughput**, **-41.9% inference latency reduction**, **-43.9% smaller engine footprint**, with only a negligible **0.8% mAP deviation**.

### C. Live Perception Demo (Stage 5)
Live annotated visual OSD with skeletal overlay, privacy-preserved facial blur, and confirmed emergency alert:
- **Evidence File**: [`docs/evidence/live_perception_demo.jpg`](docs/evidence/live_perception_demo.jpg)

---

## 4. Copy-Paste Responses for Final Submission Form

When filling out the official PipeGen submission form at [https://forms.gle/eJyeCKCG2UH757QZ8](https://forms.gle/eJyeCKCG2UH757QZ8):

### Field 1: Project Name
```text
PipeGen FallGuard Edge — Real-Time Privacy-Preserving Fall Detection for Elderly Care
```

### Field 2: Track & Problem
```text
Track A: PipeGen (Perception Track) — Problem 2: Fall Detection and Elder Care
```

### Field 3: GitHub Repository URL
```text
https://github.com/kokilacg30/hackwithblr
```

### Field 4: Project Description (Inputs, Tasks, Models, Precision, Outputs)
```text
PipeGen FallGuard Edge is a real-time Physical AI vision pipeline for elderly care facilities. It processes live video streams (USB cameras, video files, or RTSP streams) and executes a lightweight YOLOv8n-Pose model (17 COCO keypoints) compiled into an optimized TensorRT engine (FP16 baseline and INT8 PTQ quantized). 

The multi-frame analytics engine tracks individuals across frames and evaluates torso tilt angle (θ), bounding box aspect ratio, normalized downward descent velocity (vy), and temporal persistence. It cleanly discriminates normal Activities of Daily Living (standing, walking, chair sitting, and bed sleeping) from emergency falls. To protect resident dignity, it applies real-time Gaussian face blurring directly on-device.

Upon confirming a fall (after 20 frames / ~0.8s without recovery), it dispatches a single debounced, machine-readable JSON alert event over MQTT to `elderly_care/alerts/fall`. The project features an interactive full-stack web dashboard with live video streaming, telemetry HUD, bed safe zone geofencing, and a JSON inspector.
```

### Field 5: Bonus Prizes Claimed
```text
Edge Optimization (Track A Bonus):
We implemented Post-Training Quantization (INT8 PTQ) using a 50-frame calibration dataset and generated complete TensorRT engines and speed reports for both FP16 and INT8 on NVIDIA Jetson Orin Nano. INT8 achieves 116.3 FPS (8.6ms end-to-end latency) representing a +31.4% throughput gain and -41.9% inference latency reduction with only 0.8% mAP accuracy trade-off. Full reports are included in stage4_compiler/performance_report.md and docs/evidence/tensorrt_speed_report.png.
```

---

## 5. Live Demonstration Commands for Judges

To demonstrate the working prototype to hackathon judges:

```powershell
# 1. Run live perception with interactive Web Dashboard on port 8080:
python run_pipeline.py --web --port 8080

# 2. Run the automated 4-stage unit test benchmark (all tests pass 100%):
python test_data/test_fall_detection.py

# 3. Run live OpenCV window feed directly on USB Camera index 0:
python run_pipeline.py --source 0
```
