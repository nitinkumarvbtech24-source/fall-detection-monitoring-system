# PipeGen FallGuard Edge: Real-Time Privacy-Preserving Fall Detection for Elderly Care

### **Physical AI Hackathon: Embedded SF — HackwithBangalore 4.0**
- **Track**: **Track A — PipeGen (Perception Track)**
- **Problem Statement**: **Problem 2 — Fall Detection and Elder Care**
- **Bonus Prize Target**: **Edge Optimization** (INT8 PTQ Quantization on NVIDIA Jetson)
- **Official Submission Document**: [SUBMISSION.md](SUBMISSION.md)
- **Target Hardware**: NVIDIA Jetson Orin Nano / AGX Orin / NVIDIA RTX GPUs
- **Core Frameworks**: CraftifAI PipeGen, NVIDIA DeepStream SDK 6.x/7.x, TensorRT 10.x, GStreamer, Paho-MQTT, FastAPI

---

## 1. Executive Summary & Problem Statement

In elderly care facilities, independent senior living communities, and private homes, unassisted falls are the leading cause of fatal and severe trauma injuries. Wearable fall detectors suffer from high non-compliance, dead batteries, or abandonment during sleep or bathing. Furthermore, cloud-streaming surveillance cameras introduce severe privacy violations (HIPAA/GDPR non-compliance) and latency bottlenecks.

**PipeGen FallGuard Edge** is a complete, production-grade **Edge AI Vision Perception Pipeline** designed to solve this crisis. Operating **100% on-device**, the system:
1. Detects and tracks multiple individuals across consecutive video frames with persistent person IDs.
2. Extracts 17-point skeletal pose keypoints in real time.
3. Analyzes multi-frame kinematics (torso vertical tilt angle $\theta_{\text{torso}}$, bounding box aspect ratio $w/h$, normalized downward descent velocity $v_y$, and temporal posture persistence).
4. Distinguishes genuine emergency falls from normal Activities of Daily Living (ADLs) such as walking, sitting in chairs, bending over, or controlled intentional lying down.
5. Employs a state machine (`NORMAL` $\rightarrow$ `POTENTIAL FALL` $\rightarrow$ `FALL CONFIRMED`) with temporal hysteresis, upright recovery detection, and debouncing cooldowns to eliminate duplicate alert spam.
6. Enforces **Privacy-by-Design** by dynamically pixelating and Gaussian-blurring facial regions directly on edge frames before display or rendering.
7. Dispatches real-time, actionable machine-readable JSON alerts over **MQTT** to the topic `elderly_care/alerts/fall`.

---

## 2. PipeGen 5-Stage Lifecycle Architecture

This project is built around the **PipeGen** lifecycle stages:

```
+-----------------------------------------------------------------------------------------+
|                                PIPEGEN PIPELINE WORKFLOW                                |
+-----------------------------------------------------------------------------------------+
| [STAGE 1: REQUIREMENT]                                                                  |
|   - Formal specs: Target latency < 33.3ms (>=30 FPS), Jetson Orin Nano / RTX GPU       |
|   - 17-point COCO pose kinematics, ADL discrimination criteria, debounced MQTT events    |
|   - Files: stage1_requirement/requirement.yaml, stage1_requirement/requirements_doc.md  |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| [STAGE 2: PIPELINE IR]                                                                  |
|   - Hardware-neutral Intermediate Representation (IR) defining nodes, ports, and edges  |
|   - DAG: Source -> StreamMux -> Preproc -> PGIE (TRT) -> Tracker -> FallAnalytics       |
|          -> PrivacyFilter -> OSD -> VideoSink & MQTTAlertSink                           |
|   - Files: stage2_pipeline_ir/pipeline.ir.yaml, pipeline.ir.json, pipeline_graph.md     |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| [STAGE 3: MODEL]                                                                        |
|   - Lightweight YOLOv8n-Pose architecture (3.29M params, 9.2 GFLOPs, 17 COCO keypoints)|
|   - ONNX export (opset 17, static/dynamic tensor profiles)                              |
|   - Synthetic calibration dataset generation for INT8 Post-Training Quantization (PTQ)  |
|   - Files: stage3_model/export_onnx.py, model_card.md, coco_keypoints.json              |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| [STAGE 4: COMPILER]                                                                     |
|   - Synthesizes PipeGen IR into DeepStream configurations & TensorRT compilation targets|
|   - TensorRT FP16 (baseline) and INT8 (quantized) engine compilation and profiling      |
|   - Detailed speed, latency, VRAM, and mAP accuracy trade-off report                    |
|   - Files: stage4_compiler/compiler.py, trt_compiler.py, performance_report.md          |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
| [STAGE 5: RUN]                                                                          |
|   - Live execution on USB Camera (device 0), Video File (.mp4), or RTSP IP Camera       |
|   - Multi-person tracking (NvDCF/ByteTrack) with debounced fall state machine            |
|   - Edge Privacy Filter (dynamic facial blur/pixelation) & Visual OSD overlay           |
|   - MQTT Alert Dispatcher (elderly_care/alerts/fall) & Interactive Live Web Dashboard  |
|   - Files: run_pipeline.py, stage5_run/, test_data/test_fall_detection.py, web_ui/      |
+-----------------------------------------------------------------------------------------+
```

### 2.1 Pipeline IR Node Graph (Hardware-Neutral DAG)
![Pipeline IR Graph](docs/evidence/pipeline_ir_graph.png)

---

## 3. Fall Detection Kinematic Decision Logic

### 3.1 Mathematical Formulation

1. **Torso Vertical Angle ($\theta_{\text{torso}}$)**:
   Computed between the midpoint of the shoulders $M_S = \frac{K_5 + K_6}{2}$ and the midpoint of the hips $M_H = \frac{K_{11} + K_{12}}{2}$:
   $$\Delta x = |M_{S,x} - M_{H,x}|, \quad \Delta y = |M_{S,y} - M_{H,y}|$$
   $$\theta_{\text{torso}} = \arctan2(\Delta y, \Delta x) \times \frac{180}{\pi}$$
   - **Upright (Standing / Walking)**: $\theta_{\text{torso}} \in [62^\circ, 90^\circ]$
   - **Seated (Chair / Bedside)**: $\theta_{\text{torso}} \in [50^\circ, 80^\circ]$
   - **Horizontal (Fallen)**: $\theta_{\text{torso}} \le 38^\circ$

2. **Bounding Box Aspect Ratio ($AR$)**:
   $$AR = \frac{\text{Width}}{\text{Height}}$$
   - **Upright**: $AR \in [0.25, 0.75]$
   - **Seated**: $AR \in [0.70, 1.05]$
   - **Fallen**: $AR \ge 1.15$ (typically $1.5 - 3.5$)

3. **Normalized Downward Centroid Velocity ($v_y$)**:
   Computed across a temporal sliding window $\Delta t$:
   $$v_y = \frac{y_{\text{centroid}}(t) - y_{\text{centroid}}(t - \Delta t)}{H_{\text{person}} \cdot \Delta t}$$
   - **Emergency Fall**: Sudden downward spike ($v_y > 0.15 \text{ s}^{-1}$)
   - **Controlled Lying Down / Resting**: Smooth, gradual descent ($v_y < 0.08 \text{ s}^{-1}$)

### 3.2 State Machine Transitions

```mermaid
stateDiagram-v2
    [*] --> NORMAL
    NORMAL --> RESTING_SLEEPING : Safe Bed Zone OR Controlled Gradual Recline
    RESTING_SLEEPING --> NORMAL : Waking Up & Standing Up (Torso > 62°)
    NORMAL --> POTENTIAL_FALL : Horizontal Posture Outside Bed OR Downward Velocity Spike
    POTENTIAL_FALL --> RESTING_SLEEPING : Settles into Safe Rest Zone / Controlled Recline
    POTENTIAL_FALL --> NORMAL : Quick Posture Recovery (<4 frames)
    POTENTIAL_FALL --> FALL_CONFIRMED : Posture Persists >= 20 frames (~0.8s)
    FALL_CONFIRMED --> FALL_CONFIRMED : Maintain state (Debounced: 1 Alert Only)
    FALL_CONFIRMED --> NORMAL : Person Stands Up (Torso > 62° for 15 frames)
```

- **Intentional Sleeping (`RESTING / SLEEPING`)**: When an individual intentionally lies down in bed or reclines on a sofa, the system enters the calm `RESTING / SLEEPING` state (Sky Blue badge). **Zero emergency alarms are emitted**.
- **Debouncing & Cooldown**: When an accidental fall on the floor is confirmed, **exactly ONE** actionable JSON event is published to MQTT. The system remains in `FALL_CONFIRMED` without duplicate alert spamming until the person recovers or the operator resets.
- **Upright Recovery**: If the individual stands back up, the system automatically detects upright kinematics and smoothly transitions back to `NORMAL`.

### 3.3 Intentional Sleeping vs. Accidental Fall Discrimination
To ensure high caregiver trust and zero false alarms during sleep:
1. **Bed & Couch Safe Rest Zones (Geofencing)**: Spatial ROIs can be designated for beds and sofas. Any horizontal posture within these boundaries is immediately recognized as safe resting.
2. **Kinematic Velocity & Angular Collapse Analysis**:
   - *Accidental Falls*: Exhibit rapid downward velocity spikes ($v_y > 0.08 \text{ s}^{-1}$) and abrupt collapse ($\Delta \theta > 30^\circ$ within 20 frames).
   - *Controlled Lying Down*: Characterized by slow, controlled descent ($v_y \approx 0$) onto an elevated mattress.
3. **Bed Elevation Ratio**: Verifies that the recumbent centroid is elevated above floor level ($y_{\text{centroid}} < 0.80 \times H_{\text{frame}}$).

---

## 4. Privacy-by-Design Architecture

To respect patient dignity and comply with healthcare data privacy laws:
1. **Edge Isolation**: Zero raw video feeds or identifiable frames are sent to external cloud servers. All neural inference is 100% on-device.
2. **Dynamic Facial Obfuscation**: The system uses facial keypoints (Nose 0, Eyes 1-2, Ears 3-4) to derive an adaptive facial bounding region, applying high-strength Gaussian blur and mosaic pixelation before rendering or display.
3. **Data Minimization**: In steady state, video frames are never saved to disk. Only machine-readable, lightweight JSON event logs are persisted.

---

## 5. TensorRT & PipeGen Performance Report

![TensorRT Speed Report](docs/evidence/tensorrt_speed_report.png)

Benchmark results compiled from Stage 4 across precision targets:

### 5.1 Jetson Orin Nano (8GB, 20W MAXN) — Production Edge Target
| Metric | FP32 Baseline | FP16 (Real-Time Target) | INT8 (Quantized PTQ) | Trade-Off Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Inference Latency** | 14.8 ms | **6.2 ms** | **3.6 ms** | **-41.9% latency reduction in INT8** |
| **End-to-End Latency** | 20.3 ms | **11.3 ms** | **8.6 ms** | Comfortably &lt; 33.3ms budget |
| **Throughput (FPS)** | 49.2 FPS | **88.5 FPS** | **116.3 FPS** | **+31.4% FPS gain** |
| **Engine File Size** | 25.8 MB | **13.2 MB** | **7.4 MB** | -43.9% disk space |
| **GPU VRAM Usage** | 420 MB | **285 MB** | **195 MB** | -31.6% memory savings |
| **mAP@0.5:0.95 (Pose)** | 50.4% | **50.3%** | **49.5%** | **Negligible 0.8% accuracy delta** |

### 5.2 Multi-Platform Scalability
- **Jetson Orin Nano (8GB)**: 88.5 FPS (FP16), 116.3 FPS (INT8) &bull; Supports up to 2 concurrent 1080p cameras.
- **Jetson AGX Orin (64GB)**: 208.3 FPS (FP16), 294.1 FPS (INT8) &bull; Supports up to 8 concurrent 1080p cameras.
- **NVIDIA RTX 4090**: 476.2 FPS (FP16), 625.0 FPS (INT8) &bull; Supports up to 20 concurrent 1080p cameras.

---

## 6. Actionable Output Formats

### 6.1 Structured JSON Alert Format
Published immediately to MQTT topic `elderly_care/alerts/fall`:
```json
{
  "event": "fall_detected",
  "person_id": 1,
  "confidence": 0.94,
  "timestamp": "2026-10-17T22:27:08",
  "status": "FALL_CONFIRMED",
  "location": "Room 101 - Primary Care Area",
  "metrics": {
    "torso_angle_deg": 14.2,
    "aspect_ratio": 2.76,
    "descent_velocity": 0.28,
    "fallen_duration_sec": 1.15,
    "persistence_frames": 20
  },
  "privacy_applied": true
}
```

### 6.2 MQTT Topics
- **Alert Topic**: `elderly_care/alerts/fall` (QoS 1)
- **Telemetry Topic**: `elderly_care/telemetry/pipeline` (QoS 0)
- **Heartbeat Topic**: `elderly_care/heartbeat`

---

## 7. Quickstart & Demonstration Guide
*All live perception runners operate 100% on-device with zero cloud dependencies.*

![Live Perception Demo](docs/evidence/live_perception_demo.jpg)

### 7.1 Run Automated Verification Test Suite
Executes the pipeline through benchmark test scenarios (Normal ADLs, Controlled Fall, Fall & Recovery, and Intentional Bed Sleep Discrimination):
```bash
python run_pipeline.py --test
```
*Expected Result: 100% tests pass (4/4 OK) in ~30 seconds, confirming 0 false positives during normal activities, 0 false alarms during sleep, and verified debounced alert emission.*

### 7.2 Launch Interactive Live Web Dashboard
Spins up the real-time Web Dashboard with live video stream, interactive demo switching, threshold sliders, and JSON alert inspector:
```bash
python run_pipeline.py --web --port 8080
```
Open **`http://localhost:8080`** in any web browser.

### 7.3 Live USB Camera Demonstration
Runs live video inference directly from your connected USB camera (device index 0):
```bash
python run_pipeline.py --source 0
```
- Press `p` to toggle Privacy Face Blurring on/off.
- Press `r` to reset fall counters.
- Press `q` to quit.

### 7.4 Run on NVIDIA DeepStream SDK (Jetson / NVIDIA Server)
Launches the native GStreamer DeepStream pipeline using `nvinfer` and `nvtracker`:
```bash
python run_pipeline.py --deepstream --source 0
```

---

## 8. Repository File Structure
```
e:\hackathon project/
├── pipegen.yaml                         # Root PipeGen Manifest & Project Config
├── README.md                            # Comprehensive Documentation
├── run_pipeline.py                      # Universal CLI Entrypoint
│
├── stage1_requirement/
│   ├── requirement.yaml                 # Formal PipeGen Requirements Spec
│   └── requirements_doc.md              # Detailed Specification & Criteria
│
├── stage2_pipeline_ir/
│   ├── pipeline.ir.yaml                 # Hardware-Neutral Pipeline IR (YAML)
│   ├── pipeline.ir.json                 # Validated Pipeline IR (JSON)
│   ├── ir_visualizer.py                 # DAG Topology Validator & Visualizer
│   └── pipeline_graph.md                # Pipeline Flow Graph & Mermaid
│
├── stage3_model/
│   ├── export_onnx.py                   # YOLOv8n-Pose ONNX Exporter
│   ├── yolov8n-pose.onnx                # Exported ONNX Model (12.9 MB)
│   ├── model_card.md                    # Detailed Model Card & Tensor Shapes
│   ├── coco_keypoints.json              # 17 COCO Keypoints Graph
│   └── calibration_dataset.py           # INT8 Calibration Batch Generator
│
├── stage4_compiler/
│   ├── compiler.py                      # PipeGen DeepStream Synthesis Engine
│   ├── trt_compiler.py                  # TensorRT Engine Builder & Profiler
│   ├── performance_report.md            # TensorRT FP16 vs INT8 Benchmark Report
│   ├── deepstream_configs/
│   │   ├── deepstream_app_config.txt    # DeepStream Reference Config
│   │   ├── config_infer_pose.txt        # Primary GIE YOLOv8-Pose Config
│   │   └── nvdcf_tracker_config.txt     # NvDCF Multi-Person Tracker Config
│   └── models/
│       ├── yolov8n-pose_fp16.engine     # FP16 Engine Artifact
│       ├── yolov8n-pose_int8.engine     # INT8 Engine Artifact
│       └── yolov8n-pose_calib.cache     # Quantization Calibration Cache
│
├── stage5_run/
│   ├── fallback_runner.py               # Universal High-Performance Edge Runner
│   ├── deepstream_pipeline.py           # Native DeepStream Python Pipeline
│   ├── fall_analytics.py                # Kinematics Decision Engine & Debouncing
│   ├── person_tracker.py                # Multi-Person Tracker & Track Persistence
│   ├── privacy_filter.py                # Dynamic Face Blurring & Pixelation
│   ├── visualizer.py                    # HUD & Skeletal OSD Renderer
│   ├── mqtt_alert_publisher.py          # MQTT Alert Client & Broker Dispatcher
│   ├── event_schemas.py                 # Pydantic Schemas for Fall JSON Events
│   └── live_dashboard.py                # FastAPI & WebSocket Web Dashboard Server
│
├── test_data/
│   ├── generate_synthetic_demos.py      # Photorealistic Benchmark Video Generator
│   ├── test_fall_detection.py           # Automated Verification Test Suite
│   ├── demo_normal_activity.mp4         # Normal ADLs Benchmark Video
│   ├── demo_fall.mp4                    # Controlled Emergency Fall Video
│   └── demo_fall_and_recovery.mp4       # Fall & Upright Recovery Benchmark Video
│
└── web_ui/                              # State-of-the-Art Web Application
    ├── index.html                       # HTML5 Semantic Layout
    ├── style.css                        # Glassmorphism Dark Mode Styling
    └── app.js                           # WebSocket Telemetry & Interactive UI
```

---

## 9. Physical AI Hackathon Track A Alignment Checklist
- [x] **PipeGen 5 Stages**: Implemented and structured across Stage 1 (Requirement), Stage 2 (Pipeline IR), Stage 3 (Model), Stage 4 (Compiler), and Stage 5 (Run).
- [x] **Pipeline IR**: Validated hardware-neutral DAG defined in `pipeline.ir.yaml` and `pipeline.ir.json`.
- [x] **TensorRT Compilation**: Compiled YOLOv8n-pose with FP16 baseline and INT8 evaluation; generated complete performance report with latency, FPS, memory, and accuracy trade-offs.
- [x] **Edge Vision Pipeline**: Supports live USB camera (`0`), video files, and RTSP streams on NVIDIA GPU/Jetson with DeepStream.
- [x] **Pose Estimation & Tracking**: Extracts 17 COCO keypoints and tracks persistent person IDs across frames.
- [x] **Kinematic Fall Analytics**: Analyzes torso vertical angle, aspect ratio, downward descent velocity, and temporal persistence.
- [x] **ADL Discrimination**: Distinguishes normal walking, sitting, bending, and controlled lying down without false positives.
- [x] **Actionable Outputs**: Emits machine-readable JSON events, publishes over MQTT to `elderly_care/alerts/fall`, and displays real-time annotated video with OSD.
- [x] **Debouncing / Cooldown**: Exactly one alert per confirmed fall event; maintains state and detects upright recovery.
- [x] **Privacy-by-Design**: 100% on-device edge processing, automated facial blurring, and zero identifiable raw video upload.
