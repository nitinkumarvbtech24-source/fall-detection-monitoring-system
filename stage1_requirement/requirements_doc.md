# PipeGen Stage 1: Requirements & Specification Document
## AI-Based Real-Time Privacy-Preserving Fall Detection for Elderly Care
### Physical AI Hackathon — PipeGen Track A (Edge Vision)

---

## 1. Executive Summary & Objective

In elderly care facilities and private residences, unassisted falls represent the leading cause of fatal and non-fatal injuries. Traditional wearable pendants frequently suffer from non-compliance, forgotten charging, or loss during sleep and bathing. Cloud-based video monitoring introduces unacceptable privacy violations, bandwidth saturations, and latency bottlenecks.

This project delivers a **100% on-device, real-time edge AI perception pipeline** built specifically using **PipeGen** for NVIDIA hardware (Jetson Orin / x86 GPU) and **DeepStream/TensorRT**. The system detects human presence, tracks individuals across frames, continuously extracts 17-point skeletal poses, computes kinematic metrics (torso tilt angle, aspect ratio, downward descent velocity, posture persistence), discriminates normal ADLs (Activities of Daily Living) from genuine emergency falls, and dispatches real-time structured JSON alerts via MQTT.

---

## 2. PipeGen 5-Stage Lifecycle Architecture

The system adheres strictly to the 5 mandatory PipeGen stages:

```
+-----------------------------------------------------------------------------+
|                                PIPEGEN STAGES                               |
+-----------------------------------------------------------------------------+
| [1. REQUIREMENT]                                                            |
|     - Hardware targets (Jetson Orin, NVIDIA GPU)                            |
|     - Perception specs: Person detection, 17-pt pose, multi-object tracking  |
|     - Fall kinematics, normal ADL filtering, privacy & alert criteria        |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| [2. PIPELINE IR]                                                            |
|     - Hardware-neutral Intermediate Representation (pipeline.ir.yaml)        |
|     - Formal DAG: Source -> Preprocess -> PGIE -> Tracker -> Fall Analytics |
|       -> Privacy Blurring -> OSD -> Sinks (Video Display + MQTT + JSON)     |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| [3. MODEL]                                                                  |
|     - YOLOv8n-Pose (17 COCO keypoints, lightweight edge architecture)        |
|     - ONNX Export with static/dynamic execution profiles                    |
|     - INT8 PTQ Calibration dataset generation                               |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| [4. COMPILER]                                                               |
|     - TensorRT compilation: FP16 baseline & INT8 quantized engine builds    |
|     - DeepStream configuration generation (nvstreammux, nvinfer, nvtracker)  |
|     - Automated benchmark report generation (FPS, Latency, Memory, Acc)     |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| [5. RUN]                                                                    |
|     - Live execution on USB Camera (index 0), Video File, or RTSP Stream    |
|     - Real-time Fall Detection Engine with Debounce & State Machine          |
|     - Live Annotated Visual OSD + Privacy-Preserving Face Blurring          |
|     - Actionable JSON Event Generation & Real-Time MQTT Alert Broker         |
|     - Interactive Web Dashboard for Live Hackathon Demonstration            |
+-----------------------------------------------------------------------------+
```

---

## 3. Fall Detection Decision Concept & Logic

### 3.1 Keypoint Topology (COCO 17)
The system tracks 17 spatial keypoints:
- Head: 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
- Upper Body: 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow, 9: Left Wrist, 10: Right Wrist
- Lower Body: 11: Left Hip, 12: Right Hip, 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

### 3.2 Postural & Kinematic Metrics

1. **Torso Vertical Angle ($\theta_{\text{torso}}$)**:
   Calculated between the midpoint of the shoulders $M_{\text{shoulder}} = \frac{K_5 + K_6}{2}$ and the midpoint of the hips $M_{\text{hip}} = \frac{K_{11} + K_{12}}{2}$:
   $$\Delta x = |M_{\text{shoulder}, x} - M_{\text{hip}, x}|, \quad \Delta y = |M_{\text{shoulder}, y} - M_{\text{hip}, y}|$$
   $$\theta_{\text{torso}} = \arctan2(\Delta y, \Delta x) \times \frac{180}{\pi}$$
   - **Upright / Standing / Walking**: $\theta_{\text{torso}} \in [65^\circ, 90^\circ]$
   - **Bending / Sitting**: $\theta_{\text{torso}} \in [45^\circ, 70^\circ]$
   - **Horizontal / Fallen**: $\theta_{\text{torso}} \in [0^\circ, 35^\circ]$

2. **Aspect Ratio ($AR$)**:
   Ratio of bounding box width to height:
   $$AR = \frac{\text{Width}}{\text{Height}}$$
   - **Upright**: $AR \in [0.25, 0.75]$
   - **Sitting**: $AR \in [0.70, 1.05]$
   - **Horizontal / Fallen**: $AR \ge 1.15$

3. **Centroid Descent Velocity ($v_y$)**:
   Normalized downward displacement rate over a sliding temporal window $\Delta t$:
   $$v_y = \frac{y_{\text{centroid}}(t) - y_{\text{centroid}}(t - \Delta t)}{H_{\text{person}} \cdot \Delta t}$$
   - A sharp spike ($v_y > 0.15 \text{ s}^{-1}$) captures the sudden collapse characteristic of a fall, distinguishing it from intentional, controlled lying down.

4. **Multi-Frame State Machine & Temporal Persistence**:
   - `NORMAL`: Person is upright, sitting, bending, walking, or moving normally.
   - `RESTING / SLEEPING`: Person is reclining or lying down horizontally in a designated Safe Rest Zone (bed or couch) or entered horizontal posture through controlled descent without an impact velocity spike. **Zero emergency alarms are generated**, and a calm Sky Blue badge is displayed.
   - `POTENTIAL FALL`: Triggered when an uncontrolled horizontal collapse ($\theta_{\text{torso}} < 38^\circ$) occurs outside designated safe rest zones or is accompanied by a downward velocity spike.
   - `FALL CONFIRMED`: Elevated to confirmed when the low, horizontal posture on the floor persists continuously for $N_{\text{persist}} \ge 20$ frames ($\sim 0.7 - 1.0$ sec) without recovery.
   - `RECOVERY`: When a fallen or resting person stands back up ($\theta_{\text{torso}} > 60^\circ$ for $\ge 15$ frames), state seamlessly reverts to `NORMAL`.

5. **Alert Debouncing & Cooldown**:
   - Once `FALL CONFIRMED` is reached, exactly one alert event is dispatched to MQTT and JSON logs.
   - Cooldown timer ($T_{\text{cooldown}} = 15$ seconds) prevents duplicate alert storms while emergency responders are alerted.

### 3.3 Intentional Sleeping vs. Accidental Fall Discrimination
To eliminate false alarms during sleep, bed rest, or intentional couch reclining:
1. **Safe Rest Zones (Geofencing)**: Spatial regions corresponding to beds, couches, and recliners are marked in the pipeline. Any recumbent posture located within these boundaries is classified as `RESTING / SLEEPING`.
2. **Kinematic Descent Velocity & Collapse Spike Analysis**:
   - *Accidental Fall*: Characterized by an abrupt angular drop ($\Delta \theta > 30^\circ$ within 20 frames) accompanied by a vertical centroid descent velocity spike ($v_y > 0.08 \text{ s}^{-1}$).
   - *Intentional Lying Down*: Characterized by controlled, smooth transition into horizontal posture ($v_y \approx 0$) without kinematic shock.
3. **Bed Elevation Ratio**: In typical indoor camera perspectives, beds and sofas elevate the subject ($y_{\text{centroid}} < 0.80 \times H_{\text{frame}}$), whereas genuine falls place the subject on the ground level floor.
4. **Upright Waking Transition**: When the person wakes up and sits or stands upright ($\theta_{\text{torso}} > 62^\circ$), the state smoothly transitions back to `NORMAL`.

---

## 4. Privacy-by-Design Architecture

To honor elder dignity and comply with strict health privacy regulations (HIPAA, GDPR):
1. **Edge-Native Inference**: All neural network inference is executed directly on the local NVIDIA Jetson/GPU edge device. Zero raw video streams are transmitted to any cloud server.
2. **Dynamic Facial Obfuscation**: Head/face keypoints ($K_0$ through $K_4$) are used to compute an adaptive elliptical bounding region over the face, applying real-time Gaussian blur and mosaic pixelation before any display or rendering.
3. **Data Minimization**: In continuous operation, no video or frames are saved to disk. Only machine-readable, lightweight JSON event records containing timestamps and kinematics are preserved.
