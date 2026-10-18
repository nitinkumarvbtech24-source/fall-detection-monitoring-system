"""
PipeGen Stage 5: High-Performance Edge Perception Runner
Executes live inference on USB camera, video file, or RTSP stream.
Performs pose estimation, multi-person tracking, fall kinematics analytics,
privacy-preserving facial blurring, and real-time visual OSD rendering.
"""

import cv2
import time
import json
import logging
from pathlib import Path
from typing import Optional, Generator, Tuple, List

import numpy as np
from ultralytics import YOLO

from .person_tracker import MultiPersonTracker
from .fall_analytics import FallAnalyticsEngine
from .privacy_filter import PrivacyFilter
from .visualizer import VisualAnalyticsOSD
from .mqtt_alert_publisher import MQTTAlertPublisher
from .event_schemas import PipelineTelemetry

logger = logging.getLogger("PipeGenEdgeRunner")

class EdgeFallGuardPipeline:
    def __init__(
        self,
        source: str = "0", # "0" for USB camera, or file path, or RTSP URL
        model_path: Optional[str] = None,
        enable_privacy: bool = True,
        enable_mqtt: bool = True,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        show_display: bool = True,
        persistence_frames: int = 20,
        torso_angle_threshold: float = 38.0,
        safe_rest_zones: Optional[List[dict]] = None
    ):
        self.source = source
        self.enable_privacy = enable_privacy
        self.enable_mqtt = enable_mqtt
        self.show_display = show_display

        # Resolve model path
        if model_path is None:
            local_model = Path(__file__).parent.parent / "yolov8n-pose.pt"
            if not local_model.exists():
                local_model = Path("yolov8n-pose.pt")
            self.model_path = str(local_model)
        else:
            self.model_path = model_path

        print(f"[*] Initializing PipeGen Edge Pipeline with model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Core pipeline components
        self.tracker = MultiPersonTracker(iou_threshold=0.30, max_age=30)
        self.alert_publisher = MQTTAlertPublisher(
            broker_host=mqtt_broker,
            broker_port=mqtt_port
        ) if enable_mqtt else None

        self.analytics = FallAnalyticsEngine(
            alert_publisher=self.alert_publisher,
            torso_angle_fall_threshold=torso_angle_threshold,
            persistence_frames=persistence_frames,
            safe_rest_zones=safe_rest_zones
        )
        self.privacy_filter = PrivacyFilter(enabled=enable_privacy)
        self.osd = VisualAnalyticsOSD()

        # Telemetry & performance metrics
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frame_count = 0
        self.is_running = False

    def set_privacy(self, enabled: bool):
        self.enable_privacy = enabled
        self.privacy_filter.set_enabled(enabled)

    def set_thresholds(self, persistence_frames: Optional[int] = None, torso_angle: Optional[float] = None):
        self.analytics.set_thresholds(
            persistence_frames=persistence_frames,
            torso_angle_threshold=torso_angle
        )

    def set_safe_rest_zones(self, zones: List[dict]):
        self.analytics.safe_rest_zones = zones

    def _open_capture(self) -> cv2.VideoCapture:
        # Check if source is integer (camera index)
        try:
            dev_idx = int(self.source)
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
        except ValueError:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[!] Warning: Could not open source '{self.source}'. Attempting fallback...")
            # Try default camera 0
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")

        # Set optimal resolution for real-time edge processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Runs complete perception pipeline on a single frame:
        Inference -> Tracking -> Fall Analytics -> Privacy Masking -> OSD Overlay
        """
        t0 = time.time()
        
        # 1. AI Pose Estimation Inference
        results = self.model(frame, imgsz=640, verbose=False, conf=0.35)
        
        raw_detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            kpts_data = None
            if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                # Shape (N, 17, 3) -> [x, y, conf]
                kpts_data = results[0].keypoints.data.cpu().numpy()

            for i in range(len(boxes)):
                kpt = kpts_data[i] if (kpts_data is not None and i < len(kpts_data)) else None
                raw_detections.append({
                    "bbox": boxes[i],
                    "confidence": float(confs[i]),
                    "keypoints": kpt
                })

        # 2. Multi-Person Tracking
        tracked_detections = self.tracker.update(raw_detections)

        # 3. Fall Kinematics Analytics & Debounced State Machine
        analyzed_detections = self.analytics.process_frame_detections(tracked_detections)

        # 4. Privacy-Preserving Facial Obfuscation
        anonymized_frame = self.privacy_filter.apply(frame, analyzed_detections)

        # Measure latency
        t1 = time.time()
        self.latency_ms = (t1 - t0) * 1000.0
        self.frame_count += 1

        # 5. Visual OSD Rendering
        rendered_frame = self.osd.render(
            anonymized_frame,
            analyzed_detections,
            fps=self.fps,
            latency_ms=self.latency_ms,
            total_potential=self.analytics.total_potential_falls,
            total_confirmed=self.analytics.total_confirmed_falls,
            privacy_enabled=self.enable_privacy,
            mqtt_connected=(self.alert_publisher.is_connected if self.alert_publisher else False),
            safe_rest_zones=self.analytics.safe_rest_zones
        )

        return rendered_frame, analyzed_detections

    def run_live(self):
        """Runs the live pipeline in an interactive OpenCV window."""
        cap = self._open_capture()
        self.is_running = True

        print("=================================================================")
        print(f" PipeGen FallGuard Edge: Live Perception Running ({self.source})")
        print(" Press 'q' to exit | 'p' to toggle Privacy Blur | 'r' to reset falls")
        print("=================================================================")

        prev_time = time.time()
        fps_smoothing = 0.90

        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # If video file, loop back to start
                    if not self.source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps = fps_smoothing * self.fps + (1.0 - fps_smoothing) * current_fps if self.fps > 0 else current_fps

                rendered_frame, detections = self.process_frame(frame)

                if self.show_display:
                    cv2.imshow("PipeGen FallGuard Edge - Live Visual Analytics", rendered_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.set_privacy(not self.enable_privacy)
                        print(f"[*] Privacy Face Masking toggled: {self.enable_privacy}")
                    elif key == ord('r'):
                        self.analytics.total_confirmed_falls = 0
                        self.analytics.total_potential_falls = 0
                        print("[*] Analytics fall counters reset.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.alert_publisher:
                self.alert_publisher.close()
            self.is_running = False
            print("[+] PipeGen Edge Pipeline stopped gracefully.")
