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
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator, Tuple, List

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

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

        # Root directories & snapshot storage
        self.root_dir = Path(__file__).parent.parent
        self.snapshots_dir = self.root_dir / "web_ui" / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.last_frame: Optional[np.ndarray] = None
        self.last_rendered_frame: Optional[np.ndarray] = None
        self.last_snapshot_url: Optional[str] = None
        self.alarm_active: bool = False
        self.alarm_silenced: bool = False
        self._alarm_thread: Optional[threading.Thread] = None

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
            safe_rest_zones=safe_rest_zones,
            screenshot_provider=self.capture_screenshot
        )
        self.privacy_filter = PrivacyFilter(enabled=enable_privacy)
        self.osd = VisualAnalyticsOSD()

        # Telemetry & performance metrics
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frame_count = 0
        self.is_running = False

    def capture_screenshot(self, person_id: Optional[int] = None, prefix: str = "fall_alert") -> Optional[str]:
        """
        Captures and saves a high-resolution snapshot with visual alert annotations to web_ui/snapshots.
        Returns relative URL accessible by the web dashboard (e.g. /snapshots/fall_alert_person_1_20261018_171120.jpg).
        """
        try:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            id_str = f"_person_{person_id}" if person_id is not None else ""
            filename = f"{prefix}{id_str}_{ts_str}.jpg"
            filepath = self.snapshots_dir / filename

            # Prefer rendered frame with OSD overlay, fallback to last raw frame
            frame_to_save = self.last_rendered_frame if self.last_rendered_frame is not None else self.last_frame
            if frame_to_save is not None:
                snap = frame_to_save.copy()
                h, w = snap.shape[:2]

                # Render Emergency Header Banner with Alert Symbol directly on the saved screenshot
                overlay = snap.copy()
                cv2.rectangle(overlay, (0, 0), (w, 54), (18, 18, 190), -1)
                cv2.addWeighted(overlay, 0.88, snap, 0.12, 0, snap)
                cv2.line(snap, (0, 54), (w, 54), (0, 40, 255), 2)

                cv2.putText(snap, "[!] EMERGENCY: FALL CONFIRMED", (18, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(snap, f"Track #{person_id or 1} | {time_str}",
                            (max(10, w - 380), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 255), 1, cv2.LINE_AA)

                cv2.imwrite(str(filepath), snap)
                rel_url = f"/snapshots/{filename}"
                self.last_snapshot_url = rel_url
                print(f"[📸 SCREENSHOT CAPTURED] Saved verified fall image: {filepath}")

                # Trigger audible siren until switched off
                if not self.alarm_silenced:
                    self.trigger_alarm()

                return rel_url
        except Exception as e:
            logger.error(f"[!] Error saving fall screenshot: {e}")
        return None

    def trigger_alarm(self):
        """Activates continuous audible emergency siren until someone switches it off."""
        if self.alarm_active:
            return
        self.alarm_active = True
        self.alarm_silenced = False
        print("\n🚨 [ALARM ACTIVE] Emergency fall alert sound triggered! Playing until switched off...")
        if WINSOUND_AVAILABLE and (self._alarm_thread is None or not self._alarm_thread.is_alive()):
            self._alarm_thread = threading.Thread(target=self._run_cli_siren, daemon=True)
            self._alarm_thread.start()

    def silence_alarm(self):
        """Switches off / silences the emergency alarm sound."""
        self.alarm_active = False
        self.alarm_silenced = True
        print("[🔕 ALARM SWITCHED OFF] Emergency alert siren has been muted by caregiver.")

    def _run_cli_siren(self):
        """Alternating two-tone emergency buzzer loop (Windows winsound)."""
        while self.alarm_active:
            try:
                winsound.Beep(950, 220)
                if not self.alarm_active:
                    break
                time.sleep(0.08)
                winsound.Beep(700, 220)
                if not self.alarm_active:
                    break
                time.sleep(0.08)
            except Exception:
                break

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
        self.last_frame = frame
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

        self.last_rendered_frame = rendered_frame
        return rendered_frame, analyzed_detections

    def run_live(self):
        """Runs the live pipeline in an interactive OpenCV window."""
        cap = self._open_capture()
        self.is_running = True

        print("=================================================================")
        print(f" PipeGen FallGuard Edge: Live Perception Running ({self.source})")
        print(" Press 'q' to exit | 's' to switch off alarm | 'p' to toggle Privacy | 'r' to reset")
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
                    display_frame = rendered_frame.copy()
                    
                    # Render Emergency Flashing Alert Symbol Banner when Alarm is Active
                    if self.alarm_active:
                        h_f, w_f = display_frame.shape[:2]
                        flash_color = (0, 0, 255) if int(time.time() * 3) % 2 == 0 else (0, 160, 255)
                        # Emergency outer border
                        cv2.rectangle(display_frame, (0, 0), (w_f - 1, h_f - 1), flash_color, 6)
                        # Bottom warning banner with Alert Symbol
                        banner_y1 = max(10, h_f - 65)
                        cv2.rectangle(display_frame, (10, banner_y1), (w_f - 10, h_f - 10), (15, 15, 22), -1)
                        cv2.rectangle(display_frame, (10, banner_y1), (w_f - 10, h_f - 10), flash_color, 2)
                        cv2.putText(display_frame, "[!] EMERGENCY: FALL DETECTED! ALARM SOUNDING", (25, h_f - 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, flash_color, 2, cv2.LINE_AA)
                        cv2.putText(display_frame, "Press 's' to SWITCH OFF ALARM", (max(300, w_f - 380), h_f - 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.imshow("PipeGen FallGuard Edge - Live Visual Analytics", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.silence_alarm()
                    elif key == ord('p'):
                        self.set_privacy(not self.enable_privacy)
                        print(f"[*] Privacy Face Masking toggled: {self.enable_privacy}")
                    elif key == ord('r'):
                        self.silence_alarm()
                        self.analytics.total_confirmed_falls = 0
                        self.analytics.total_potential_falls = 0
                        print("[*] Analytics fall counters and alarms reset.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.alert_publisher:
                self.alert_publisher.close()
            self.is_running = False
            print("[+] PipeGen Edge Pipeline stopped gracefully.")
