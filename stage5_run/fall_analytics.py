"""
PipeGen Stage 5: Fall Kinematics Analytics Engine
Analyzes human posture, temporal kinematics, descent velocity, and posture persistence across frames.
Distinguishes genuine emergency falls from normal Activities of Daily Living (ADLs).
"""

import math
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .event_schemas import FallAlertEvent, FallMetrics
from .mqtt_alert_publisher import MQTTAlertPublisher

class PersonKinematicState:
    """Maintains temporal sliding history for a single tracked person."""
    def __init__(self, track_id: int, max_history: int = 30):
        self.track_id = track_id
        self.state = "NORMAL" # "NORMAL", "POTENTIAL FALL", "FALL CONFIRMED"
        self.history = deque(maxlen=max_history) # stores dict of frame metrics

        # Fall persistence counters
        self.fallen_consecutive_frames = 0
        self.recovery_consecutive_frames = 0
        self.first_fallen_timestamp = None

        # Debounce / Cooldown management
        self.alert_dispatched = False
        self.last_alert_timestamp = 0.0

        # Smoothed metrics
        self.current_torso_angle = 90.0
        self.current_aspect_ratio = 0.50
        self.current_velocity = 0.0
        self.current_confidence = 0.0

class FallAnalyticsEngine:
    def __init__(
        self,
        alert_publisher: Optional[MQTTAlertPublisher] = None,
        torso_angle_fall_threshold: float = 38.0,
        torso_angle_upright_threshold: float = 62.0,
        aspect_ratio_fall_threshold: float = 1.15,
        aspect_ratio_upright_threshold: float = 0.85,
        vertical_velocity_threshold: float = 0.08,
        persistence_frames: int = 20,
        recovery_frames: int = 15,
        cooldown_seconds: float = 15.0,
        bed_elevation_ratio: float = 0.68,
        safe_rest_zones: Optional[List[dict]] = None
    ):
        self.alert_publisher = alert_publisher
        self.torso_angle_fall_threshold = torso_angle_fall_threshold
        self.torso_angle_upright_threshold = torso_angle_upright_threshold
        self.aspect_ratio_fall_threshold = aspect_ratio_fall_threshold
        self.aspect_ratio_upright_threshold = aspect_ratio_upright_threshold
        self.vertical_velocity_threshold = vertical_velocity_threshold
        self.persistence_frames = persistence_frames
        self.recovery_frames = recovery_frames
        self.cooldown_seconds = cooldown_seconds
        self.bed_elevation_ratio = bed_elevation_ratio
        self.safe_rest_zones = safe_rest_zones or []

        self.person_states: Dict[int, PersonKinematicState] = {}
        self.total_potential_falls = 0
        self.total_confirmed_falls = 0
        self.total_resting_events = 0

    def set_thresholds(
        self,
        persistence_frames: Optional[int] = None,
        torso_angle_threshold: Optional[float] = None,
        velocity_threshold: Optional[float] = None
    ):
        """Allows dynamic live tuning from Web Dashboard / CLI."""
        if persistence_frames is not None:
            self.persistence_frames = persistence_frames
        if torso_angle_threshold is not None:
            self.torso_angle_fall_threshold = torso_angle_threshold
        if velocity_threshold is not None:
            self.vertical_velocity_threshold = velocity_threshold

    def process_frame_detections(
        self,
        detections: List[dict],
        frame_time: Optional[float] = None
    ) -> List[dict]:
        """
        Analyzes tracked detections, updates kinematics, determines state transitions,
        and triggers alerts when fall criteria are met.
        """
        now = frame_time if frame_time is not None else time.time()
        active_track_ids = set()

        for det in detections:
            track_id = det.get("track_id", 1)
            active_track_ids.add(track_id)

            if track_id not in self.person_states:
                self.person_states[track_id] = PersonKinematicState(track_id)

            p_state = self.person_states[track_id]

            # 1. Compute instantaneous postural metrics
            torso_angle, aspect_ratio, centroid_y = self._calculate_postural_metrics(det)
            
            # 2. Compute downward descent velocity from temporal history
            descent_velocity = self._calculate_downward_velocity(p_state, centroid_y, det["bbox"], now)

            # Store in temporal sliding window
            p_state.history.append({
                "time": now,
                "torso_angle": torso_angle,
                "aspect_ratio": aspect_ratio,
                "centroid_y": centroid_y,
                "velocity": descent_velocity
            })

            # Smoothed instantaneous metrics
            p_state.current_torso_angle = torso_angle
            p_state.current_aspect_ratio = aspect_ratio
            p_state.current_velocity = descent_velocity

            # 3. Evaluate State Machine & ADL Discrimination
            self._update_state_machine(p_state, det, now)

            # Augment detection dict with analytics metadata for visualization
            det["fall_state"] = p_state.state
            det["torso_angle"] = p_state.current_torso_angle
            det["aspect_ratio"] = p_state.current_aspect_ratio
            det["descent_velocity"] = p_state.current_velocity
            det["fall_confidence"] = p_state.current_confidence
            det["fallen_duration"] = (now - p_state.first_fallen_timestamp) if p_state.first_fallen_timestamp else 0.0

        # Clean up stale person states that haven't been seen in > 60 seconds
        for tid in list(self.person_states.keys()):
            if tid not in active_track_ids and len(self.person_states[tid].history) > 0:
                if (now - self.person_states[tid].history[-1]["time"]) > 60.0:
                    del self.person_states[tid]

        return detections

    def _calculate_postural_metrics(self, det: dict) -> Tuple[float, float, float]:
        """Calculates torso vertical tilt angle, aspect ratio, and vertical centroid."""
        bbox = det["bbox"] # [x1, y1, x2, y2]
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        aspect_ratio = bw / bh
        centroid_y = (float(bbox[1]) + float(bbox[3])) / 2.0

        kpts = det.get("keypoints")
        torso_angle = None
        head_is_upright = False

        # 1. Check Head vs Shoulders orientation (prevents desk/webcam crop false alarms)
        if kpts is not None and len(kpts) >= 7:
            head_pts = [k for k in kpts[:5] if k[2] > 0.25]
            sh_pts = [k for k in [kpts[5], kpts[6]] if k[2] > 0.25]

            if head_pts and sh_pts:
                avg_head_y = sum(k[1] for k in head_pts) / len(head_pts)
                avg_sh_y = sum(k[1] for k in sh_pts) / len(sh_pts)
                # If head is clearly above shoulders by at least 10 pixels:
                if (avg_sh_y - avg_head_y) > 10.0:
                    head_is_upright = True

            # 2. If hips are visible, calculate anatomical torso angle
            if len(kpts) >= 13:
                l_sh, r_sh = kpts[5], kpts[6]
                l_hip, r_hip = kpts[11], kpts[12]

                valid_shoulders = (l_sh[2] > 0.25) or (r_sh[2] > 0.25)
                valid_hips = (l_hip[2] > 0.25) or (r_hip[2] > 0.25)

                if valid_shoulders and valid_hips:
                    sh_x = (l_sh[0] + r_sh[0]) / 2.0 if (l_sh[2] > 0.25 and r_sh[2] > 0.25) else (l_sh[0] if l_sh[2] > 0.25 else r_sh[0])
                    sh_y = (l_sh[1] + r_sh[1]) / 2.0 if (l_sh[2] > 0.25 and r_sh[2] > 0.25) else (l_sh[1] if l_sh[2] > 0.25 else r_sh[1])

                    hip_x = (l_hip[0] + r_hip[0]) / 2.0 if (l_hip[2] > 0.25 and r_hip[2] > 0.25) else (l_hip[0] if l_hip[2] > 0.25 else r_hip[0])
                    hip_y = (l_hip[1] + r_hip[1]) / 2.0 if (l_hip[2] > 0.25 and r_hip[2] > 0.25) else (l_hip[1] if l_hip[2] > 0.25 else r_hip[1])

                    dx = abs(sh_x - hip_x)
                    dy = abs(sh_y - hip_y)

                    # Angle relative to horizontal ground (90 deg is vertical, 0 deg is flat horizontal)
                    angle_rad = math.atan2(dy, max(1e-5, dx))
                    torso_angle = math.degrees(angle_rad)

        # 3. Robust fallback: if lower body/hips are out of frame (e.g. sitting at desk):
        if torso_angle is None:
            if head_is_upright:
                # Head above shoulders -> unambiguously upright sitting or standing
                torso_angle = 85.0
            elif aspect_ratio >= 1.6:
                # Wide horizontal box with head NOT upright -> flat recumbent on floor
                torso_angle = 15.0
            else:
                torso_angle = 75.0

        # If head is clearly upright, force torso angle to upright range (min 70 deg)
        if head_is_upright and torso_angle < 50.0:
            torso_angle = 75.0

        return float(torso_angle), float(aspect_ratio), float(centroid_y)

    def _calculate_downward_velocity(
        self,
        p_state: PersonKinematicState,
        current_centroid_y: float,
        bbox: np.ndarray,
        current_time: float
    ) -> float:
        """
        Calculates normalized downward velocity: (delta_y / person_height) / delta_t.
        Positive velocity indicates downward motion / collapse.
        """
        if len(p_state.history) < 3:
            return 0.0

        bh = max(1.0, float(bbox[3] - bbox[1]))
        
        # Look back 5 to 10 frames (~0.2 - 0.4 seconds)
        lookback_idx = max(0, len(p_state.history) - 6)
        past_frame = p_state.history[lookback_idx]

        dt = current_time - past_frame["time"]
        if dt <= 0.01:
            return 0.0

        dy = current_centroid_y - past_frame["centroid_y"]
        normalized_velocity = (dy / bh) / dt
        return float(normalized_velocity)

    def _is_in_safe_rest_zone(self, bbox: np.ndarray, frame_w: float = 1280.0, frame_h: float = 720.0) -> bool:
        """Checks if the person's centroid is inside any defined safe resting zone (bed / sofa)."""
        if not self.safe_rest_zones:
            return False
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        for zone in self.safe_rest_zones:
            zx1 = float(zone.get("x1", 0))
            zy1 = float(zone.get("y1", 0))
            zx2 = float(zone.get("x2", 0))
            zy2 = float(zone.get("y2", 0))
            # Support normalized coordinates [0.0 - 1.0]
            if 0.0 <= zx1 <= 1.0 and 0.0 <= zx2 <= 1.0 and zx2 > 0:
                zx1 *= frame_w
                zx2 *= frame_w
                zy1 *= frame_h
                zy2 *= frame_h
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return True
        return False

    def _update_state_machine(self, p_state: PersonKinematicState, det: dict, now: float):
        """
        Evaluates multi-frame state transitions:
        NORMAL -> [RESTING / SLEEPING] (safe zone or controlled bed descent)
               -> [POTENTIAL FALL] -> [FALL CONFIRMED] (sudden collapse / persistence)
               -> [RECOVERY] -> NORMAL
        """
        torso = p_state.current_torso_angle
        ar = p_state.current_aspect_ratio
        vel = p_state.current_velocity
        bbox = det["bbox"]
        centroid_y = (float(bbox[1]) + float(bbox[3])) / 2.0

        # Condition 1: Horizontal Posture Signature
        is_horizontal = (torso < self.torso_angle_fall_threshold)
        
        # Condition 2: Sudden Collapse Signature (from upright standing to horizontal in <20 frames)
        recent_torsos = [h["torso_angle"] for h in list(p_state.history)[-20:]]
        max_recent_torso = max(recent_torsos) if recent_torsos else torso
        is_sudden_collapse = (max_recent_torso > 62.0) and (max_recent_torso - torso > 30.0)

        # Condition 3: Sudden Downward Velocity Spike
        has_downward_spike = (vel > self.vertical_velocity_threshold) or any(
            h["velocity"] > self.vertical_velocity_threshold for h in list(p_state.history)[-10:]
        )

        # Condition 4: Upright Posture (Standing or Sitting upright)
        is_upright = (torso > self.torso_angle_upright_threshold)

        # Condition 5: Safe Rest Zone / Bed Context
        in_safe_rest_zone = self._is_in_safe_rest_zone(bbox)

        # Condition 6: Controlled Bed Recline (Elevated surface + no violent impact)
        is_elevated_bed = (centroid_y < 720.0 * self.bed_elevation_ratio)
        is_controlled_recline = (not is_sudden_collapse and not has_downward_spike and is_elevated_bed)

        # -------------------------------------------------------------
        # STATE 1: CURRENTLY IN NORMAL
        # -------------------------------------------------------------
        if p_state.state == "NORMAL":
            if is_horizontal:
                # Discriminate: Is this intentional sleeping in a designated Safe Rest Zone (bed / couch)?
                if in_safe_rest_zone:
                    # In Bed or Couch -> Intentional Sleeping / Resting (Zero alarms)
                    p_state.state = "RESTING / SLEEPING"
                    p_state.fallen_consecutive_frames = 0
                    p_state.recovery_consecutive_frames = 0
                    p_state.first_fallen_timestamp = None
                    p_state.current_confidence = 0.0
                    self.total_resting_events += 1
                else:
                    # Fallen on the floor outside safe rest zones -> Potential Emergency Fall!
                    p_state.state = "POTENTIAL FALL"
                    p_state.fallen_consecutive_frames = 1
                    p_state.first_fallen_timestamp = now
                    p_state.recovery_consecutive_frames = 0
                    self.total_potential_falls += 1
                    p_state.current_confidence = 0.85
            else:
                p_state.fallen_consecutive_frames = 0
                p_state.current_confidence = 0.0

        # -------------------------------------------------------------
        # STATE 2: CURRENTLY IN RESTING / SLEEPING
        # -------------------------------------------------------------
        elif p_state.state == "RESTING / SLEEPING":
            # Person is sleeping or resting horizontally in a safe rest zone. Zero alarms are triggered.
            if is_horizontal:
                p_state.current_confidence = 0.0
                p_state.fallen_consecutive_frames = 0
            elif is_upright:
                # Person woke up and sat or stood up!
                p_state.recovery_consecutive_frames += 1
                if p_state.recovery_consecutive_frames >= 4:
                    p_state.state = "NORMAL"
                    p_state.recovery_consecutive_frames = 0
                    p_state.first_fallen_timestamp = None

        # -------------------------------------------------------------
        # STATE 3: CURRENTLY IN POTENTIAL FALL
        # -------------------------------------------------------------
        elif p_state.state == "POTENTIAL FALL":
            if in_safe_rest_zone:
                p_state.state = "RESTING / SLEEPING"
                p_state.fallen_consecutive_frames = 0
                p_state.recovery_consecutive_frames = 0
                p_state.first_fallen_timestamp = None
                p_state.current_confidence = 0.0
                self.total_resting_events += 1
            elif is_horizontal:
                p_state.fallen_consecutive_frames += 1
                
                # Confidence grows with persistence & rapid velocity history
                persistence_ratio = min(1.0, p_state.fallen_consecutive_frames / float(self.persistence_frames))
                base_conf = 0.65 if has_downward_spike else 0.45
                p_state.current_confidence = base_conf + 0.30 * persistence_ratio

                # Check if persistence duration threshold is met
                if p_state.fallen_consecutive_frames >= self.persistence_frames:
                    p_state.state = "FALL CONFIRMED"
                    p_state.current_confidence = min(0.99, max(0.85, p_state.current_confidence))
                    self.total_confirmed_falls += 1

                    # Trigger ACTIONABLE EVENT (Single alert with debouncing)
                    self._dispatch_fall_event(p_state, det, now)
            else:
                # Person quickly regained composure (e.g. temporary bend, picking up dropped item)
                p_state.recovery_consecutive_frames += 1
                if p_state.recovery_consecutive_frames >= 4:
                    p_state.state = "NORMAL"
                    p_state.fallen_consecutive_frames = 0
                    p_state.recovery_consecutive_frames = 0
                    p_state.first_fallen_timestamp = None
                    p_state.current_confidence = 0.0

        # -------------------------------------------------------------
        # STATE 4: CURRENTLY IN FALL CONFIRMED
        # -------------------------------------------------------------
        elif p_state.state == "FALL CONFIRMED":
            # Maintain FALL CONFIRMED state while person remains down
            if is_horizontal:
                p_state.fallen_consecutive_frames += 1
                p_state.recovery_consecutive_frames = 0
                p_state.current_confidence = 0.96
            elif is_upright:
                # Person stands back up!
                p_state.recovery_consecutive_frames += 1
                if p_state.recovery_consecutive_frames >= self.recovery_frames:
                    print(f"[*] Track {p_state.track_id}: Recovery confirmed. Returning to NORMAL.")
                    p_state.state = "NORMAL"
                    p_state.fallen_consecutive_frames = 0
                    p_state.recovery_consecutive_frames = 0
                    p_state.first_fallen_timestamp = None
                    p_state.alert_dispatched = False
                    p_state.current_confidence = 0.0

    def _dispatch_fall_event(self, p_state: PersonKinematicState, det: dict, timestamp: float):
        """Constructs and emits the structured JSON fall alert event."""
        # Debounce check
        if p_state.alert_dispatched and (timestamp - p_state.last_alert_timestamp) < self.cooldown_seconds:
            return

        fallen_duration = (timestamp - p_state.first_fallen_timestamp) if p_state.first_fallen_timestamp else 1.0

        metrics = FallMetrics(
            torso_angle_deg=round(p_state.current_torso_angle, 1),
            aspect_ratio=round(p_state.current_aspect_ratio, 2),
            descent_velocity=round(p_state.current_velocity, 2),
            fallen_duration_sec=round(fallen_duration, 2),
            persistence_frames=p_state.fallen_consecutive_frames
        )

        iso_timestamp = datetime.now().isoformat(timespec="seconds")

        event = FallAlertEvent(
            event="fall_detected",
            person_id=p_state.track_id,
            confidence=round(p_state.current_confidence, 2),
            timestamp=iso_timestamp,
            status="FALL_CONFIRMED",
            location="Room 101 - Primary Care Area",
            metrics=metrics,
            privacy_applied=True
        )

        p_state.alert_dispatched = True
        p_state.last_alert_timestamp = timestamp

        if self.alert_publisher:
            self.alert_publisher.publish_fall_alert(event)
