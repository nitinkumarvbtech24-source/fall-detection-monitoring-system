"""
PipeGen Stage 5: Visual Analytics OSD Engine
Renders high-definition visual analytics overlays:
- COCO-17 Skeletal joints & bones
- Person tracking bounding box & Track ID
- Dynamic state badges (NORMAL, POTENTIAL FALL, FALL CONFIRMED)
- Kinematics metrics (Torso angle, aspect ratio, confidence)
- HUD Telemetry status bar
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

# COCO 17 Skeletal Connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), # Head
    (5, 6), # Shoulders
    (5, 7), (7, 9), # Left arm
    (6, 8), (8, 10), # Right arm
    (5, 11), (6, 12), (11, 12), # Torso
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16)  # Right leg
]

STATE_COLORS = {
    "NORMAL": (0, 230, 100),              # Vibrant Green (BGR)
    "RESTING / SLEEPING": (255, 200, 0),  # Calm Sky Blue / Cyan (BGR)
    "POTENTIAL FALL": (0, 180, 255),      # Amber / Orange (BGR)
    "FALL CONFIRMED": (40, 40, 255)       # Crimson Red (BGR)
}

class VisualAnalyticsOSD:
    def __init__(self):
        pass

    def render(
        self,
        frame: np.ndarray,
        detections: List[dict],
        fps: float = 30.0,
        latency_ms: float = 12.0,
        total_potential: int = 0,
        total_confirmed: int = 0,
        privacy_enabled: bool = True,
        mqtt_connected: bool = True,
        safe_rest_zones: Optional[List[dict]] = None
    ) -> np.ndarray:
        if frame is None:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 0. Render Safe Rest Zones (Bed / Couch designated ROIs)
        if safe_rest_zones:
            for zone in safe_rest_zones:
                zx1 = zone.get("x1", 0)
                zy1 = zone.get("y1", 0)
                zx2 = zone.get("x2", 0)
                zy2 = zone.get("y2", 0)
                if 0.0 <= zx1 <= 1.0 and 0.0 <= zx2 <= 1.0 and zx2 > 0:
                    px1, py1 = int(zx1 * w), int(zy1 * h)
                    px2, py2 = int(zx2 * w), int(zy2 * h)
                else:
                    px1, py1 = int(zx1), int(zy1)
                    px2, py2 = int(zx2), int(zy2)

                # Draw subtle translucent blue box for Bed Safe Zone
                overlay = canvas.copy()
                cv2.rectangle(overlay, (px1, py1), (px2, py2), (255, 200, 0), -1)
                cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
                cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, "BED / SAFE REST ZONE (SLEEP OK)", (px1 + 10, py1 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 1, cv2.LINE_AA)

        # 1. Render Skeletal Poses and Bounding Boxes for each person
        for det in detections:
            bbox = det.get("bbox")
            kpts = det.get("keypoints")
            track_id = det.get("track_id", 1)
            state = det.get("fall_state", "NORMAL")
            torso = det.get("torso_angle", 90.0)
            ar = det.get("aspect_ratio", 0.5)
            conf = det.get("fall_confidence", 0.0)

            color = STATE_COLORS.get(state, (0, 230, 100))

            # Draw Skeleton Bones
            if kpts is not None and len(kpts) == 17:
                for p1_idx, p2_idx in SKELETON_EDGES:
                    pt1 = kpts[p1_idx]
                    pt2 = kpts[p2_idx]
                    if pt1[2] > 0.30 and pt2[2] > 0.30:
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

                # Draw Keypoint Joints
                for idx, pt in enumerate(kpts):
                    if pt[2] > 0.30:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(canvas, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Draw Bounding Box
            if bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                thickness = 3 if state == "FALL CONFIRMED" else 2
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness, cv2.LINE_AA)

                # State Badge Header
                badge_text = f"ID: {track_id} | {state}"
                if state != "NORMAL":
                    badge_text += f" ({int(conf * 100)}%)"

                metrics_text = f"Torso: {torso:.1f}° | AR: {ar:.2f}"

                # Badge background pill
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                badge_size = cv2.getTextSize(badge_text, font, font_scale, 2)[0]
                metrics_size = cv2.getTextSize(metrics_text, font, 0.45, 1)[0]

                badge_w = max(badge_size[0], metrics_size[0]) + 16
                badge_h = badge_size[1] + metrics_size[1] + 18

                badge_y1 = max(5, by1 - badge_h - 4)
                badge_y2 = badge_y1 + badge_h

                cv2.rectangle(canvas, (bx1, badge_y1), (bx1 + badge_w, badge_y2), (20, 20, 25), -1)
                cv2.rectangle(canvas, (bx1, badge_y1), (bx1 + badge_w, badge_y2), color, 1)

                # Render Text in Badge
                cv2.putText(canvas, badge_text, (bx1 + 8, badge_y1 + badge_size[1] + 4),
                            font, font_scale, color, 2, cv2.LINE_AA)
                cv2.putText(canvas, metrics_text, (bx1 + 8, badge_y2 - 6),
                            font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 2. Render Top HUD Status Bar
        self._render_top_hud(canvas, w, h, fps, latency_ms, len(detections),
                             total_potential, total_confirmed, privacy_enabled, mqtt_connected)

        return canvas

    def _render_top_hud(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        fps: float,
        latency_ms: float,
        people_count: int,
        potential_count: int,
        confirmed_count: int,
        privacy_enabled: bool,
        mqtt_connected: bool
    ):
        hud_h = 44
        # Semi-transparent dark banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (50, 60, 75), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(canvas, "PIPEGEN FALLGUARD EDGE", (14, 28), font, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        # Telemetry metrics
        x_offset = 320
        cv2.putText(canvas, f"FPS: {fps:.1f}", (x_offset, 27), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Latency: {latency_ms:.1f}ms", (x_offset + 95, 27), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"People: {people_count}", (x_offset + 240, 27), font, 0.48, (200, 220, 255), 1, cv2.LINE_AA)

        # Alert Counters
        pot_color = (0, 180, 255) if potential_count > 0 else (160, 160, 160)
        cv2.putText(canvas, f"Potential: {potential_count}", (x_offset + 350, 27), font, 0.48, pot_color, 1, cv2.LINE_AA)

        conf_color = (40, 40, 255) if confirmed_count > 0 else (160, 160, 160)
        cv2.putText(canvas, f"Confirmed: {confirmed_count}", (x_offset + 480, 27), font, 0.48, conf_color, 2, cv2.LINE_AA)

        # Privacy Badge
        priv_text = "PRIVACY: ON" if privacy_enabled else "PRIVACY: OFF"
        priv_color = (0, 230, 100) if privacy_enabled else (100, 100, 100)
        cv2.putText(canvas, priv_text, (w - 240, 27), font, 0.45, priv_color, 1, cv2.LINE_AA)

        # MQTT Badge
        mqtt_text = "MQTT: ACTIVE" if mqtt_connected else "MQTT: STANDBY"
        mqtt_color = (0, 230, 100) if mqtt_connected else (0, 180, 255)
        cv2.putText(canvas, mqtt_text, (w - 120, 27), font, 0.45, mqtt_color, 1, cv2.LINE_AA)
