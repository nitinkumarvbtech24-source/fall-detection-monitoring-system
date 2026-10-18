"""
PipeGen Automated Verification & Test Suite
Validates the Fall Detection Pipeline against benchmark test sequences:
1. Test Normal ADLs: 0 false positives.
2. Test Emergency Fall: Confirms fall, produces valid JSON schema, triggers debouncing.
3. Test Recovery: Confirms fall, then detects person standing up and transitions back to NORMAL.
"""

import cv2
import json
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage5_run.fallback_runner import EdgeFallGuardPipeline
from stage5_run.event_schemas import FallAlertEvent

class TestFallDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent
        cls.normal_video = cls.test_dir / "demo_normal_activity.mp4"
        cls.fall_video = cls.test_dir / "demo_fall.mp4"
        cls.recovery_video = cls.test_dir / "demo_fall_and_recovery.mp4"

    def test_01_normal_activity_no_false_alerts(self):
        print("\n---------------------------------------------------------")
        print("TEST 1: Normal Activity Discrimination (Walking / Sitting / Bending)")
        print("---------------------------------------------------------")
        pipeline = EdgeFallGuardPipeline(
            source=str(self.normal_video),
            enable_privacy=True,
            enable_mqtt=False,
            show_display=False,
            persistence_frames=18
        )

        cap = cv2.VideoCapture(str(self.normal_video))
        dispatched_alerts = []
        pipeline.analytics.alert_publisher = None # Capture directly

        def on_alert(event):
            dispatched_alerts.append(event)

        # Mock alert publisher
        class MockPublisher:
            def publish_fall_alert(self, ev):
                dispatched_alerts.append(ev)
        
        pipeline.analytics.alert_publisher = MockPublisher()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rendered, dets = pipeline.process_frame(frame)
            frame_idx += 1

        cap.release()
        print(f"[*] Processed {frame_idx} frames.")
        print(f"[*] Confirmed Falls: {pipeline.analytics.total_confirmed_falls}")
        print(f"[*] Dispatched Alerts: {len(dispatched_alerts)}")

        self.assertEqual(pipeline.analytics.total_confirmed_falls, 0, "Normal activity triggered false positive fall!")
        self.assertEqual(len(dispatched_alerts), 0, "Dispatched alert during normal activity!")
        print("[+] PASS: Zero false positives during normal activities.")

    def test_02_emergency_fall_detection_and_alert(self):
        print("\n---------------------------------------------------------")
        print("TEST 2: Controlled Emergency Fall Detection & JSON Event Output")
        print("---------------------------------------------------------")
        dispatched_alerts = []

        class MockPublisher:
            def publish_fall_alert(self, ev):
                dispatched_alerts.append(ev)

        pipeline = EdgeFallGuardPipeline(
            source=str(self.fall_video),
            enable_privacy=True,
            enable_mqtt=False,
            show_display=False,
            persistence_frames=15, # Quick persistence for test
            torso_angle_threshold=42.0
        )
        pipeline.analytics.alert_publisher = MockPublisher()

        cap = cv2.VideoCapture(str(self.fall_video))
        frame_idx = 0
        potential_fall_seen = False
        confirmed_fall_seen = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rendered, dets = pipeline.process_frame(frame)
            for d in dets:
                if d.get("fall_state") == "POTENTIAL FALL":
                    potential_fall_seen = True
                elif d.get("fall_state") == "FALL CONFIRMED":
                    confirmed_fall_seen = True
            frame_idx += 1

        cap.release()
        print(f"[*] Processed {frame_idx} frames.")
        print(f"[*] Potential Fall Observed: {potential_fall_seen}")
        print(f"[*] Confirmed Fall Observed: {confirmed_fall_seen}")
        print(f"[*] Total Confirmed Count: {pipeline.analytics.total_confirmed_falls}")
        print(f"[*] Total Dispatched Alerts: {len(dispatched_alerts)}")

        self.assertTrue(potential_fall_seen, "Did not observe POTENTIAL FALL state transition!")
        self.assertTrue(confirmed_fall_seen, "Did not observe FALL CONFIRMED state transition!")
        self.assertEqual(len(dispatched_alerts), 1, "Expected exactly 1 debounced alert event for single fall!")

        # Validate JSON Schema of Dispatched Event
        alert = dispatched_alerts[0]
        print("\n[*] Validated Machine-Readable JSON Event:")
        print(json.dumps(alert.model_dump(), indent=2))
        
        self.assertEqual(alert.event, "fall_detected")
        self.assertEqual(alert.status, "FALL_CONFIRMED")
        self.assertGreaterEqual(alert.confidence, 0.70)
        self.assertLessEqual(alert.metrics.torso_angle_deg, 42.0)
        print("[+] PASS: Fall confirmed with valid machine-readable JSON schema and single debounced alert.")

    def test_03_fall_and_recovery(self):
        print("\n---------------------------------------------------------")
        print("TEST 3: Fall Detection with Upright Recovery")
        print("---------------------------------------------------------")
        dispatched_alerts = []

        class MockPublisher:
            def publish_fall_alert(self, ev):
                dispatched_alerts.append(ev)

        pipeline = EdgeFallGuardPipeline(
            source=str(self.recovery_video),
            enable_privacy=True,
            enable_mqtt=False,
            show_display=False,
            persistence_frames=15,
            torso_angle_threshold=42.0
        )
        pipeline.analytics.recovery_frames = 10
        pipeline.analytics.alert_publisher = MockPublisher()

        cap = cv2.VideoCapture(str(self.recovery_video))
        final_state = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rendered, dets = pipeline.process_frame(frame)
            if dets:
                final_state = dets[0].get("fall_state")

        cap.release()
        print(f"[*] Final Track State after standing back up: {final_state}")
        self.assertEqual(len(dispatched_alerts), 1, "Expected 1 alert during fall phase")
        self.assertEqual(final_state, "NORMAL", "State did not revert to NORMAL after person stood back up!")
        print("[+] PASS: Successfully transitioned back to NORMAL upon upright recovery.")

    def test_04_intentional_sleeping_no_false_alarm(self):
        print("\n---------------------------------------------------------")
        print("TEST 4: Intentional Sleeping / Reclining Discrimination")
        print("---------------------------------------------------------")
        dispatched_alerts = []

        class MockPublisher:
            def publish_fall_alert(self, ev):
                dispatched_alerts.append(ev)

        pipeline = EdgeFallGuardPipeline(
            source=str(self.fall_video),
            enable_privacy=True,
            enable_mqtt=False,
            show_display=False,
            persistence_frames=15,
            torso_angle_threshold=42.0
        )
        pipeline.analytics.alert_publisher = MockPublisher()

        # Simulate intentional lying down on a bed / safe resting zone:
        pipeline.analytics.safe_rest_zones = [{"x1": 0, "y1": 0, "x2": 2000, "y2": 2000}] # All resting area

        cap = cv2.VideoCapture(str(self.fall_video))
        resting_state_observed = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rendered, dets = pipeline.process_frame(frame)
            if dets and dets[0].get("fall_state") == "RESTING / SLEEPING":
                resting_state_observed = True

        cap.release()
        print(f"[*] Resting / Sleeping State Observed: {resting_state_observed}")
        print(f"[*] Confirmed Falls Count: {pipeline.analytics.total_confirmed_falls}")
        print(f"[*] Dispatched Alerts: {len(dispatched_alerts)}")

        self.assertTrue(resting_state_observed, "Expected RESTING / SLEEPING state in safe rest zone!")
        self.assertEqual(pipeline.analytics.total_confirmed_falls, 0, "Sleeping fired a false fall alarm!")
        self.assertEqual(len(dispatched_alerts), 0, "Sleeping dispatched an emergency alert!")
        print("[+] PASS: Sleeping/resting identified accurately with ZERO false alarms.")

    def test_05_straight_fall_longitudinal_detection(self):
        print("\n---------------------------------------------------------")
        print("TEST 5: Straight Longitudinal Fall (Depth Axis on Floor outside Bed Zone)")
        print("---------------------------------------------------------")
        from stage5_run.fall_analytics import FallAnalyticsEngine
        import numpy as np

        dispatched_alerts = []
        class MockPublisher:
            def publish_fall_alert(self, ev):
                dispatched_alerts.append(ev)

        engine = FallAnalyticsEngine(
            alert_publisher=MockPublisher(),
            persistence_frames=15,
            safe_rest_zones=[{"x1": 800, "y1": 100, "x2": 1200, "y2": 400}], # Bed is elsewhere
            screenshot_provider=lambda tid: "/snapshots/test_straight_fall.jpg"
        )

        # Exact posture from user's webcam:
        # AR = 2.11, head further back (smaller y), shoulders closer (larger y), hips at bottom
        # Width: 422, Height: 200 -> AR: 2.11
        bbox = np.array([100.0, 400.0, 522.0, 600.0])
        # 17 keypoints
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Head keypoints (indices 0..4) at y=420
        for i in range(5):
            kpts[i] = [280.0 + i * 5, 420.0, 0.9]
        # Shoulders (5, 6) at y=460 (10+ px below head in 2D perspective)
        kpts[5] = [220.0, 460.0, 0.9] # L shoulder
        kpts[6] = [340.0, 460.0, 0.9] # R shoulder
        # Hips (11, 12) at y=550 (below shoulders)
        kpts[11] = [250.0, 550.0, 0.9] # L hip
        kpts[12] = [310.0, 550.0, 0.9] # R hip

        det = {
            "track_id": 3,
            "bbox": bbox,
            "confidence": 0.92,
            "keypoints": kpts
        }

        # Process frames simulating persistence on floor
        states_seen = []
        for f in range(25):
            dets_out = engine.process_frame_detections([det], frame_time=1000.0 + f * 0.033)
            states_seen.append(dets_out[0]["fall_state"])

        print(f"[*] States seen during straight fall: {set(states_seen)}")
        print(f"[*] Final Torso Angle: {dets_out[0]['torso_angle']}°")
        print(f"[*] Final Aspect Ratio: {dets_out[0]['aspect_ratio']}")
        print(f"[*] Total Confirmed Falls: {engine.total_confirmed_falls}")
        print(f"[*] Dispatched Alerts: {len(dispatched_alerts)}")

        self.assertIn("POTENTIAL FALL", states_seen, "Straight fall should enter POTENTIAL FALL!")
        self.assertIn("FALL CONFIRMED", states_seen, "Straight fall should be confirmed!")
        self.assertEqual(dets_out[0]["fall_state"], "FALL CONFIRMED", "Final state must be FALL CONFIRMED!")
        self.assertLessEqual(dets_out[0]["torso_angle"], 15.0, "Torso angle must reflect horizontal ground posture, not 75°!")
        self.assertEqual(len(dispatched_alerts), 1, "Expected exactly 1 dispatched fall alert for straight fall!")
        print("[+] PASS: Straight longitudinal fall on floor successfully confirmed as emergency fall!")

if __name__ == "__main__":
    unittest.main()

