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

if __name__ == "__main__":
    unittest.main()
