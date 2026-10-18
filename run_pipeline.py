"""
PipeGen Track A: Real-Time Privacy-Preserving Fall Detection System
Universal CLI Entrypoint for Edge Vision Pipeline
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from stage5_run.fallback_runner import EdgeFallGuardPipeline
from stage5_run.deepstream_pipeline import DeepStreamPipelineRunner

def main():
    parser = argparse.ArgumentParser(
        description="PipeGen Track A: AI-Based Privacy-Preserving Fall Detection Edge Vision System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        default="0",
        help="Input video source: camera index (e.g. '0' for USB camera), file path (e.g. 'test_data/demo_fall.mp4'), or RTSP URL"
    )
    parser.add_argument(
        "--deepstream",
        action="store_true",
        help="Execute using native NVIDIA DeepStream SDK GStreamer pipeline"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch Interactive Live Web Dashboard & Telemetry Monitor"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Web Dashboard"
    )
    parser.add_argument(
        "--no-privacy",
        action="store_true",
        help="Disable privacy-preserving facial blurring"
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Disable MQTT alert publishing"
    )
    parser.add_argument(
        "--broker",
        type=str,
        default="localhost",
        help="MQTT Broker Host"
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=20,
        help="Number of consecutive fallen frames required to confirm fall"
    )
    parser.add_argument(
        "--torso-angle",
        type=float,
        default=38.0,
        help="Torso angle threshold (deg) below which posture is considered horizontal"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the automated test suite across normal activity and fall benchmarks"
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  PIPEGEN TRACK A: REAL-TIME PRIVACY-PRESERVING FALL DETECTION")
    print("  Elderly Care Edge Vision Perception Pipeline (DeepStream + TensorRT)")
    print("=" * 65)

    if args.test:
        import unittest
        from test_data.test_fall_detection import TestFallDetectionPipeline
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFallDetectionPipeline)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        return

    if args.web:
        print(f"[*] Starting Interactive Web Dashboard on http://localhost:{args.port}...")
        from stage5_run.live_dashboard import start_dashboard
        start_dashboard(
            source=args.source,
            port=args.port,
            enable_privacy=not args.no_privacy,
            enable_mqtt=not args.no_mqtt,
            broker_host=args.broker,
            persistence_frames=args.persistence,
            torso_angle=args.torso_angle
        )
        return

    if args.deepstream:
        config_path = ROOT_DIR / "stage4_compiler" / "deepstream_configs" / "deepstream_app_config.txt"
        source_type = "usb" if args.source.isdigit() else "file"
        dev_node = int(args.source) if args.source.isdigit() else 0
        ds_runner = DeepStreamPipelineRunner(
            config_path=str(config_path),
            source_type=source_type,
            device_node=dev_node,
            file_path=args.source if source_type == "file" else None
        )
        success = ds_runner.run()
        if success:
            return
        print("[*] Transitioning to edge perception runner...")

    # Edge Perception Runner
    pipeline = EdgeFallGuardPipeline(
        source=args.source,
        enable_privacy=not args.no_privacy,
        enable_mqtt=not args.no_mqtt,
        mqtt_broker=args.broker,
        show_display=True,
        persistence_frames=args.persistence,
        torso_angle_threshold=args.torso_angle
    )
    pipeline.run_live()

if __name__ == "__main__":
    main()
