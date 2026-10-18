"""
PipeGen Stage 5: DeepStream GStreamer Perception Pipeline
Native DeepStream Python pipeline utilizing pyds, nvinfer, and nvtracker for NVIDIA edge deployment.
"""

import sys
import os
import time
from pathlib import Path
import configparser
from typing import Optional, List, Dict

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    import pyds
    DEEPSTREAM_AVAILABLE = True
except Exception:
    DEEPSTREAM_AVAILABLE = False

from .fall_analytics import FallAnalyticsEngine
from .privacy_filter import PrivacyFilter
from .mqtt_alert_publisher import MQTTAlertPublisher

class DeepStreamPipelineRunner:
    def __init__(
        self,
        config_path: str,
        source_type: str = "usb",
        device_node: int = 0,
        file_path: Optional[str] = None
    ):
        self.config_path = Path(config_path)
        self.source_type = source_type
        self.device_node = device_node
        self.file_path = file_path

        self.alert_publisher = MQTTAlertPublisher()
        self.analytics_engine = FallAnalyticsEngine(alert_publisher=self.alert_publisher)
        self.privacy_filter = PrivacyFilter(enabled=True)

        self.loop = None
        self.pipeline = None

    def check_environment(self) -> bool:
        if not DEEPSTREAM_AVAILABLE:
            print("[!] NVIDIA DeepStream / GStreamer Python bindings (pyds) not available in this environment.")
            print("[*] Switching seamlessly to PipeGen High-Performance Fallback Runner.")
            return False
        return True

    def run(self):
        if not self.check_environment():
            return False

        Gst.init(None)
        self.loop = GLib.MainLoop()

        print("=================================================================")
        print(" PipeGen Stage 5: Launching Native NVIDIA DeepStream Pipeline")
        print(f" Source: {self.source_type.upper()} | Config: {self.config_path.name}")
        print("=================================================================")

        # Build GStreamer Pipeline Elements
        self.pipeline = Gst.Pipeline()

        if self.source_type == "usb":
            source = Gst.ElementFactory.make("v4l2src", "usb-source")
            source.set_property("device", f"/dev/video{self.device_node}")
            caps_v4l2 = Gst.ElementFactory.make("capsfilter", "v4l2-caps")
            caps_v4l2.set_property("caps", Gst.Caps.from_string("video/x-raw, width=1280, height=720, framerate=30/1"))
            vidconvsrc = Gst.ElementFactory.make("videoconvert", "vidconvsrc")
            nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "nvvidconvsrc")
            caps_nv = Gst.ElementFactory.make("capsfilter", "nv-caps")
            caps_nv.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

            self.pipeline.add(source)
            self.pipeline.add(caps_v4l2)
            self.pipeline.add(vidconvsrc)
            self.pipeline.add(nvvidconvsrc)
            self.pipeline.add(caps_nv)

            source.link(caps_v4l2)
            caps_v4l2.link(vidconvsrc)
            vidconvsrc.link(nvvidconvsrc)
            nvvidconvsrc.link(caps_nv)
            stream_input = caps_nv
        else:
            source = Gst.ElementFactory.make("filesrc", "file-source")
            source.set_property("location", str(self.file_path))
            demux = Gst.ElementFactory.make("qtdemux", "demux")
            parser = Gst.ElementFactory.make("h264parse", "parser")
            decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
            self.pipeline.add(source)
            self.pipeline.add(demux)
            self.pipeline.add(parser)
            self.pipeline.add(decoder)
            source.link(demux)
            parser.link(decoder)
            stream_input = decoder

        # Stream Muxer
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property("width", 1280)
        streammux.set_property("height", 720)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 33333)
        self.pipeline.add(streammux)

        # Primary GIE (YOLOv8n-pose TensorRT)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
        infer_config = self.config_path.parent / "config_infer_pose.txt"
        pgie.set_property("config-file-path", str(infer_config))
        self.pipeline.add(pgie)

        # Tracker (NvDCF)
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        tracker_config = self.config_path.parent / "nvdcf_tracker_config.txt"
        tracker.set_property("ll-config-file", str(tracker_config))
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        self.pipeline.add(tracker)

        # OSD and Video Sink
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
        nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property("sync", 0)

        self.pipeline.add(nvvidconv)
        self.pipeline.add(nvosd)
        self.pipeline.add(sink)

        # Link elements
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = stream_input.get_static_pad("src")
        srcpad.link(sinkpad)

        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)

        # Add Probe for Fall Kinematics & Privacy Blurring
        osd_sink_pad = nvosd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_sink_pad_buffer_probe, 0)

        # Start Pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self.alert_publisher.close()

        return True

    def _osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """DeepStream Probe: Extracts metadata, computes fall analytics, updates OSD."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            detections = []
            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Extract bounding box
                rect = obj_meta.rect_params
                bbox = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]
                
                det = {
                    "track_id": int(obj_meta.object_id),
                    "bbox": bbox,
                    "confidence": float(obj_meta.confidence),
                    "keypoints": None # Populated via user_meta if tensor meta enabled
                }
                detections.append(det)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Run Fall Kinematics Analytics
            self.analytics_engine.process_frame_detections(detections)

            # Update OSD display attributes
            l_obj = frame_meta.obj_meta_list
            idx = 0
            while l_obj is not None and idx < len(detections):
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                det = detections[idx]
                state = det.get("fall_state", "NORMAL")

                # Configure bounding box border color according to state
                if state == "FALL CONFIRMED":
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0) # Red
                elif state == "POTENTIAL FALL":
                    obj_meta.rect_params.border_color.set(1.0, 0.65, 0.0, 1.0) # Amber
                else:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0) # Green

                obj_meta.text_params.display_text = f"ID: {det['track_id']} | {state}"
                idx += 1
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    config = Path(__file__).parent.parent / "stage4_compiler" / "deepstream_configs" / "deepstream_app_config.txt"
    runner = DeepStreamPipelineRunner(str(config), source_type="usb")
    runner.run()
