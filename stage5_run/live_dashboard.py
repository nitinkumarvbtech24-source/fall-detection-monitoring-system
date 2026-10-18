"""
PipeGen Stage 5: Interactive Web Dashboard & Real-Time Edge Monitor
Provides live MJPEG video streaming, WebSocket telemetry, JSON alert feeds,
privacy toggling, and interactive threshold calibration.
"""

import cv2
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .fallback_runner import EdgeFallGuardPipeline

app = FastAPI(title="PipeGen FallGuard Edge Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance and state
pipeline_instance: Optional[EdgeFallGuardPipeline] = None
pipeline_lock = threading.Lock()
connected_websockets: List[WebSocket] = []
event_history: List[dict] = []
current_source = "0"
is_streaming = True

class ThresholdUpdate(BaseModel):
    persistence_frames: Optional[int] = None
    torso_angle: Optional[float] = None

class SourceUpdate(BaseModel):
    source: str

class PrivacyUpdate(BaseModel):
    enabled: bool

class SafeZoneUpdate(BaseModel):
    enabled: bool
    zones: Optional[List[dict]] = None

def generate_video_stream():
    """Generator yielding multipart MJPEG frames from the live pipeline."""
    global pipeline_instance, is_streaming, current_source
    
    cap = None
    try:
        # Initialize capture
        if current_source.isdigit():
            cap = cv2.VideoCapture(int(current_source), cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(current_source)

        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        prev_time = time.time()
        fps_val = 30.0

        while is_streaming:
            with pipeline_lock:
                if pipeline_instance is None:
                    time.sleep(0.05)
                    continue

            ret, frame = cap.read()
            if not ret:
                # Loop video files continuously
                if not current_source.isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                else:
                    time.sleep(0.05)
                    continue

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps_val = 0.85 * fps_val + 0.15 * current_fps
                with pipeline_lock:
                    pipeline_instance.fps = fps_val

            with pipeline_lock:
                rendered_frame, _ = pipeline_instance.process_frame(frame)

            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', rendered_frame, encode_param)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Maintain smooth pacing
            time.sleep(0.01)

    except Exception as e:
        print(f"[!] Stream generator error: {e}")
    finally:
        if cap:
            cap.release()

@app.get("/video_feed")
def video_feed():
    """Video streaming route for HTML <img> tag."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/telemetry")
def get_telemetry():
    with pipeline_lock:
        if pipeline_instance is None:
            return {}
        
        current_state = "NORMAL"
        if pipeline_instance.analytics.person_states:
            states = [p.state for p in pipeline_instance.analytics.person_states.values()]
            if "FALL CONFIRMED" in states:
                current_state = "FALL CONFIRMED"
            elif "POTENTIAL FALL" in states:
                current_state = "POTENTIAL FALL"
            elif "RESTING / SLEEPING" in states:
                current_state = "RESTING / SLEEPING"

        return {
            "fps": round(pipeline_instance.fps, 1),
            "latency_ms": round(pipeline_instance.latency_ms, 1),
            "potential_falls": pipeline_instance.analytics.total_potential_falls,
            "confirmed_falls": pipeline_instance.analytics.total_confirmed_falls,
            "resting_count": pipeline_instance.analytics.total_resting_events,
            "current_state": current_state,
            "privacy_enabled": pipeline_instance.enable_privacy,
            "current_source": current_source,
            "events_count": len(event_history),
            "mqtt_connected": pipeline_instance.alert_publisher.is_connected if pipeline_instance.alert_publisher else False,
            "persistence_frames": pipeline_instance.analytics.persistence_frames,
            "torso_angle_threshold": pipeline_instance.analytics.torso_angle_fall_threshold,
            "safe_zone_enabled": len(pipeline_instance.analytics.safe_rest_zones) > 0
        }

@app.get("/api/events")
def get_events():
    return JSONResponse(content=event_history)

@app.post("/api/controls/privacy")
def update_privacy(data: PrivacyUpdate):
    with pipeline_lock:
        if pipeline_instance:
            pipeline_instance.set_privacy(data.enabled)
    return {"status": "success", "privacy_enabled": data.enabled}

@app.post("/api/controls/safe_zone")
def update_safe_zone(data: SafeZoneUpdate):
    with pipeline_lock:
        if pipeline_instance:
            if data.enabled:
                zones = data.zones or [{"name": "Bed Area", "x1": 0.05, "y1": 0.40, "x2": 0.95, "y2": 0.95}]
                pipeline_instance.set_safe_rest_zones(zones)
            else:
                pipeline_instance.set_safe_rest_zones([])
    return {"status": "success", "safe_zone_enabled": data.enabled}

@app.post("/api/controls/thresholds")
def update_thresholds(data: ThresholdUpdate):
    with pipeline_lock:
        if pipeline_instance:
            pipeline_instance.set_thresholds(
                persistence_frames=data.persistence_frames,
                torso_angle=data.torso_angle
            )
    return {"status": "success", "updated": data.dict(exclude_unset=True)}

@app.post("/api/controls/source")
def update_source(data: SourceUpdate):
    global current_source
    current_source = data.source
    return {"status": "success", "new_source": current_source}

@app.post("/api/controls/reset")
def reset_counters():
    with pipeline_lock:
        if pipeline_instance:
            pipeline_instance.analytics.total_potential_falls = 0
            pipeline_instance.analytics.total_confirmed_falls = 0
    event_history.clear()
    return {"status": "success", "message": "Counters and events cleared"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # Broadcast telemetry every 500ms
            data = get_telemetry()
            await websocket.send_json({"type": "telemetry", "data": data})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
    except Exception:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

def on_fall_event_callback(event_dict: dict):
    """Called whenever the fall analytics engine confirms a fall."""
    event_history.insert(0, event_dict)
    # Broadcast to all connected WebSockets
    for ws in list(connected_websockets):
        try:
            asyncio.run(ws.send_json({"type": "fall_alert", "data": event_dict}))
        except Exception:
            pass

def start_dashboard(
    source: str = "0",
    port: int = 8080,
    enable_privacy: bool = True,
    enable_mqtt: bool = True,
    broker_host: str = "localhost",
    persistence_frames: int = 20,
    torso_angle: float = 38.0
):
    global pipeline_instance, current_source
    current_source = source

    pipeline_instance = EdgeFallGuardPipeline(
        source=source,
        enable_privacy=enable_privacy,
        enable_mqtt=enable_mqtt,
        mqtt_broker=broker_host,
        show_display=False,
        persistence_frames=persistence_frames,
        torso_angle_threshold=torso_angle
    )

    # Register local event callback
    if pipeline_instance.alert_publisher:
        pipeline_instance.alert_publisher.register_listener(on_fall_event_callback)

    # Serve static assets from web_ui
    static_dir = Path(__file__).parent.parent / "web_ui"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    print(f"\n==================================================================")
    print(f" PipeGen FallGuard Edge Web Dashboard running at: http://localhost:{port}")
    print(f" Active Source: {current_source}")
    print(f"==================================================================\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
