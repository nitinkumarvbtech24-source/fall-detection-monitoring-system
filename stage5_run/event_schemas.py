"""
PipeGen Stage 5: Event Schemas
Defines machine-readable Pydantic schemas for fall detection alerts and telemetry.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class FallMetrics(BaseModel):
    torso_angle_deg: float = Field(..., description="Angle of torso relative to ground plane in degrees")
    aspect_ratio: float = Field(..., description="Bounding box width / height ratio")
    descent_velocity: float = Field(..., description="Downward centroid velocity normalized by person height/sec")
    fallen_duration_sec: float = Field(..., description="Time elapsed in fallen posture in seconds")
    persistence_frames: int = Field(..., description="Consecutive frames person remained in fallen state")

class FallAlertEvent(BaseModel):
    event: str = Field(default="fall_detected", description="Event type identifier")
    person_id: int = Field(..., description="Persistent track ID of the detected individual")
    confidence: float = Field(..., description="Combined fall detection confidence [0.0 - 1.0]")
    timestamp: str = Field(..., description="ISO-8601 formatted event timestamp")
    status: str = Field(default="FALL_CONFIRMED", description="Fall event status: FALL_CONFIRMED")
    location: str = Field(default="Elderly Care Room 101", description="Camera deployment location")
    metrics: FallMetrics = Field(..., description="Detailed posture and kinematic measurements")
    privacy_applied: bool = Field(default=True, description="Indicates if facial privacy blurring was applied")
    screenshot_url: Optional[str] = Field(default=None, description="URL or relative path to captured fall screenshot")
    alarm_state: Optional[str] = Field(default="TRIGGERED", description="Alert sound and emergency indicator status: TRIGGERED, SILENCED")

class PipelineTelemetry(BaseModel):
    timestamp: str
    fps: float
    pipeline_latency_ms: float
    active_people_count: int
    potential_falls_count: int
    confirmed_falls_count: int
    hardware_target: str
    precision_mode: str
