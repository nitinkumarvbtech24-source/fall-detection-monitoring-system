"""
PipeGen Stage 5: MQTT Alert Publisher
Publishes real-time fall detection alerts and system telemetry over MQTT.
"""

import json
import time
import logging
from typing import Optional, Callable, List
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from .event_schemas import FallAlertEvent, PipelineTelemetry

logger = logging.getLogger("MQTTAlertPublisher")

class MQTTAlertPublisher:
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        alert_topic: str = "elderly_care/alerts/fall",
        telemetry_topic: str = "elderly_care/telemetry/pipeline",
        client_id: str = "pipegen_fall_guard_edge"
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.alert_topic = alert_topic
        self.telemetry_topic = telemetry_topic
        self.client_id = client_id

        self.is_connected = False
        self.client = None
        self.event_history: List[dict] = []
        self.local_listeners: List[Callable[[dict], None]] = []

        self._init_client()

    def _init_client(self):
        if not PAHO_AVAILABLE:
            logger.warning("[!] paho-mqtt not available. Operating in local simulation mode.")
            return

        try:
            self.client = mqtt.Client(
                client_id=self.client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            # Connect asynchronously without blocking pipeline startup
            self.client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            print(f"[*] Attempting MQTT connection to {self.broker_host}:{self.broker_port}...")
        except Exception as e:
            logger.warning(f"[!] MQTT connection initialization error: {e}. Fallback active.")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.is_connected = True
            print(f"[+] MQTT Connected successfully to broker {self.broker_host}:{self.broker_port}")
        else:
            logger.warning(f"[!] MQTT Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self.is_connected = False
        logger.warning("[!] MQTT Disconnected from broker.")

    def register_listener(self, callback: Callable[[dict], None]):
        """Registers a callback for local consumers (e.g. Web Dashboard / UI)."""
        self.local_listeners.append(callback)

    def publish_fall_alert(self, event: FallAlertEvent):
        """Publishes a confirmed fall alert event."""
        payload_dict = event.model_dump() if hasattr(event, "model_dump") else event.dict()
        payload_json = json.dumps(payload_dict, indent=2)

        # Store in event history
        self.event_history.append(payload_dict)

        print("\n" + "=" * 60)
        print("🚨 ACTIONABLE FALL ALERT GENERATED (MQTT & JSON)")
        print(f"Timestamp: {event.timestamp} | Person ID: {event.person_id} | Status: {event.status}")
        print(f"Confidence: {event.confidence:.2f} | Torso: {event.metrics.torso_angle_deg:.1f}° | AR: {event.metrics.aspect_ratio:.2f}")
        print("Payload:")
        print(payload_json)
        print("=" * 60 + "\n")

        # Broadcast to local UI / WebSocket listeners
        for listener in self.local_listeners:
            try:
                listener(payload_dict)
            except Exception as e:
                logger.error(f"Error in local listener: {e}")

        # Publish to MQTT broker if online
        if self.client and self.is_connected:
            try:
                self.client.publish(self.alert_topic, payload_json, qos=1)
                print(f"[+] Alert published to MQTT topic: {self.alert_topic}")
            except Exception as e:
                logger.error(f"[!] Failed to publish MQTT message: {e}")

    def publish_telemetry(self, telemetry: PipelineTelemetry):
        """Publishes pipeline operational telemetry."""
        payload_dict = telemetry.model_dump() if hasattr(telemetry, "model_dump") else telemetry.dict()
        payload_json = json.dumps(payload_dict)

        if self.client and self.is_connected:
            try:
                self.client.publish(self.telemetry_topic, payload_json, qos=0)
            except Exception:
                pass

    def close(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
