/**
 * PipeGen FallGuard Edge Web Application Frontend
 * Handles WebSocket telemetry, real-time alert event stream, continuous emergency audio siren,
 * automated fall screenshot capture & inspection, and caregiver silence / switch-off controls.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Video & Overlays
  const liveStreamImg = document.getElementById("live-video-stream");
  const videoWrapper = document.getElementById("video-wrapper");
  const inVideoAlertOverlay = document.getElementById("in-video-alert-overlay");
  const inVideoSilenceBtn = document.getElementById("in-video-silence-btn");

  // Global State & HUD
  const globalStateBadge = document.getElementById("global-state-badge");
  const globalStateText = document.getElementById("global-state-text");
  const hudFps = document.getElementById("hud-fps");
  const hudLatency = document.getElementById("hud-latency");
  const currentSourceLabel = document.getElementById("current-source-label");

  // Emergency Persistent Banner Elements
  const emergencyGlobalBar = document.getElementById("emergency-global-bar");
  const emergencySubline = document.getElementById("emergency-subline");
  const switchOffAlarmBarBtn = document.getElementById("switch-off-alarm-bar-btn");
  const viewScreenshotBarBtn = document.getElementById("view-screenshot-bar-btn");

  // Emergency Modal Elements
  const emergencyModal = document.getElementById("emergency-modal");
  const minimizeEmergencyModalBtn = document.getElementById("minimize-emergency-modal-btn");
  const switchOffAlarmBtn = document.getElementById("switch-off-alarm-btn");
  const emergencyScreenshotImg = document.getElementById("emergency-screenshot-img");
  const screenshotLoading = document.getElementById("screenshot-loading");
  const screenshotTimestampLabel = document.getElementById("screenshot-timestamp-label");
  const downloadScreenshotBtn = document.getElementById("download-screenshot-btn");
  const expandScreenshotBtn = document.getElementById("expand-screenshot-btn");
  const alertPersonId = document.getElementById("alert-person-id");
  const alertConfidence = document.getElementById("alert-confidence");
  const alertTorso = document.getElementById("alert-torso");

  // Fullscreen Image Viewer Modal
  const imageViewerModal = document.getElementById("image-viewer-modal");
  const viewerImg = document.getElementById("viewer-img");
  const viewerTitle = document.getElementById("viewer-title");
  const viewerDownloadLink = document.getElementById("viewer-download-link");
  const closeViewerBtn = document.getElementById("close-viewer-btn");
  const closeViewerBtn2 = document.getElementById("close-viewer-btn-2");

  // KPI Elements
  const kpiPeople = document.getElementById("kpi-people");
  const kpiPotential = document.getElementById("kpi-potential");
  const kpiConfirmed = document.getElementById("kpi-confirmed");
  const kpiResting = document.getElementById("kpi-resting");
  const kpiLatency = document.getElementById("kpi-latency");

  // Status Pills & Controls
  const privacyPill = document.getElementById("privacy-pill");
  const privacyStatusText = document.getElementById("privacy-status-text");
  const mqttPill = document.getElementById("mqtt-pill");
  const mqttStatusText = document.getElementById("mqtt-status-text");
  const alarmPill = document.getElementById("alarm-pill");
  const alarmStatusText = document.getElementById("alarm-status-text");
  const testSirenBtn = document.getElementById("test-siren-btn");

  // Buttons & Controls
  const togglePrivacyBtn = document.getElementById("toggle-privacy-btn");
  const toggleBedzoneBtn = document.getElementById("toggle-bedzone-btn");
  const bedzoneBtnText = document.getElementById("bedzone-btn-text");
  const resetCountersBtn = document.getElementById("reset-counters-btn");
  const sourceChips = document.querySelectorAll(".btn-chip");
  const sliderPersistence = document.getElementById("slider-persistence");
  const valPersistence = document.getElementById("val-persistence");
  const sliderAngle = document.getElementById("slider-angle");
  const valAngle = document.getElementById("val-angle");

  // Event Feed & Snapshots Gallery
  const eventListContainer = document.getElementById("event-list-container");
  const eventCountBadge = document.getElementById("event-count-badge");
  const emptyPlaceholder = document.getElementById("empty-events-placeholder");
  const snapshotsGalleryGrid = document.getElementById("snapshots-gallery-grid");
  const emptySnapshotsMsg = document.getElementById("empty-snapshots-msg");
  const snapshotsTabCount = document.getElementById("snapshots-tab-count");
  const refreshSnapshotsBtn = document.getElementById("refresh-snapshots-btn");

  // JSON Modal Elements
  const jsonModal = document.getElementById("json-modal");
  const modalJsonContent = document.getElementById("modal-json-content");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const closeModalBtn2 = document.getElementById("close-modal-btn-2");
  const copyJsonBtn = document.getElementById("copy-json-btn");

  // State
  let privacyEnabled = true;
  let receivedEvents = [];
  let alarmActive = false;
  let alarmSilenced = false;
  let currentActiveScreenshotUrl = null;

  // -------------------------------------------------------------
  // 1. Web Audio API Emergency Siren Engine
  // -------------------------------------------------------------
  const EmergencyAudioEngine = {
    ctx: null,
    osc: null,
    gain: null,
    isPlaying: false,
    intervalId: null,

    init() {
      if (!this.ctx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
          this.ctx = new AudioContextClass();
        }
      }
      if (this.ctx && this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
      }
    },

    startSiren() {
      this.init();
      if (!this.ctx) return;
      if (this.isPlaying) return;

      if (this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
      }

      this.isPlaying = true;
      let pitchToggle = false;

      // Primary Oscillator
      this.osc = this.ctx.createOscillator();
      this.osc.type = "sawtooth";
      this.osc.frequency.setValueAtTime(880, this.ctx.currentTime); // High alarm pitch (A5)

      this.gain = this.ctx.createGain();
      this.gain.gain.setValueAtTime(0.22, this.ctx.currentTime);

      this.osc.connect(this.gain);
      this.gain.connect(this.ctx.destination);
      this.osc.start();

      // Penetrating dual-tone alternating emergency siren (880Hz / 660Hz every 320ms)
      this.intervalId = setInterval(() => {
        if (!this.isPlaying || !this.ctx) return;
        const now = this.ctx.currentTime;
        pitchToggle = !pitchToggle;
        const targetFreq = pitchToggle ? 660 : 920;
        this.osc.frequency.exponentialRampToValueAtTime(targetFreq, now + 0.12);
      }, 320);

      console.log("[🚨 EMERGENCY SIREN ACTIVE] Continuous alarm sound playing until switched off.");
    },

    stopSiren() {
      if (!this.isPlaying) return;
      this.isPlaying = false;

      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }

      if (this.gain && this.ctx) {
        try {
          this.gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.1);
        } catch (e) {}
      }

      setTimeout(() => {
        if (this.osc) {
          try {
            this.osc.stop();
            this.osc.disconnect();
          } catch (e) {}
          this.osc = null;
        }
      }, 120);

      console.log("[🔕 SIREN MUTED] Alarm sound deactivated by caregiver.");
      this.playAcknowledgmentChime();
    },

    playAcknowledgmentChime() {
      this.init();
      if (!this.ctx) return;
      try {
        const now = this.ctx.currentTime;
        const chimeOsc = this.ctx.createOscillator();
        const chimeGain = this.ctx.createGain();
        chimeOsc.type = "sine";
        chimeGain.gain.setValueAtTime(0.20, now);
        chimeGain.gain.exponentialRampToValueAtTime(0.001, now + 0.7);

        // Pleasant two-tone ascending resolution chime (C5 -> E5)
        chimeOsc.frequency.setValueAtTime(523.25, now);
        chimeOsc.frequency.setValueAtTime(659.25, now + 0.22);

        chimeOsc.connect(chimeGain);
        chimeGain.connect(this.ctx.destination);
        chimeOsc.start(now);
        chimeOsc.stop(now + 0.75);
      } catch (e) {}
    },

    testSiren() {
      this.init();
      if (!this.ctx) {
        alert("Web Audio is not supported in this browser.");
        return;
      }
      if (this.isPlaying) {
        this.stopSiren();
        return;
      }
      this.startSiren();
      setTimeout(() => {
        if (this.isPlaying && !alarmActive) {
          this.stopSiren();
        }
      }, 1600);
    }
  };

  // Pre-authorize audio context on any user click
  window.addEventListener("click", () => EmergencyAudioEngine.init(), { once: true });
  window.addEventListener("keydown", () => EmergencyAudioEngine.init(), { once: true });

  // -------------------------------------------------------------
  // 2. Client-Side Video Snapshot Fallback
  // -------------------------------------------------------------
  function captureLiveSnapshotCanvas() {
    try {
      const canvas = document.createElement("canvas");
      const w = liveStreamImg.naturalWidth || liveStreamImg.clientWidth || 1280;
      const h = liveStreamImg.naturalHeight || liveStreamImg.clientHeight || 720;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(liveStreamImg, 0, 0, w, h);

      // Render Emergency Header Banner with Alert Symbol directly on client snapshot
      ctx.fillStyle = "rgba(180, 15, 25, 0.88)";
      ctx.fillRect(0, 0, w, 52);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 22px 'Outfit', sans-serif";
      ctx.fillText("⚠️ EMERGENCY FALL DETECTED — PIPEGEN VERIFIED SNAPSHOT", 20, 34);
      ctx.font = "15px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#FFCDD2";
      ctx.fillText(new Date().toISOString().replace("T", " ").substring(0, 19), Math.max(10, w - 240), 34);

      return canvas.toDataURL("image/jpeg", 0.90);
    } catch (e) {
      console.warn("Client canvas snapshot capture error:", e);
      return null;
    }
  }

  // -------------------------------------------------------------
  // 3. Emergency Alarm Trigger & Silence (UNTIL SOMEONE SWITCHES IT OFF)
  // -------------------------------------------------------------
  function triggerEmergencyAlarm(alertData = {}) {
    // If already sounding, do not restart
    if (alarmActive) return;

    alarmActive = true;
    alarmSilenced = false;

    // 1. Play continuous emergency siren
    EmergencyAudioEngine.startSiren();

    // 2. Update Nav Status Pill
    if (alarmPill && alarmStatusText) {
      alarmPill.className = "status-pill status-alarm alarm-ringing";
      alarmStatusText.textContent = "🚨 SIREN SOUNDING";
    }

    const personId = alertData.person_id || 1;
    const confVal = alertData.confidence ? Math.round(alertData.confidence * 100) : 96;
    const torsoVal = alertData.metrics ? alertData.metrics.torso_angle_deg : 18.5;
    const nowTimeStr = alertData.timestamp ? (alertData.timestamp.replace("T", " ").split(".")[0]) : new Date().toLocaleTimeString();

    // 3. Show Top Persistent Emergency Banner (Pinned until switched off)
    if (emergencyGlobalBar) {
      emergencyGlobalBar.style.display = "block";
      if (emergencySubline) {
        emergencySubline.textContent = `Continuous audible siren is active for Person #${personId}. Caregiver action required.`;
      }
    }

    // 4. In-Video Red Strobe & In-Video Alert Overlay
    if (videoWrapper) videoWrapper.classList.add("alarm-active-border");
    if (inVideoAlertOverlay) inVideoAlertOverlay.style.display = "flex";

    // 5. Capture & Show Fall Screenshot
    const clientSnapshotUri = captureLiveSnapshotCanvas();
    const finalScreenshotUrl = alertData.screenshot_url || clientSnapshotUri || `/snapshots/fall_alert.jpg?t=${Date.now()}`;
    currentActiveScreenshotUrl = finalScreenshotUrl;

    if (screenshotTimestampLabel) screenshotTimestampLabel.textContent = nowTimeStr;
    if (alertPersonId) alertPersonId.textContent = `Person #${personId}`;
    if (alertConfidence) alertConfidence.textContent = `${confVal}%`;
    if (alertTorso) alertTorso.textContent = `${torsoVal}° (Horizontal)`;

    if (emergencyScreenshotImg) {
      if (screenshotLoading) screenshotLoading.style.display = "flex";
      emergencyScreenshotImg.style.display = "none";

      emergencyScreenshotImg.onload = () => {
        if (screenshotLoading) screenshotLoading.style.display = "none";
        emergencyScreenshotImg.style.display = "block";
      };

      emergencyScreenshotImg.onerror = () => {
        if (clientSnapshotUri && emergencyScreenshotImg.src !== clientSnapshotUri) {
          emergencyScreenshotImg.src = clientSnapshotUri;
        } else {
          if (screenshotLoading) screenshotLoading.style.display = "none";
          emergencyScreenshotImg.style.display = "block";
        }
      };

      emergencyScreenshotImg.src = finalScreenshotUrl;
    }

    if (downloadScreenshotBtn) {
      downloadScreenshotBtn.href = finalScreenshotUrl;
      downloadScreenshotBtn.download = `fall_verification_person_${personId}_${Date.now()}.jpg`;
    }

    // Open Emergency Modal
    if (emergencyModal) {
      emergencyModal.style.display = "flex";
    }

    // Refresh Snapshots Gallery
    setTimeout(fetchSnapshotsGallery, 500);
  }

  async function silenceEmergencyAlarm() {
    if (!alarmActive && alarmSilenced) return;

    alarmActive = false;
    alarmSilenced = true;

    // 1. Immediately switch off the siren sound
    EmergencyAudioEngine.stopSiren();

    // 2. Hide Emergency Modal
    if (emergencyModal) emergencyModal.style.display = "none";

    // 3. Hide In-Video Overlay & Border Strobe
    if (inVideoAlertOverlay) inVideoAlertOverlay.style.display = "none";
    if (videoWrapper) videoWrapper.classList.remove("alarm-active-border");

    // 4. Update Nav Pill
    if (alarmPill && alarmStatusText) {
      alarmPill.className = "status-pill status-alarm";
      alarmStatusText.textContent = "🔕 ALARM SILENCED";
    }

    // 5. Update Top Persistent Bar to Acknowledged state and fade after a delay
    if (emergencyGlobalBar) {
      if (emergencySubline) {
        emergencySubline.textContent = `Alarm siren was switched off & acknowledged by caregiver at ${new Date().toLocaleTimeString()}.`;
      }
      setTimeout(() => {
        if (!alarmActive && emergencyGlobalBar) {
          emergencyGlobalBar.style.display = "none";
        }
      }, 5500);
    }

    // 6. Notify backend to silence alarm
    try {
      await fetch("/api/alerts/silence", { method: "POST" });
    } catch (err) {
      console.warn("Could not notify backend of alarm silence:", err);
    }
  }

  // -------------------------------------------------------------
  // 4. Snapshots Gallery & Fullscreen Viewer
  // -------------------------------------------------------------
  async function fetchSnapshotsGallery() {
    if (!snapshotsGalleryGrid) return;
    try {
      const res = await fetch("/api/snapshots");
      const files = await res.json();

      if (snapshotsTabCount) {
        snapshotsTabCount.textContent = files.length;
      }

      if (!Array.isArray(files) || files.length === 0) {
        if (emptySnapshotsMsg) emptySnapshotsMsg.style.display = "flex";
        return;
      }

      if (emptySnapshotsMsg) emptySnapshotsMsg.style.display = "none";

      // Clear existing cards
      const existingCards = snapshotsGalleryGrid.querySelectorAll(".snapshot-card");
      existingCards.forEach(c => c.remove());

      files.forEach(file => {
        const card = document.createElement("div");
        card.className = "snapshot-card";
        card.innerHTML = `
          <div class="snapshot-thumb-wrap" title="Click to inspect snapshot in full resolution">
            <img src="${file.url}" alt="${file.filename}" class="snapshot-thumb" loading="lazy">
            <span class="snapshot-thumb-overlay">📸 VERIFIED FALL</span>
          </div>
          <div class="snapshot-info">
            <div class="snapshot-name-date">
              <span class="snapshot-date">${file.timestamp}</span>
              <span class="snapshot-size">${file.size_kb} KB</span>
            </div>
            <a href="${file.url}" download="${file.filename}" class="btn btn-secondary btn-sm" title="Download Screenshot">
              💾 Download
            </a>
          </div>
        `;

        card.querySelector(".snapshot-thumb-wrap").addEventListener("click", () => {
          openImageViewer(file.url, file.filename, file.timestamp);
        });

        snapshotsGalleryGrid.appendChild(card);
      });
    } catch (err) {
      console.warn("Snapshots gallery fetch error:", err);
    }
  }

  function openImageViewer(url, filename, timestamp) {
    if (!imageViewerModal || !viewerImg) return;
    viewerImg.src = url;
    if (viewerTitle) viewerTitle.textContent = `📸 Fall Verification Screenshot — ${timestamp || filename}`;
    if (viewerDownloadLink) {
      viewerDownloadLink.href = url;
      viewerDownloadLink.download = filename || "fall_snapshot.jpg";
    }
    imageViewerModal.style.display = "flex";
  }

  function closeImageViewer() {
    if (imageViewerModal) imageViewerModal.style.display = "none";
  }

  // -------------------------------------------------------------
  // 5. WebSocket Telemetry & Alert Stream
  // -------------------------------------------------------------
  function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[+] Connected to PipeGen Telemetry & Alert WebSocket.");
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "telemetry") {
          updateTelemetryUI(msg.data);
        } else if (msg.type === "fall_alert") {
          handleIncomingFallAlert(msg.data);
          triggerEmergencyAlarm(msg.data);
        } else if (msg.type === "alarm_silenced") {
          if (alarmActive) {
            silenceEmergencyAlarm();
          }
        } else if (msg.type === "alarm_reset") {
          silenceEmergencyAlarm();
          receivedEvents = [];
          if (eventCountBadge) eventCountBadge.textContent = "0 Alerts";
          renderEvents();
          fetchSnapshotsGallery();
        }
      } catch (err) {
        console.error("WebSocket message parse error:", err);
      }
    };

    socket.onclose = () => {
      console.warn("[!] WebSocket disconnected. Reconnecting in 2s...");
      setTimeout(initWebSocket, 2000);
    };

    socket.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }

  // Dual Redundancy: HTTP Polling Fallback (Every 600ms)
  setInterval(async () => {
    try {
      const res = await fetch("/api/telemetry");
      if (res.ok) {
        const data = await res.json();
        updateTelemetryUI(data);
      }
    } catch (e) {}
  }, 600);

  // -------------------------------------------------------------
  // 6. UI Updates
  // -------------------------------------------------------------
  function updateTelemetryUI(data) {
    if (!data) return;

    hudFps.textContent = `${data.fps} FPS`;
    hudLatency.textContent = `${data.latency_ms} ms`;
    kpiLatency.textContent = `${data.latency_ms} ms`;
    kpiPotential.textContent = data.potential_falls;
    kpiConfirmed.textContent = data.confirmed_falls;
    if (kpiResting) kpiResting.textContent = data.resting_count || 0;
    if (bedzoneBtnText) {
      bedzoneBtnText.textContent = data.safe_zone_enabled ? "Bed Zone: ON" : "Bed Zone: OFF";
    }

    // Update Global State Badge
    if (data.current_state === "FALL CONFIRMED") {
      globalStateBadge.className = "state-badge-large state-confirmed";
      globalStateText.textContent = "FALL CONFIRMED";

      // If alarm is not active and not silenced yet, trigger alarm!
      if (!alarmActive && !alarmSilenced) {
        triggerEmergencyAlarm({
          person_id: 1,
          confidence: 0.96,
          timestamp: data.latest_fall_time || new Date().toISOString(),
          screenshot_url: data.latest_screenshot
        });
      }
    } else if (data.current_state === "POTENTIAL FALL") {
      globalStateBadge.className = "state-badge-large state-potential";
      globalStateText.textContent = "POTENTIAL FALL";
    } else if (data.current_state === "RESTING / SLEEPING") {
      globalStateBadge.className = "state-badge-large state-resting";
      globalStateText.textContent = "RESTING / SLEEPING";
    } else {
      globalStateBadge.className = "state-badge-large state-normal";
      globalStateText.textContent = "NORMAL";

      // When person has recovered back to NORMAL, reset the alarm silenced flag
      // so any future genuine fall will sound the siren again!
      if (alarmSilenced && !alarmActive) {
        alarmSilenced = false;
        if (alarmPill && alarmStatusText) {
          alarmPill.className = "status-pill status-alarm";
          alarmStatusText.textContent = "SIREN ARMED";
        }
      }
    }

    // Privacy Status
    privacyEnabled = data.privacy_enabled;
    if (privacyEnabled) {
      privacyStatusText.textContent = "PRIVACY BLUR ON";
      privacyPill.style.color = "var(--accent-green)";
    } else {
      privacyStatusText.textContent = "PRIVACY BLUR OFF";
      privacyPill.style.color = "var(--text-muted)";
    }

    // MQTT Status
    if (data.mqtt_connected) {
      mqttStatusText.textContent = "MQTT CONNECTED";
      mqttPill.querySelector(".indicator-dot").style.background = "var(--accent-green)";
    } else {
      mqttStatusText.textContent = "MQTT STANDBY";
      mqttPill.querySelector(".indicator-dot").style.background = "var(--accent-amber)";
    }

    // Sync sliders if not actively being dragged
    if (!sliderPersistence.matches(":active")) {
      sliderPersistence.value = data.persistence_frames;
      valPersistence.textContent = `${data.persistence_frames} frames (~${(data.persistence_frames / 25).toFixed(1)}s)`;
    }
    if (!sliderAngle.matches(":active")) {
      sliderAngle.value = data.torso_angle_threshold;
      valAngle.textContent = `${data.torso_angle_threshold}°`;
    }
  }

  function handleIncomingFallAlert(alertData) {
    if (alertData.screenshot_url) {
      currentActiveScreenshotUrl = alertData.screenshot_url;
    }
    receivedEvents.unshift(alertData);
    eventCountBadge.textContent = `${receivedEvents.length} Alerts`;
    renderEvents();
  }

  function renderEvents() {
    if (receivedEvents.length === 0) {
      emptyPlaceholder.style.display = "flex";
      return;
    }
    emptyPlaceholder.style.display = "none";

    // Clear existing
    const items = eventListContainer.querySelectorAll(".event-item");
    items.forEach(el => el.remove());

    receivedEvents.forEach((ev) => {
      const item = document.createElement("div");
      item.className = "event-item";

      const torso = ev.metrics ? ev.metrics.torso_angle_deg : '--';
      const ar = ev.metrics ? ev.metrics.aspect_ratio : '--';
      const conf = Math.round((ev.confidence || 0.85) * 100);
      const timeStr = ev.timestamp ? (ev.timestamp.split("T")[1] || ev.timestamp) : '--:--:--';
      
      // Determine screenshot URL for this specific event
      const snapshotUrl = ev.screenshot_url || currentActiveScreenshotUrl || `/snapshots/fall_alert_person_${ev.person_id}.jpg`;

      item.innerHTML = `
        <div class="event-top">
          <span class="event-tag">🚨 PERSON ${ev.person_id} FALL CONFIRMED</span>
          <span class="event-time">${timeStr}</span>
        </div>
        <div class="event-content-row">
          <div class="event-snapshot-preview" title="Click to inspect fall verification snapshot">
            <img src="${snapshotUrl}" alt="Fall Snapshot Person ${ev.person_id}" class="event-snapshot-img" loading="lazy" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'140\\' height=\\'78\\' viewBox=\\'0 0 140 78\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%23121620\\'/><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%23ff5252\\' font-size=\\'12\\'>🚨 SNAPSHOT</text></svg>';">
            <span class="event-snapshot-badge">📸 SNAPSHOT</span>
          </div>
          <div class="event-details-col">
            <div class="event-metrics-grid">
              <div class="event-metric-chip">
                <span class="em-lbl">Torso</span>
                <strong class="em-val">${torso}°</strong>
              </div>
              <div class="event-metric-chip">
                <span class="em-lbl">Aspect Ratio</span>
                <strong class="em-val">${ar}</strong>
              </div>
              <div class="event-metric-chip">
                <span class="em-lbl">Confidence</span>
                <strong class="em-val em-alert">${conf}%</strong>
              </div>
            </div>
            <div class="event-actions-row">
              <button class="btn btn-outline btn-xs btn-inspect-snap" title="Inspect full quality snapshot">
                🔍 View Fullscreen
              </button>
              <a href="${snapshotUrl}" download="fall_alert_person_${ev.person_id}.jpg" class="btn btn-secondary btn-xs btn-download-snap" title="Download Screenshot">
                💾 Download
              </a>
              <button class="btn btn-outline btn-xs btn-view-json" title="View JSON payload">
                { } JSON
              </button>
            </div>
          </div>
        </div>
      `;

      // Event listener for thumbnail & View Snapshot button
      const preview = item.querySelector(".event-snapshot-preview");
      const inspectBtn = item.querySelector(".btn-inspect-snap");
      const jsonBtn = item.querySelector(".btn-view-json");

      if (preview) {
        preview.addEventListener("click", (e) => {
          e.stopPropagation();
          openImageViewer(snapshotUrl, `fall_alert_person_${ev.person_id}.jpg`, timeStr);
        });
      }
      if (inspectBtn) {
        inspectBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          openImageViewer(snapshotUrl, `fall_alert_person_${ev.person_id}.jpg`, timeStr);
        });
      }
      if (jsonBtn) {
        jsonBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          openJsonModal(ev);
        });
      }

      // Clicking rest of item opens JSON modal
      item.addEventListener("click", () => openJsonModal(ev));

      eventListContainer.appendChild(item);
    });
  }

  function openJsonModal(ev) {
    modalJsonContent.textContent = JSON.stringify(ev, null, 2);
    jsonModal.style.display = "flex";
  }

  function closeModal() {
    jsonModal.style.display = "none";
  }

  // -------------------------------------------------------------
  // 7. Event Listeners & Controls
  // -------------------------------------------------------------

  // Siren Test Button
  if (testSirenBtn) {
    testSirenBtn.addEventListener("click", () => {
      EmergencyAudioEngine.testSiren();
    });
  }

  // SWITCH OFF ALARM Buttons (Caregiver Silence Action)
  if (switchOffAlarmBtn) {
    switchOffAlarmBtn.addEventListener("click", silenceEmergencyAlarm);
  }
  if (switchOffAlarmBarBtn) {
    switchOffAlarmBarBtn.addEventListener("click", silenceEmergencyAlarm);
  }
  if (inVideoSilenceBtn) {
    inVideoSilenceBtn.addEventListener("click", silenceEmergencyAlarm);
  }

  // Minimize Modal (Alarm keeps sounding until switched off!)
  if (minimizeEmergencyModalBtn) {
    minimizeEmergencyModalBtn.addEventListener("click", () => {
      if (emergencyModal) emergencyModal.style.display = "none";
      // Ensure the top persistent bar remains visible so caregiver doesn't forget
      if (alarmActive && emergencyGlobalBar) {
        emergencyGlobalBar.style.display = "block";
      }
    });
  }

  // View Screenshot button on top persistent bar
  if (viewScreenshotBarBtn) {
    viewScreenshotBarBtn.addEventListener("click", () => {
      if (emergencyModal) emergencyModal.style.display = "flex";
    });
  }

  // Expand screenshot to fullscreen viewer
  if (expandScreenshotBtn) {
    expandScreenshotBtn.addEventListener("click", () => {
      const src = emergencyScreenshotImg ? emergencyScreenshotImg.src : currentActiveScreenshotUrl;
      if (src) {
        openImageViewer(src, "fall_verified_snapshot.jpg", screenshotTimestampLabel.textContent);
      }
    });
  }

  // Fullscreen Viewer Closers
  if (closeViewerBtn) closeViewerBtn.addEventListener("click", closeImageViewer);
  if (closeViewerBtn2) closeViewerBtn2.addEventListener("click", closeImageViewer);
  if (imageViewerModal) {
    imageViewerModal.addEventListener("click", (e) => {
      if (e.target === imageViewerModal) closeImageViewer();
    });
  }

  // Refresh Snapshots Gallery Button
  if (refreshSnapshotsBtn) {
    refreshSnapshotsBtn.addEventListener("click", fetchSnapshotsGallery);
  }

  // Keyboard shortcut to silence alarm (Escape or 's')
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" || e.key === "s" || e.key === "S") {
      if (imageViewerModal && imageViewerModal.style.display === "flex") {
        closeImageViewer();
        return;
      }
      if (jsonModal && jsonModal.style.display === "flex") {
        closeModal();
        return;
      }
      if (alarmActive) {
        silenceEmergencyAlarm();
      }
    }
  });

  // Privacy Toggle
  togglePrivacyBtn.addEventListener("click", async () => {
    try {
      const newState = !privacyEnabled;
      const res = await fetch("/api/controls/privacy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: newState })
      });
      const data = await res.json();
      privacyEnabled = data.privacy_enabled;
    } catch (err) {
      console.error("Privacy toggle failed:", err);
    }
  });

  // Safe Bed Zone Toggle
  let safeZoneActive = false;
  toggleBedzoneBtn.addEventListener("click", async () => {
    safeZoneActive = !safeZoneActive;
    try {
      await fetch("/api/controls/safe_zone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: safeZoneActive })
      });
      if (bedzoneBtnText) {
        bedzoneBtnText.textContent = safeZoneActive ? "Bed Zone: ON" : "Bed Zone: OFF";
      }
    } catch (err) {
      console.error("Safe zone toggle failed:", err);
    }
  });

  // Reset Counters & Alarms
  resetCountersBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/controls/reset", { method: "POST" });
      silenceEmergencyAlarm();
      receivedEvents = [];
      eventCountBadge.textContent = "0 Alerts";
      renderEvents();
      fetchSnapshotsGallery();
    } catch (err) {
      console.error("Reset failed:", err);
    }
  });

  // Source Switcher Chips
  sourceChips.forEach(chip => {
    chip.addEventListener("click", async () => {
      sourceChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const src = chip.getAttribute("data-source");
      const isBedDemo = chip.getAttribute("data-bedzone") === "true";
      currentSourceLabel.textContent = `Source: ${chip.textContent.trim()}`;

      try {
        if (isBedDemo) {
          safeZoneActive = true;
          await fetch("/api/controls/safe_zone", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled: true })
          });
          if (bedzoneBtnText) bedzoneBtnText.textContent = "Bed Zone: ON";
        }

        await fetch("/api/controls/source", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: src })
        });

        // Reload stream image
        liveStreamImg.src = `/video_feed?t=${Date.now()}`;
      } catch (err) {
        console.error("Source update failed:", err);
      }
    });
  });

  // Sliders
  sliderPersistence.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    valPersistence.textContent = `${val} frames (~${(val / 25).toFixed(1)}s)`;
  });

  sliderPersistence.addEventListener("change", async (e) => {
    await fetch("/api/controls/thresholds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persistence_frames: parseInt(e.target.value) })
    });
  });

  sliderAngle.addEventListener("input", (e) => {
    valAngle.textContent = `${parseFloat(e.target.value).toFixed(1)}°`;
  });

  sliderAngle.addEventListener("change", async (e) => {
    await fetch("/api/controls/thresholds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ torso_angle: parseFloat(e.target.value) })
    });
  });

  // Stage Explorer Tabs
  const stageTabs = document.querySelectorAll(".stage-tab");
  stageTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      stageTabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const targetPane = document.getElementById(tab.getAttribute("data-tab"));
      if (targetPane) {
        targetPane.classList.add("active");
        if (tab.getAttribute("data-tab") === "tab-snapshots") {
          fetchSnapshotsGallery();
        }
      }
    });
  });

  // Modal Closers & Copy JSON
  closeModalBtn.addEventListener("click", closeModal);
  closeModalBtn2.addEventListener("click", closeModal);
  jsonModal.addEventListener("click", (e) => {
    if (e.target === jsonModal) closeModal();
  });

  copyJsonBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(modalJsonContent.textContent);
    copyJsonBtn.textContent = "Copied!";
    setTimeout(() => { copyJsonBtn.textContent = "Copy JSON"; }, 2000);
  });

  // Initial Fetch of existing events, snapshots, and alert status
  fetch("/api/events")
    .then(res => res.json())
    .then(data => {
      if (Array.isArray(data)) {
        receivedEvents = data;
        eventCountBadge.textContent = `${receivedEvents.length} Alerts`;
        
        // Ensure any existing event without screenshot_url gets paired with latest snapshots
        fetch("/api/snapshots")
          .then(r => r.json())
          .then(snaps => {
            if (Array.isArray(snaps) && snaps.length > 0) {
              receivedEvents.forEach((ev, idx) => {
                if (!ev.screenshot_url && snaps[idx]) {
                  ev.screenshot_url = snaps[idx].url;
                }
              });
            }
            renderEvents();
          })
          .catch(() => {
            renderEvents();
          });
      }
    })
    .catch(() => {});

  fetch("/api/alert_status")
    .then(res => res.json())
    .then(status => {
      if (status && status.active) {
        triggerEmergencyAlarm({
          person_id: status.person_id || 1,
          confidence: status.confidence || 0.95,
          timestamp: status.last_fall_time,
          screenshot_url: status.screenshot_url
        });
      }
    })
    .catch(() => {});

  fetchSnapshotsGallery();
  initWebSocket();
});
