/**
 * PipeGen FallGuard Edge Web Application Frontend
 * Handles WebSocket telemetry, real-time alert event stream, threshold calibration, and demo switching.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const liveStreamImg = document.getElementById("live-video-stream");
  const globalStateBadge = document.getElementById("global-state-badge");
  const globalStateText = document.getElementById("global-state-text");
  const hudFps = document.getElementById("hud-fps");
  const hudLatency = document.getElementById("hud-latency");
  const currentSourceLabel = document.getElementById("current-source-label");

  // KPI Elements
  const kpiPeople = document.getElementById("kpi-people");
  const kpiPotential = document.getElementById("kpi-potential");
  const kpiConfirmed = document.getElementById("kpi-confirmed");
  const kpiResting = document.getElementById("kpi-resting");
  const kpiLatency = document.getElementById("kpi-latency");

  // Status Pills
  const privacyPill = document.getElementById("privacy-pill");
  const privacyStatusText = document.getElementById("privacy-status-text");
  const mqttPill = document.getElementById("mqtt-pill");
  const mqttStatusText = document.getElementById("mqtt-status-text");

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

  // Event Feed Elements
  const eventListContainer = document.getElementById("event-list-container");
  const eventCountBadge = document.getElementById("event-count-badge");
  const emptyPlaceholder = document.getElementById("empty-events-placeholder");

  // Modal Elements
  const jsonModal = document.getElementById("json-modal");
  const modalJsonContent = document.getElementById("modal-json-content");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const closeModalBtn2 = document.getElementById("close-modal-btn-2");
  const copyJsonBtn = document.getElementById("copy-json-btn");

  let privacyEnabled = true;
  let receivedEvents = [];

  // -------------------------------------------------------------
  // 1. WebSocket Telemetry & Alert Stream
  // -------------------------------------------------------------
  function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[+] Connected to PipeGen Telemetry WebSocket.");
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "telemetry") {
          updateTelemetryUI(msg.data);
        } else if (msg.type === "fall_alert") {
          handleIncomingFallAlert(msg.data);
        }
      } catch (err) {
        console.error("WS parse error:", err);
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

  // -------------------------------------------------------------
  // 2. UI Updates
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
    } else if (data.current_state === "POTENTIAL FALL") {
      globalStateBadge.className = "state-badge-large state-potential";
      globalStateText.textContent = "POTENTIAL FALL";
    } else if (data.current_state === "RESTING / SLEEPING") {
      globalStateBadge.className = "state-badge-large state-resting";
      globalStateText.textContent = "RESTING / SLEEPING";
    } else {
      globalStateBadge.className = "state-badge-large state-normal";
      globalStateText.textContent = "NORMAL";
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

    // Sync sliders if not currently being dragged
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

    receivedEvents.forEach((ev, idx) => {
      const item = document.createElement("div");
      item.className = "event-item";
      item.innerHTML = `
        <div class="event-top">
          <span class="event-tag">🚨 PERSON ${ev.person_id} FALL CONFIRMED</span>
          <span class="event-time">${ev.timestamp.split("T")[1] || ev.timestamp}</span>
        </div>
        <div class="event-metrics">
          <span>Torso: ${ev.metrics.torso_angle_deg}°</span>
          <span>AR: ${ev.metrics.aspect_ratio}</span>
          <span>Conf: ${(ev.confidence * 100).toFixed(0)}%</span>
        </div>
      `;
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
  // 3. User Controls & API Dispatch
  // -------------------------------------------------------------
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

  // Bed Safe Zone Toggle Button
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

  resetCountersBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/controls/reset", { method: "POST" });
      receivedEvents = [];
      eventCountBadge.textContent = "0 Alerts";
      renderEvents();
    } catch (err) {
      console.error("Reset failed:", err);
    }
  });

  // Source Switcher
  sourceChips.forEach(chip => {
    chip.addEventListener("click", async () => {
      sourceChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const src = chip.getAttribute("data-source");
      const isBedDemo = chip.getAttribute("data-bedzone") === "true";
      currentSourceLabel.textContent = `Source: ${chip.textContent.trim()}`;

      try {
        // If bed demo selected, automatically activate the safe bed zone
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
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Modal Closers & Copy
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

  // Initial Fetch of existing events & Start WebSocket
  fetch("/api/events")
    .then(res => res.json())
    .then(data => {
      if (Array.isArray(data)) {
        receivedEvents = data;
        eventCountBadge.textContent = `${receivedEvents.length} Alerts`;
        renderEvents();
      }
    })
    .catch(() => {});

  initWebSocket();
});
