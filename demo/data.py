"""
demo/data.py — Realistic mock data engine for the portfolio demo.

Generates:
  - A stream of DetectionFrame objects simulating real YOLO output
  - Pre-seeded alert history (last 48 hours)
  - Synthetic annotated frames rendered with OpenCV
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

VIOLENCE_CLASSES  = ["Fighting", "Weapon", "Aggression"]
ALL_CLASSES       = ["Normal", "Fighting", "Weapon", "Aggression"]

CLASS_COLORS = {
    "Fighting":   (0,   50,  220),
    "Weapon":     (0,   0,   180),
    "Aggression": (20,  80,  200),
    "Normal":     (30,  160, 30),
}

# Scenario timeline: defines what the "camera" sees over a 60-second loop
# Each entry: (start_sec, end_sec, class, confidence_range)
SCENARIO: list[tuple] = [
    (0,   8,  "Normal",     (0.80, 0.95)),
    (8,   9,  "Aggression", (0.56, 0.65)),
    (9,   11, "Aggression", (0.70, 0.82)),
    (11,  14, "Fighting",   (0.78, 0.91)),
    (14,  15, "Fighting",   (0.88, 0.96)),   # ← alert fires here
    (15,  22, "Normal",     (0.82, 0.94)),
    (22,  24, "Weapon",     (0.60, 0.72)),
    (24,  27, "Weapon",     (0.75, 0.89)),   # ← alert fires here
    (27,  35, "Normal",     (0.85, 0.97)),
    (35,  37, "Aggression", (0.55, 0.68)),
    (37,  39, "Fighting",   (0.72, 0.84)),
    (39,  42, "Fighting",   (0.83, 0.93)),   # ← alert fires here
    (42,  60, "Normal",     (0.88, 0.98)),
]

SCENARIO_DURATION = 60   # seconds before loop restarts

# Alert times within the scenario (seconds)
ALERT_SECONDS = {14, 26, 41}


@dataclass
class DetectionFrame:
    """Simulated output for a single processed frame."""
    frame_id:       int
    scenario_sec:   float
    detected_class: str
    confidence:     float
    is_violent:     bool
    alert_triggered: bool
    bbox:           tuple   # x1, y1, x2, y2  (relative to 640×360)


@dataclass
class AlertRecord:
    id:              int
    timestamp:       datetime
    detected_class:  str
    confidence:      float
    location:        str
    email_sent:      bool
    whatsapp_sent:   bool


# ── Frame Generator ───────────────────────────────────────────────────────────

class DemoStream:
    """
    Simulates a live detection feed.
    Call .next_frame() at ~20 FPS.
    """

    def __init__(self, fps: int = 20):
        self._fps      = fps
        self._frame_id = 0
        self._t0       = time.time()
        self._last_alert_sec: float = -999

    def next_frame(self) -> DetectionFrame:
        self._frame_id += 1
        elapsed   = (time.time() - self._t0) % SCENARIO_DURATION
        scene     = self._get_scene(elapsed)
        cls       = scene[2]
        conf      = round(random.uniform(*scene[3]), 3)
        is_violent = cls in VIOLENCE_CLASSES

        # Alert fires if: violent AND >=5 consecutive violent frames AND cooldown passed
        alert = False
        if is_violent and (elapsed - self._last_alert_sec) > 25:
            # probabilistic: fire alert near the designated alert seconds
            for a_sec in ALERT_SECONDS:
                if abs(elapsed - a_sec) < 0.8:
                    alert = True
                    self._last_alert_sec = elapsed
                    break

        # Bounding box: randomise slightly around a central region
        cx = random.randint(200, 440)
        cy = random.randint(100, 260)
        w  = random.randint(80, 160)
        h  = random.randint(80, 160)
        bbox = (cx - w//2, cy - h//2, cx + w//2, cy + h//2)

        return DetectionFrame(
            frame_id        = self._frame_id,
            scenario_sec    = elapsed,
            detected_class  = cls,
            confidence      = conf,
            is_violent      = is_violent,
            alert_triggered = alert,
            bbox            = bbox,
        )

    @staticmethod
    def _get_scene(t: float) -> tuple:
        for scene in SCENARIO:
            if scene[0] <= t < scene[1]:
                return scene
        return SCENARIO[-1]

    @property
    def scenario_progress(self) -> float:
        """0–1 progress through the 60-second scenario loop."""
        return ((time.time() - self._t0) % SCENARIO_DURATION) / SCENARIO_DURATION

    @property
    def fps(self) -> float:
        elapsed = time.time() - self._t0
        return self._frame_id / elapsed if elapsed > 0 else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_id


# ── Synthetic Frame Renderer ──────────────────────────────────────────────────

def render_frame(det: DetectionFrame, width: int = 640, height: int = 360) -> np.ndarray:
    """
    Render a synthetic surveillance-style frame with detection overlay.
    Returns a BGR numpy array.
    """
    import cv2

    # ── Background: dark grey with subtle grid (CCTV aesthetic) ──────────────
    frame = np.full((height, width, 3), 18, dtype=np.uint8)

    # Grid lines
    for x in range(0, width,  40):
        cv2.line(frame, (x, 0), (x, height), (25, 25, 25), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y),  (25, 25, 25), 1)

    # Simulated "scene" silhouettes — static figures
    _draw_scene_silhouettes(frame, det)

    # ── Status banner ─────────────────────────────────────────────────────────
    if det.alert_triggered:
        banner = (0, 0, 180)
        status_text = "!! VIOLENCE DETECTED !!"
    elif det.is_violent:
        banner = (0, 80, 200)
        status_text = "SUSPICIOUS ACTIVITY"
    else:
        banner = (20, 80, 20)
        status_text = "MONITORING"

    cv2.rectangle(frame, (0, 0), (width, 34), banner, -1)
    cv2.putText(frame, status_text, (10, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (width - 210, 24),
                cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)

    # ── Bounding box ──────────────────────────────────────────────────────────
    if det.is_violent:
        x1, y1, x2, y2 = det.bbox
        colour = CLASS_COLORS.get(det.detected_class, (0, 0, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        label = f"{det.detected_class}  {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── Alert flash overlay ───────────────────────────────────────────────────
    if det.alert_triggered:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.rectangle(frame, (2, 2), (width - 2, height - 2), (0, 0, 200), 3)

    # ── Corner info ───────────────────────────────────────────────────────────
    cv2.putText(frame, "CAM-01  |  DEMO MODE", (10, height - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (80, 80, 80), 1)
    cv2.putText(frame, f"FRM {det.frame_id:06d}", (width - 130, height - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (80, 80, 80), 1)

    return frame


def _draw_scene_silhouettes(frame: np.ndarray, det: DetectionFrame) -> None:
    """Draw simple stick-figure silhouettes that animate based on violence state."""
    import cv2

    cx, cy = 320, 200
    t = det.scenario_sec

    if det.is_violent:
        # Two figures in confrontation
        offset = int(10 * np.sin(t * 8))   # shaking animation

        # Figure 1
        f1x = cx - 60 + offset
        cv2.circle(frame, (f1x, cy - 50), 14, (60, 60, 60), -1)
        cv2.line(frame, (f1x, cy - 36), (f1x, cy + 20), (60, 60, 60), 4)
        cv2.line(frame, (f1x - 20, cy - 10), (f1x + 20, cy - 10), (60, 60, 60), 3)
        cv2.line(frame, (f1x, cy + 20), (f1x - 15, cy + 55), (60, 60, 60), 4)
        cv2.line(frame, (f1x, cy + 20), (f1x + 15, cy + 55), (60, 60, 60), 4)

        # Figure 2
        f2x = cx + 60 - offset
        cv2.circle(frame, (f2x, cy - 50), 14, (55, 55, 55), -1)
        cv2.line(frame, (f2x, cy - 36), (f2x, cy + 20), (55, 55, 55), 4)
        cv2.line(frame, (f2x - 20, cy - 10), (f2x + 20, cy - 10), (55, 55, 55), 3)
        cv2.line(frame, (f2x, cy + 20), (f2x - 15, cy + 55), (55, 55, 55), 4)
        cv2.line(frame, (f2x, cy + 20), (f2x + 15, cy + 55), (55, 55, 55), 4)

    else:
        # Single calm walking figure
        walk = int(5 * np.sin(t * 3))
        cv2.circle(frame, (cx, cy - 50), 14, (50, 50, 50), -1)
        cv2.line(frame, (cx, cy - 36), (cx, cy + 20), (50, 50, 50), 4)
        cv2.line(frame, (cx - 18, cy - 8 + walk), (cx + 18, cy - 8 - walk), (50, 50, 50), 3)
        cv2.line(frame, (cx, cy + 20), (cx - 12, cy + 55 + walk), (50, 50, 50), 4)
        cv2.line(frame, (cx, cy + 20), (cx + 12, cy + 55 - walk), (50, 50, 50), 4)


# ── Pre-seeded Alert History ──────────────────────────────────────────────────

def generate_alert_history(n: int = 38) -> List[AlertRecord]:
    """Generate realistic alert history for the past 48 hours."""
    random.seed(42)
    now    = datetime.now()
    alerts = []

    class_weights = [0.45, 0.30, 0.25]   # Fighting, Weapon, Aggression

    for i in range(n):
        hours_ago = random.uniform(0.2, 47)
        ts        = now - timedelta(hours=hours_ago)
        cls       = random.choices(VIOLENCE_CLASSES, weights=class_weights)[0]
        conf      = round(random.uniform(0.62, 0.97), 3)

        alerts.append(AlertRecord(
            id             = n - i,
            timestamp      = ts,
            detected_class = cls,
            confidence     = conf,
            location       = random.choice(["Camera-01", "Camera-02", "Camera-03", "Camera-04"]),
            email_sent     = random.random() > 0.05,
            whatsapp_sent  = random.random() > 0.08,
        ))

    return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
