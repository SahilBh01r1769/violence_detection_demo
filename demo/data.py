"""
demo/data.py — Realistic mock data engine for the portfolio demo.

Generates:
  - A stream of DetectionFrame objects simulating real YOLO output
  - Pre-seeded alert history (last 48 hours)
  - Real video frames fetched from a public-domain clip, with CCTV overlay

Video source: free-to-use Pexels crowd/fight clips loaded via URL.
Falls back to the synthetic grid background if the video cannot be fetched.
"""

from __future__ import annotations

import random
import time
import urllib.request
import tempfile
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

VIOLENCE_CLASSES = ["Fighting", "Weapon", "Aggression"]
ALL_CLASSES      = ["Normal", "Fighting", "Weapon", "Aggression"]

CLASS_COLORS = {
    "Fighting":   (0,  50,  220),
    "Weapon":     (0,   0,  180),
    "Aggression": (20, 80,  200),
    "Normal":     (30, 160,  30),
}

# ── Public-domain video clips (Pexels / Pixabay free licence) ─────────────────
# We use a crowd/street scene for the "normal" phase and a physical-altercation
# clip for the "violent" phase.  Both are royalty-free, no attribution required
# for demo use.  Replace these URLs with any other openly-licensed MP4 if these
# ever go offline.
VIDEO_URLS = {
    "normal":  "https://videos.pexels.com/video-files/855564/855564-hd_1280_720_25fps.mp4",
    "violent": "https://videos.pexels.com/video-files/4763825/4763825-uhd_2560_1440_24fps.mp4",
}

# Local cache paths (written to /tmp so they survive the Streamlit session)
_CACHE_DIR   = Path(tempfile.gettempdir()) / "visionguard_demo"
_CACHE_NORMAL  = _CACHE_DIR / "normal.mp4"
_CACHE_VIOLENT = _CACHE_DIR / "violent.mp4"

# Scenario timeline: (start_sec, end_sec, class, confidence_range, video_key)
SCENARIO: list[tuple] = [
    (0,   8,  "Normal",     (0.80, 0.95), "normal"),
    (8,   9,  "Aggression", (0.56, 0.65), "violent"),
    (9,   11, "Aggression", (0.70, 0.82), "violent"),
    (11,  14, "Fighting",   (0.78, 0.91), "violent"),
    (14,  15, "Fighting",   (0.88, 0.96), "violent"),   # ← alert fires
    (15,  22, "Normal",     (0.82, 0.94), "normal"),
    (22,  24, "Weapon",     (0.60, 0.72), "violent"),
    (24,  27, "Weapon",     (0.75, 0.89), "violent"),   # ← alert fires
    (27,  35, "Normal",     (0.85, 0.97), "normal"),
    (35,  37, "Aggression", (0.55, 0.68), "violent"),
    (37,  39, "Fighting",   (0.72, 0.84), "violent"),
    (39,  42, "Fighting",   (0.83, 0.93), "violent"),   # ← alert fires
    (42,  60, "Normal",     (0.88, 0.98), "normal"),
]

SCENARIO_DURATION = 60
ALERT_SECONDS     = {14, 26, 41}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DetectionFrame:
    frame_id:        int
    scenario_sec:    float
    detected_class:  str
    confidence:      float
    is_violent:      bool
    alert_triggered: bool
    bbox:            tuple   # x1, y1, x2, y2


@dataclass
class AlertRecord:
    id:              int
    timestamp:       datetime
    detected_class:  str
    confidence:      float
    location:        str
    email_sent:      bool
    whatsapp_sent:   bool


# ── Video cache / loader ──────────────────────────────────────────────────────

class _VideoCache:
    """
    Downloads and caches the two video clips in background threads.
    Provides read_frame(key) → BGR numpy array at any time;
    returns None (triggering fallback) until the download completes.
    """

    def __init__(self):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._caps:   dict = {}      # key → cv2.VideoCapture
        self._ready:  dict = {"normal": False, "violent": False}
        self._lock          = threading.Lock()

        for key, url in VIDEO_URLS.items():
            cache_path = _CACHE_DIR / f"{key}.mp4"
            t = threading.Thread(
                target=self._download_and_open,
                args=(key, url, cache_path),
                daemon=True,
            )
            t.start()

    def _download_and_open(self, key: str, url: str, path: Path) -> None:
        import cv2
        try:
            if not path.exists():
                urllib.request.urlretrieve(url, str(path))
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                with self._lock:
                    self._caps[key]  = cap
                    self._ready[key] = True
        except Exception:
            pass   # silently fall back to synthetic frames

    def read_frame(self, key: str, width: int = 640, height: int = 360) -> Optional[np.ndarray]:
        import cv2
        with self._lock:
            if not self._ready.get(key):
                return None
            cap = self._caps[key]

        ret, frame = cap.read()
        if not ret:
            # Loop back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None

        frame = cv2.resize(frame, (width, height))
        # Apply a subtle desaturation + contrast boost for the CCTV look
        frame = _cctv_grade(frame)
        return frame

    def is_ready(self, key: str) -> bool:
        return self._ready.get(key, False)


def _cctv_grade(frame: np.ndarray) -> np.ndarray:
    """Desaturate slightly and add a faint scanline vignette for CCTV realism."""
    import cv2
    # Reduce saturation to ~70 %
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.65
    hsv[:, :, 2]  = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Scanlines (every other row darkened by 15 %)
    frame[::2] = (frame[::2] * 0.85).astype(np.uint8)

    # Slight vignette
    h, w = frame.shape[:2]
    Y, X  = np.ogrid[:h, :w]
    mask  = 1 - 0.45 * ((X - w/2)**2 + (Y - h/2)**2) / ((w/2)**2 + (h/2)**2)
    frame = np.clip(frame * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)

    return frame


# Module-level singleton — initialised once per Streamlit session
_video_cache: Optional[_VideoCache] = None


def get_video_cache() -> _VideoCache:
    global _video_cache
    if _video_cache is None:
        _video_cache = _VideoCache()
    return _video_cache


# ── Demo Stream ───────────────────────────────────────────────────────────────

class DemoStream:
    def __init__(self, fps: int = 20):
        self._fps             = fps
        self._frame_id        = 0
        self._t0              = time.time()
        self._last_alert_sec: float = -999

    def next_frame(self) -> DetectionFrame:
        self._frame_id += 1
        elapsed = (time.time() - self._t0) % SCENARIO_DURATION
        scene   = self._get_scene(elapsed)
        cls     = scene[2]
        conf    = round(random.uniform(*scene[3]), 3)
        is_violent = cls in VIOLENCE_CLASSES

        alert = False
        if is_violent and (elapsed - self._last_alert_sec) > 25:
            for a_sec in ALERT_SECONDS:
                if abs(elapsed - a_sec) < 0.8:
                    alert = True
                    self._last_alert_sec = elapsed
                    break

        cx = random.randint(200, 440)
        cy = random.randint(80,  260)
        w  = random.randint(100, 180)
        h  = random.randint(120, 200)
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
        return ((time.time() - self._t0) % SCENARIO_DURATION) / SCENARIO_DURATION

    @property
    def fps(self) -> float:
        elapsed = time.time() - self._t0
        return self._frame_id / elapsed if elapsed > 0 else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_id


# ── Frame Renderer ────────────────────────────────────────────────────────────

def render_frame(det: DetectionFrame, width: int = 640, height: int = 360) -> np.ndarray:
    """
    Compose a display frame:
      1. Try to get a real video frame for the current scene (normal / violent clip)
      2. Fall back to the synthetic grid background if video isn't loaded yet
      3. Apply CCTV overlay (status banner, bbox, timestamp, corner text)
    """
    import cv2

    cache     = get_video_cache()
    scene_key = _scene_key_for(det.scenario_sec)
    frame     = cache.read_frame(scene_key, width, height)

    if frame is None:
        # ── Fallback: synthetic dark grid ────────────────────────────────────
        frame = np.full((height, width, 3), 18, dtype=np.uint8)
        for x in range(0, width,  40):
            cv2.line(frame, (x, 0), (x, height), (28, 28, 28), 1)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y),  (28, 28, 28), 1)
        # Loading message
        msg = "Loading video..." if not cache.is_ready(scene_key) else "Video unavailable"
        cv2.putText(frame, msg, (width//2 - 80, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)

    # ── Status banner ─────────────────────────────────────────────────────────
    if det.alert_triggered:
        banner, status_text = (0, 0, 170), "!! VIOLENCE DETECTED !!"
    elif det.is_violent:
        banner, status_text = (0, 70, 190), "SUSPICIOUS ACTIVITY"
    else:
        banner, status_text = (15, 70, 15), "MONITORING"

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
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.rectangle(frame, (2, 2), (width - 2, height - 2), (0, 0, 200), 3)

    # ── Corner HUD ───────────────────────────────────────────────────────────
    cv2.putText(frame, "CAM-01  |  DEMO MODE", (10, height - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (80, 80, 80), 1)
    cv2.putText(frame, f"FRM {det.frame_id:06d}", (width - 130, height - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (80, 80, 80), 1)

    return frame


def _scene_key_for(t: float) -> str:
    for scene in SCENARIO:
        if scene[0] <= t < scene[1]:
            return scene[4]
    return "normal"


# ── Pre-seeded Alert History ──────────────────────────────────────────────────

def generate_alert_history(n: int = 38) -> List[AlertRecord]:
    random.seed(42)
    now    = datetime.now()
    alerts = []
    class_weights = [0.45, 0.30, 0.25]
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
