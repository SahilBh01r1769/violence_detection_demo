"""
app.py — Violence Detection System: Portfolio Demo
Author: Sahil Bhoir

Self-contained demo — no camera, no GPU, no API keys required.
Runs entirely in Streamlit with simulated detections.
"""

from __future__ import annotations

import base64
import io
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Bootstrap demo data module path ──────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from demo.data import DemoStream, AlertRecord, generate_alert_history, render_frame

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VisionGuard AI — Demo",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700;800&family=Barlow+Condensed:wght@700;800&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: #080b0f;
    color: #c8d6e5;
    font-family: 'Barlow', sans-serif;
  }
  [data-testid="stSidebar"] {
    background: #0c1017 !important;
    border-right: 1px solid #1a2535;
  }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  h1,h2,h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 1px; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0e1620;
    border: 1px solid #1c2e45;
    border-radius: 8px;
    padding: 14px 18px;
  }
  [data-testid="metric-container"] label { color: #5a7a99 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8f4f8 !important; font-family: 'Share Tech Mono', monospace; font-size: 26px !important; }

  /* Feed container */
  .feed-wrap {
    border: 1px solid #1c2e45;
    border-radius: 10px;
    overflow: hidden;
    background: #060a0e;
    position: relative;
  }
  .feed-wrap img { display: block; width: 100%; border-radius: 0; }

  /* Alert pill */
  .pill-violent {
    display: inline-block; background: #c0392b; color: #fff;
    border-radius: 4px; padding: 2px 10px; font-size: 11px;
    font-weight: 700; letter-spacing: .5px; font-family: 'Share Tech Mono', monospace;
  }
  .pill-normal {
    display: inline-block; background: #1a4a2a; color: #4ecb71;
    border: 1px solid #2a6a3a; border-radius: 4px; padding: 2px 10px;
    font-size: 11px; font-weight: 700; letter-spacing: .5px; font-family: 'Share Tech Mono', monospace;
  }
  .pill-warn {
    display: inline-block; background: #7a3a00; color: #ffaa44;
    border-radius: 4px; padding: 2px 10px; font-size: 11px;
    font-weight: 700; letter-spacing: .5px; font-family: 'Share Tech Mono', monospace;
  }

  /* Section header */
  .sec-head {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 17px; font-weight: 800; letter-spacing: 2px;
    text-transform: uppercase; color: #3a8fd1;
    border-bottom: 1px solid #1c2e45; padding-bottom: 6px; margin-bottom: 16px;
  }

  /* Alert row cards */
  .alert-row {
    background: #0e1620; border: 1px solid #1c2e45; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 8px;
    transition: border-color .2s;
  }
  .alert-row:hover { border-color: #3a8fd1; }

  /* Status indicator */
  .status-active { color: #4ecb71; font-size: 13px; }
  .status-idle   { color: #c0392b; font-size: 13px; }

  /* Timeline bar */
  .timeline-bar {
    background: #0e1620; border: 1px solid #1c2e45; border-radius: 6px;
    height: 8px; overflow: hidden; margin-top: 4px;
  }
  .timeline-fill { background: linear-gradient(90deg, #1a6fa8, #3ab8f5); height: 100%; transition: width .3s; }

  /* Tab styling */
  [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1c2e45; }
  [data-baseweb="tab"]      { color: #5a7a99 !important; font-family: 'Barlow Condensed', sans-serif; font-size: 15px; letter-spacing: 1px; }
  [aria-selected="true"]    { color: #3ab8f5 !important; border-bottom: 2px solid #3ab8f5 !important; }

  /* Demo banner */
  .demo-banner {
    background: linear-gradient(135deg, #0a1a2e, #0d2240);
    border: 1px solid #1c4a7a; border-radius: 10px;
    padding: 10px 18px; margin-bottom: 18px;
    font-size: 13px; color: #5a9fd4; letter-spacing: .3px;
  }
  .demo-banner strong { color: #3ab8f5; }

  /* Plotly chart bg */
  .js-plotly-plot { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────

if "stream" not in st.session_state:
    st.session_state.stream          = DemoStream(fps=20)
    st.session_state.alert_history   = generate_alert_history(38)
    st.session_state.live_alerts     = []
    st.session_state.total_frames    = 0
    st.session_state.session_alerts  = 0
    st.session_state.running         = True
    st.session_state.confidence_thr  = 0.55
    st.session_state.cooldown        = 30

stream: DemoStream     = st.session_state.stream
history: list[AlertRecord] = st.session_state.alert_history


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 12px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:26px;font-weight:800;
                  letter-spacing:2px;color:#3ab8f5;">🛡 VISIONGUARD</div>
      <div style="font-size:11px;color:#3a5a7a;letter-spacing:2px;text-transform:uppercase;">
        AI Surveillance System
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "nav",
        ["⬛  Live Monitor", "🚨  Alert History", "📊  Analytics", "⚙️  Settings", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System status
    is_running = st.session_state.running
    status_html = (
        '<span class="status-active">● SYSTEM ACTIVE</span>'
        if is_running else
        '<span class="status-idle">● SYSTEM PAUSED</span>'
    )
    st.markdown(status_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Resume" if not is_running else "⏸ Pause",
                     use_container_width=True, type="primary"):
            st.session_state.running = not st.session_state.running
            st.rerun()
    with col2:
        if st.button("↺ Reset", use_container_width=True):
            for k in ["stream", "alert_history", "live_alerts",
                      "total_frames", "session_alerts"]:
                del st.session_state[k]
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:#3a5a7a;line-height:1.8;">
      <div>CONF THRESHOLD &nbsp; <span style="color:#3ab8f5;">{st.session_state.confidence_thr:.0%}</span></div>
      <div>COOLDOWN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#3ab8f5;">{st.session_state.cooldown}s</span></div>
      <div>CAMERA &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#3ab8f5;">DEMO MODE</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#2a3a4a;text-align:center;">Built by Sahil Bhoir · June 2026</div>', unsafe_allow_html=True)


# ── Helper: frame → base64 ────────────────────────────────────────────────────

def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


# ── PLOTLY THEME ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1620",
    plot_bgcolor="#0a1220",
    font=dict(color="#5a7a99", family="Barlow, sans-serif", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1a2535", linecolor="#1a2535"),
    yaxis=dict(gridcolor="#1a2535", linecolor="#1a2535"),
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

if "Live Monitor" in page:

    st.markdown('<div class="demo-banner">🎬 <strong>DEMO MODE</strong> — Simulated detections replay a realistic 60-second surveillance scenario. No camera or GPU required.</div>', unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("System",         "🟢 ACTIVE" if st.session_state.running else "⏸ PAUSED")
    k2.metric("Frames",         f'{stream.frame_count:,}')
    k3.metric("Session Alerts", st.session_state.session_alerts)
    k4.metric("Live FPS",       f'{stream.fps:.1f}')
    k5.metric("Total Alerts",   len(history) + st.session_state.session_alerts)

    st.markdown("---")

    # ── Main layout ───────────────────────────────────────────────────────────
    feed_col, panel_col = st.columns([3, 1], gap="medium")

    feed_slot   = feed_col.empty()
    prog_slot   = feed_col.empty()

    with panel_col:
        st.markdown('<div class="sec-head">Detection</div>', unsafe_allow_html=True)
        det_class_slot = st.empty()
        det_conf_slot  = st.empty()
        det_frame_slot = st.empty()

        st.markdown('<div class="sec-head" style="margin-top:18px;">Last 5 Alerts</div>', unsafe_allow_html=True)
        recent_slot = st.empty()

    # ── Tick ─────────────────────────────────────────────────────────────────
    if st.session_state.running:
        det = stream.next_frame()
        st.session_state.total_frames += 1

        # Render synthetic frame
        frame = render_frame(det)
        b64   = frame_to_b64(frame)

        feed_slot.markdown(
            f'<div class="feed-wrap"><img src="data:image/jpeg;base64,{b64}"></div>',
            unsafe_allow_html=True,
        )

        # Scenario progress bar
        prog = stream.scenario_progress
        prog_slot.markdown(f"""
        <div style="margin-top:8px;">
          <div style="font-size:10px;color:#3a5a7a;letter-spacing:1px;margin-bottom:3px;">
            SCENARIO LOOP &nbsp; {prog*100:.0f}%
          </div>
          <div class="timeline-bar"><div class="timeline-fill" style="width:{prog*100:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # Detection panel
        if det.is_violent:
            pill = f'<span class="pill-violent">{det.detected_class}</span>'
        else:
            pill = '<span class="pill-normal">Normal</span>'

        det_class_slot.markdown(f"**Class** &nbsp; {pill}", unsafe_allow_html=True)
        det_conf_slot.markdown(f"**Confidence** &nbsp; `{det.confidence:.1%}`")
        det_frame_slot.markdown(f"**Frame** &nbsp; `#{det.frame_id:,}`")

        # Alert triggered
        if det.alert_triggered:
            st.session_state.session_alerts += 1
            new_alert = AlertRecord(
                id             = len(history) + st.session_state.session_alerts,
                timestamp      = datetime.now(),
                detected_class = det.detected_class,
                confidence     = det.confidence,
                location       = "Camera-01",
                email_sent     = True,
                whatsapp_sent  = True,
            )
            st.session_state.live_alerts.insert(0, new_alert)
            st.toast(f"🚨 Alert! {det.detected_class} ({det.confidence:.0%})", icon="🚨")

        # Recent alerts list
        live  = st.session_state.live_alerts[:5]
        hist5 = history[:max(0, 5 - len(live))]
        combined = (live + hist5)[:5]

        rows_html = ""
        for a in combined:
            ts = a.timestamp.strftime("%H:%M:%S")
            rows_html += f"""
            <div class="alert-row">
              <div style="font-size:11px;color:#3a5a7a;">{ts}</div>
              <div style="font-size:13px;font-weight:600;color:#c8d6e5;margin:2px 0;">
                {a.detected_class}
              </div>
              <div style="font-size:11px;color:#5a7a99;">{a.confidence:.0%} · {a.location}</div>
            </div>"""
        recent_slot.markdown(rows_html or "<div style='color:#3a5a7a;font-size:12px;'>No alerts yet</div>",
                             unsafe_allow_html=True)

        time.sleep(0.08)   # ~12 FPS refresh
        st.rerun()

    else:
        feed_slot.info("System paused. Press ▶ Resume in the sidebar to continue.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ALERT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif "Alert History" in page:
    st.markdown('<div class="sec-head">Alert History</div>', unsafe_allow_html=True)

    live   = st.session_state.live_alerts
    all_alerts = (live + history)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        classes    = ["All"] + sorted({a.detected_class for a in all_alerts})
        cls_filter = st.selectbox("Class", classes)
    with fc2:
        cams    = ["All"] + sorted({a.location for a in all_alerts})
        cam_filter = st.selectbox("Camera", cams)
    with fc3:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = [
        a for a in all_alerts
        if (cls_filter == "All" or a.detected_class == cls_filter)
        and (cam_filter == "All" or a.location == cam_filter)
        and a.confidence >= min_conf
    ]

    st.caption(f"Showing {len(filtered)} of {len(all_alerts)} alerts")
    st.markdown("---")

    # Table
    for a in filtered[:60]:
        ts   = a.timestamp.strftime("%Y-%m-%d  %H:%M:%S")
        pill = (f'<span class="pill-violent">{a.detected_class}</span>'
                if a.detected_class == "Fighting" else
                f'<span class="pill-warn">{a.detected_class}</span>')
        live_tag = ' <span style="font-size:10px;color:#3ab8f5;">● LIVE</span>' if a in live else ""

        with st.expander(f"Alert #{a.id}  ·  {a.detected_class}  ·  {a.confidence:.0%}  ·  {ts}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Class** {pill}{live_tag}", unsafe_allow_html=True)
            c2.markdown(f"**Confidence** `{a.confidence:.1%}`")
            c3.markdown(f"**Location** `{a.location}`")
            c4, c5 = st.columns(2)
            c4.markdown(f"**Email** {'✅ Sent' if a.email_sent else '❌ Failed'}")
            c5.markdown(f"**WhatsApp** {'✅ Sent' if a.whatsapp_sent else '❌ Failed'}")
            st.caption(f"Timestamp: {ts}")

    # CSV export
    st.markdown("---")
    df_export = pd.DataFrame([{
        "id": a.id, "timestamp": a.timestamp, "class": a.detected_class,
        "confidence": a.confidence, "location": a.location,
        "email_sent": a.email_sent, "whatsapp_sent": a.whatsapp_sent,
    } for a in filtered])
    st.download_button(
        "⬇️ Export CSV",
        data=df_export.to_csv(index=False),
        file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif "Analytics" in page:
    st.markdown('<div class="sec-head">Analytics</div>', unsafe_allow_html=True)

    all_alerts = st.session_state.live_alerts + history
    df = pd.DataFrame([{
        "timestamp":      a.timestamp,
        "detected_class": a.detected_class,
        "confidence":     a.confidence,
        "location":       a.location,
        "email_sent":     a.email_sent,
        "whatsapp_sent":  a.whatsapp_sent,
    } for a in all_alerts])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Alerts",      len(df))
    k2.metric("Avg Confidence",    f'{df["confidence"].mean():.1%}')
    k3.metric("Most Common Class", df["detected_class"].mode()[0])
    k4.metric("Email Success",     f'{df["email_sent"].mean():.0%}')

    st.markdown("---")

    # Alerts over time
    st.markdown("**Alerts Over Time (hourly)**")
    df["hour"] = df["timestamp"].dt.floor("H")
    time_df    = df.groupby(["hour", "detected_class"]).size().reset_index(name="count")
    fig_time = px.bar(
        time_df, x="hour", y="count", color="detected_class",
        color_discrete_map={"Fighting": "#e74c3c", "Weapon": "#e67e22", "Aggression": "#f39c12"},
        template="plotly_dark",
    )
    fig_time.update_layout(**PLOTLY_LAYOUT, barmode="stack", showlegend=True,
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_time, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Class Distribution**")
        cd = df["detected_class"].value_counts().reset_index()
        cd.columns = ["Class", "Count"]
        fig_pie = px.pie(cd, names="Class", values="Count",
                         color_discrete_sequence=["#e74c3c", "#e67e22", "#f39c12"],
                         template="plotly_dark", hole=0.45)
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        fig_pie.update_traces(textfont_color="#c8d6e5")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("**Confidence Distribution**")
        fig_hist = px.histogram(df, x="confidence", nbins=20,
                                color_discrete_sequence=["#3ab8f5"],
                                template="plotly_dark")
        fig_hist.update_layout(**PLOTLY_LAYOUT,
                               xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Camera breakdown
    st.markdown("**Alerts by Camera**")
    cam_df = df.groupby(["location", "detected_class"]).size().reset_index(name="count")
    fig_cam = px.bar(cam_df, x="location", y="count", color="detected_class",
                     color_discrete_map={"Fighting": "#e74c3c", "Weapon": "#e67e22", "Aggression": "#f39c12"},
                     template="plotly_dark")
    fig_cam.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                          legend=dict(orientation="h", y=1.1),
                          xaxis_title="", yaxis_title="Alerts")
    st.plotly_chart(fig_cam, use_container_width=True)

    # Notification success
    st.markdown("**Notification Delivery**")
    notif = pd.DataFrame({
        "Channel": ["Email", "WhatsApp"],
        "Delivered": [df["email_sent"].sum(), df["whatsapp_sent"].sum()],
        "Failed":    [(~df["email_sent"]).sum(), (~df["whatsapp_sent"]).sum()],
    })
    fig_n = px.bar(notif, x="Channel", y=["Delivered", "Failed"], barmode="group",
                   color_discrete_map={"Delivered": "#2ecc71", "Failed": "#e74c3c"},
                   template="plotly_dark")
    fig_n.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_n, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

elif "Settings" in page:
    st.markdown('<div class="sec-head">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="demo-banner">⚠️ <strong>DEMO MODE</strong> — Settings here update the UI only. In production, these are applied to the live pipeline via the FastAPI backend.</div>', unsafe_allow_html=True)

    with st.form("settings_form"):
        st.markdown("**Detection Parameters**")
        s1, s2 = st.columns(2)
        with s1:
            conf = st.slider("Confidence Threshold", 0.1, 1.0,
                             st.session_state.confidence_thr, 0.05)
            cooldown = st.number_input("Alert Cooldown (seconds)", 5, 300,
                                       st.session_state.cooldown)
        with s2:
            frame_con = st.number_input("Frame Consistency", 1, 30, 5,
                help="Consecutive violent frames before alert fires")
            location  = st.text_input("Camera Label", "Camera-01")

        st.markdown("---")
        st.markdown("**Notifications**")
        n1, n2 = st.columns(2)
        with n1:
            en_email = st.checkbox("Email Alerts", True)
            smtp_user = st.text_input("SMTP User", placeholder="your@gmail.com")
        with n2:
            en_wa = st.checkbox("WhatsApp Alerts", True)
            wa_to = st.text_input("WhatsApp To", placeholder="whatsapp:+91XXXXXXXXXX")

        submitted = st.form_submit_button("💾 Save Settings", type="primary")

    if submitted:
        st.session_state.confidence_thr = conf
        st.session_state.cooldown       = cooldown
        st.success("✅ Settings saved!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif "About" in page:
    st.markdown('<div class="sec-head">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width: 720px; line-height: 1.8; color: #8aabbf;">

    <h3 style="color:#3ab8f5;font-family:'Barlow Condensed',sans-serif;letter-spacing:1px;">
      VisionGuard AI — Real-Time Violence Detection System
    </h3>

    <p>
      VisionGuard is an end-to-end AI surveillance system that detects violent activities
      from live video streams and automatically dispatches multi-channel alerts to
      security personnel — all within 5 seconds of confirmed detection.
    </p>

    <h4 style="color:#c8d6e5;margin-top:24px;">Architecture</h4>
    </div>
    """, unsafe_allow_html=True)

    arch_cols = st.columns(5)
    layers = [
        ("📹", "Input",      "Webcam / RTSP\nIP Camera"),
        ("🧠", "Detection",  "YOLOv8\nFine-tuned"),
        ("⚖️", "Decision",   "Temporal\nConsistency"),
        ("📸", "Action",     "Screenshot\n+ Logging"),
        ("🔔", "Notify",     "Email\nWhatsApp"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, layers):
        col.markdown(f"""
        <div style="background:#0e1620;border:1px solid #1c2e45;border-radius:8px;
                    padding:14px 10px;text-align:center;">
          <div style="font-size:24px;">{icon}</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                      font-weight:700;color:#3ab8f5;letter-spacing:1px;margin:6px 0 4px;">{title}</div>
          <div style="font-size:11px;color:#3a5a7a;white-space:pre-line;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:720px;margin-top:24px;line-height:1.8;color:#8aabbf;">

    <h4 style="color:#c8d6e5;">Tech Stack</h4>
    <table style="border-collapse:collapse;width:100%;font-size:13px;">
      <tr style="border-bottom:1px solid #1c2e45;">
        <td style="padding:8px 12px;color:#5a7a99;">Object Detection</td>
        <td style="padding:8px 12px;color:#c8d6e5;">YOLOv8 / YOLOv11 (Ultralytics)</td>
      </tr>
      <tr style="border-bottom:1px solid #1c2e45;">
        <td style="padding:8px 12px;color:#5a7a99;">Computer Vision</td>
        <td style="padding:8px 12px;color:#c8d6e5;">OpenCV, NumPy</td>
      </tr>
      <tr style="border-bottom:1px solid #1c2e45;">
        <td style="padding:8px 12px;color:#5a7a99;">Backend API</td>
        <td style="padding:8px 12px;color:#c8d6e5;">FastAPI + Uvicorn</td>
      </tr>
      <tr style="border-bottom:1px solid #1c2e45;">
        <td style="padding:8px 12px;color:#5a7a99;">Dashboard</td>
        <td style="padding:8px 12px;color:#c8d6e5;">Streamlit</td>
      </tr>
      <tr style="border-bottom:1px solid #1c2e45;">
        <td style="padding:8px 12px;color:#5a7a99;">Notifications</td>
        <td style="padding:8px 12px;color:#c8d6e5;">Twilio (WhatsApp), SMTP (Email)</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;color:#5a7a99;">Deployment</td>
        <td style="padding:8px 12px;color:#c8d6e5;">Docker · AWS EC2 · Streamlit Cloud</td>
      </tr>
    </table>

    <h4 style="color:#c8d6e5;margin-top:24px;">Key Features</h4>
    <ul>
      <li>Real-time inference at 15–30 FPS from any video source</li>
      <li>Multi-class detection: Fighting, Weapon, Aggression</li>
      <li>Temporal consistency filter to eliminate false positives</li>
      <li>Dual-channel alerts (Email + WhatsApp) within 5 seconds</li>
      <li>Web dashboard with live feed, alert history, and analytics</li>
      <li>Configurable confidence threshold and alert cooldown</li>
    </ul>

    <h4 style="color:#c8d6e5;margin-top:24px;">Author</h4>
    <p>
      <strong style="color:#3ab8f5;">Sahil Bhoir</strong> — Computer Vision &amp; Deep Learning<br>
      <span style="font-size:12px;">June 2026 · Domain: CV · Deep Learning · Real-time AI Systems</span>
    </p>

    </div>
    """, unsafe_allow_html=True)
