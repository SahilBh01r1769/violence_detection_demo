# 🛡️ VisionGuard AI — Portfolio Demo

> **Live Demo:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

A real-time AI violence detection system built with YOLOv8, OpenCV, FastAPI, and Streamlit.  
This repository contains the **interactive portfolio demo** — no camera or GPU required.

---

## 🚀 Deploy to Streamlit Cloud (Free, 2 minutes)

1. **Fork this repo** on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your forked repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** → get your shareable URL

That's it. No secrets, no environment variables needed for the demo.

---

## 💻 Run Locally

```bash
git clone https://github.com/yourusername/visionguard-demo.git
cd visionguard-demo

pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

---

## 🎬 What the Demo Shows

| Page | Description |
|------|-------------|
| **⬛ Live Monitor** | Simulated camera feed with real-time detection overlay, scenario progress, and live alerts |
| **🚨 Alert History** | Filterable table of 38 pre-seeded alerts + any live detections |
| **📊 Analytics** | Plotly charts: alerts over time, class distribution, confidence histogram, camera breakdown |
| **⚙️ Settings** | Configure confidence threshold, cooldown, notifications (UI only in demo) |
| **ℹ️ About** | Architecture, tech stack, project overview |

### Demo Scenario (60-second loop)
```
0–8s   → Normal monitoring
8–15s  → Aggression escalates → Fighting → 🚨 ALERT
15–22s → Returns to normal
22–27s → Weapon detected → 🚨 ALERT  
27–35s → Returns to normal
35–42s → Aggression → Fighting → 🚨 ALERT
42–60s → Normal
```

---

## 🏗️ Full Production System

The full production codebase (with live camera, YOLOv8 inference, real Email/WhatsApp alerts, FastAPI backend, and Docker deployment) is in a separate repository:

👉 [github.com/yourusername/visionguard-ai](https://github.com/yourusername/visionguard-ai)

---

## 🛠️ Tech Stack

- **Detection:** YOLOv8 (Ultralytics) fine-tuned on violence datasets
- **Vision:** OpenCV, NumPy
- **Dashboard:** Streamlit + Plotly
- **Alerts:** Twilio WhatsApp + SMTP Email
- **Backend:** FastAPI
- **Deploy:** Docker, AWS EC2, Streamlit Cloud

---

**Author:** Sahil Bhoir · June 2026
